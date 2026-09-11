# -*- coding: utf-8 -*-
r"""streamlit_app.py — 本地 Web 界面（本地测试用；清小搭接入用 FastAPI 层）。

启动: streamlit run streamlit_app.py

对话式 AI 界面（参考 DeepSeek 布局）：
- 侧边栏会话管理：新对话 + 会话列表（悬停 ⋯ 三点菜单：重命名/删除；
  重命名为原位编辑，回车或 ✓ 保存）；
- 主区当前会话消息流（user 问题 + assistant 图文回答，每条回答附
  「复制 Markdown」一键复制按钮）；排版内容列定宽居中、与输入框长度一致；
- 底部输入区：st.chat_input 固定窗格（钉在视口底部，向主流大模型聊天界面
  看齐），整页最先渲染的可见组分（调用提到脚本最前，delta 最先到达前端），
  stBottom 自带钉底层级与不透明背景（不被其他组分遮盖）；
  CSS 固定可见输入盒宽度为原长 80% 并居中（侧边栏收起时长度不变、位置左移）；
  内置 ＋ 图片附件（accept_file，streamlit ≥1.46），回车连续追问；
  旧版 streamlit 回退为 ＋ 弹层上传 + text_input（随内容滚动）；
- 会话持久化到 data/chat_sessions.json（刷新/重启不丢；data/ 已 gitignore）；
  旧版单会话 data/chat_history.json 首次运行时自动迁移为首个会话。

将 assistant 回答（文本 + 内联 TikZ）拆段渲染，**文本先行**：按阅读顺序先输出
全部文本（markdown，`\(...\)`/`\[...\]`/`\ce{...}` 公式先转 KaTeX 可渲染格式）
与「LaTeX 源码」下拉框，TikZ/chemfig 代码段以 st.empty() 占位，全部文本输出后
再逐段编译为 PNG 回填（检测到 LaTeX 引擎时），未装 LaTeX 则回退为
st.code(language="latex")。

已同步管线能力：
- 多轮对话（A3）：提问携带当前会话历史（assistant 历史剥离渲染代码）；
- 生成进度（B2）：st.status 实时显示 LLM 生成草稿（progress_callback）；
- AI 命名（省 token）：首条提问用极简 prompt 生成对话标题；
- P0~P3、A1、B3 均在 process_question 内部生效，界面无需额外处理。
"""

import inspect
import json
import base64
import os
import re
import tempfile
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from app import process_question
from utils.latex_compile import compile_tikz_to_png, detect_backends

# 代码段：tikzpicture 整块 | schemestart 反应式 | 单个 \chemfig{...}
_CODE_RE = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"
    r"|\\schemestart.*?\\schemestop"
    r"|\\chemfig\{(?:[^{}]|\{[^{}]*\})*\}",
    re.DOTALL,
)

# 多轮对话：传给 LLM 的最大历史消息数（最近 N 条）
_MAX_HISTORY = 10

# LaTeX 公式分隔符 → Streamlit markdown（内置 KaTeX）可渲染格式。
# LLM 输出常用 \\(...\\)（行内）/ \\[...\\]（块）包裹公式；markdown 会把
# 反斜杠当转义符吃掉（\\[ → [），导致公式代码裸露显示。转换为 $...$ / $$...$$。
_LATEX_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_LATEX_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

# mhchem \ce{...}（LLM 常用化学式宏，KaTeX 无 mhchem 扩展）：转 KaTeX 兼容。
# 处理嵌套一层花括号（\ce{...} 内一般无嵌套；有则交给外层匹配）。
_CE_RE = re.compile(r"\\ce\{((?:[^{}]|\{[^{}]*\})*)\}")

# 会话文件（data/ 已 gitignore）与旧版单会话文件
_SESSIONS_FILE = Path(__file__).resolve().parent / "data" / "chat_sessions.json"
_LEGACY_FILE = Path(__file__).resolve().parent / "data" / "chat_history.json"

# 诊断流水（data/ 已 gitignore）：每次提问追加一行 JSON —— 时间戳、会话 id、
# 提问原文、原始标记（responses，含模型最终采用的标记文本；**不含渲染后
# TikZ**）、诊断列表（每轮校验/渲染失败的 round/stage/type/raw/reason/
# resolved）。用于质量回溯/统计，不参与页面逻辑。
_DIAGNOSTICS_FILE = (Path(__file__).resolve().parent / "data"
                     / "streamlit_diagnostics.jsonl")
# 一次性清空旧诊断：**进程级**只做一次（原来用 `st.session_state` 判断，那是
# "每个浏览器会话"级别的——每个新访客都会再清一次；日志已独立后影响虽小，
# 但仍按注释原意改成进程级）
_DIAG_FLUSHED = False


def _flush_diagnostics_file() -> None:
    """清空本调试日志（进程内只执行一次，之后 append）。

    ★ 用**独立文件**、不碰 `data/diagnostics.jsonl`（安全审查 R12，20260911）：
    那个文件是**线上**的 /v1（清小搭）与网页共用的诊断流水，而本函数会把它
    **清空**——两者共用一个文件时，跑一次 Streamlit 就会把线上诊断记录抹掉。
    格式也不同（本文件带 responses/diagnostics 列表，diaglog 是逐行
    failure/answer）。Streamlit 只是本地调试界面，日志理应分开。
    """
    global _DIAG_FLUSHED
    if _DIAG_FLUSHED:
        return
    _DIAG_FLUSHED = True
    try:
        _DIAGNOSTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 与 core/diaglog 同口径：日志里有真人提问原文 → 只属主可读（0600）
        fd = os.open(_DIAGNOSTICS_FILE,
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.close(fd)
        os.chmod(_DIAGNOSTICS_FILE, 0o600)
    except OSError as e:
        print(f"[streamlit] 诊断文件初始化失败: {e}")


def _append_diagnostic(session_id: str, question: str,
                       responses: list, diagnostics: list) -> None:
    """把一次提问的诊断记录追加到 diagnostics.jsonl（一行一条 JSON）。

    记录原始标记（responses，未渲染）与诊断失败项，**不记录渲染后含 TikZ
    的 answer**——避免大段 LaTeX 污染、且便于统计模型实际写出的标记。
    """
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "question": question,
            "responses": responses or [],
            "diagnostics": diagnostics or [],
        }
        _DIAGNOSTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _DIAGNOSTICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # 记录失败不拖垮主流程
        print(f"[streamlit] 诊断写入失败: {e}")

# AI 命名标题的极简系统提示（替代完整 system_prompt，省 token）
_TITLE_SYSTEM = ("你是对话标题生成器。根据用户第一条提问提炼一个"
                 "不超过 12 个字的对话标题。只输出标题本身，不要任何"
                 "解释、引号或标点。")
_TITLE_MAX_LEN = 12


def _load_sessions() -> list:
    """加载多会话列表；无文件/损坏返回空。"""
    try:
        data = json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
        sessions = data.get("sessions", []) if isinstance(data, dict) else []
        return [s for s in sessions
                if isinstance(s, dict) and s.get("id") and isinstance(s.get("messages"), list)]
    except (OSError, ValueError):
        return []


def _migrate_legacy_history() -> list:
    """旧版单会话 data/chat_history.json → 首个会话（仅首次迁移用）。"""
    try:
        msgs = json.loads(_LEGACY_FILE.read_text(encoding="utf-8"))
        if isinstance(msgs, list) and msgs:
            return [{"id": uuid.uuid4().hex, "title": "历史对话", "messages": msgs}]
    except (OSError, ValueError):
        pass
    return []


def _save_sessions(sessions: list) -> None:
    """保存会话列表（失败仅打日志，不阻断对话）。"""
    try:
        _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSIONS_FILE.write_text(
            json.dumps({"sessions": sessions}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError as e:
        print(f"[streamlit] 会话保存失败: {e}")


def _new_session(sessions: list, title: str = "新对话") -> dict:
    """新建会话并追加到列表。"""
    s = {"id": uuid.uuid4().hex, "title": title, "messages": []}
    sessions.append(s)
    return s


def _rerun() -> None:
    """触发脚本重跑（兼容新旧 streamlit API）。

    chat_input/text_input 在对话渲染循环**之后**执行，_ask 追加的新消息必须靠
    重跑才能被渲染循环显示；直接重跑即重绘含新消息的完整界面。
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _progress_updater(draft_box, throttle: float = 0.3):
    """构造进度回调：累积 LLM 草稿，节流更新 draft_box 占位（B2）。"""
    state = {"last": 0.0, "buf": []}

    def _cb(piece: str) -> None:
        state["buf"].append(piece)
        now = time.time()
        if now - state["last"] >= throttle:
            state["last"] = now
            text = "".join(state["buf"])
            tail = text[-200:] if len(text) > 200 else text
            draft_box.markdown(f"**生成中…**（已生成 {len(text)} 字）\n\n```\n{tail}\n```")

    return _cb


def _strip_render_code(text: str) -> str:
    """剥离渲染代码（TikZ/chemfig），只留纯文本。"""
    return _CODE_RE.sub("", text)


def _convert_ce_math(body: str) -> str:
    """把公式内容中的 mhchem \\ce{...} 转 KaTeX 兼容语法。

    \\ce{C6H6 + HNO3 ->[H2SO4, \\triangle] C6H5NO2 + H2O} →
    C_6H_6 + HNO_3 \\xrightarrow{H_2SO_4, \\triangle} C_6H_5NO_2 + H_2O
    转换规则（按序）：箭头（->[条件]→\\xrightarrow、->→\\rightarrow、<=>→
    \\rightleftharpoons）；电荷（元素/括号/方括号后的 数字?+/- → 上标，
    Fe2+→Fe^{2+}、[Ag(NH3)2]+→[Ag(NH_3)_2]^{+}）；数字下标（元素/括号后的
    数字 → 下标，C6→C_6）；沉淀符号（mhchem 的孤立 v 与 ↓ → \\downarrow）。
    """
    def _ce(m):
        inner = m.group(1)
        inner = re.sub(r"->\[([^\]]*)\]", r"\\xrightarrow{\1}", inner)
        inner = inner.replace("<=>", r"\rightleftharpoons")
        inner = inner.replace("->", r"\rightarrow")
        # 电荷上标：元素/圆括号/方括号后的 数字?+/-（2OH- → 2OH^{-}、
        # [Ag(NH3)2]+ → ]^{+}）；数字前导也转（Fe2+ → Fe^{2+}）
        inner = re.sub(r"([A-Za-z\)\]])(\d*)([+-])", r"\1^{\2\3}", inner)
        inner = re.sub(r"([A-Za-z\)\]])(\d+)", r"\1_\2", inner)
        # mhchem 沉淀/析出符号：孤立的 v（\ce{2Ag v} → 2Ag↓）与 ↓ 统一转
        # \downarrow（KaTeX 支持；mhchem v4 中孤立 v 即沉淀下箭头）
        inner = re.sub(r"(?<![A-Za-z])v(?![A-Za-z])", r"\\downarrow", inner)
        inner = inner.replace("↓", r"\downarrow")
        return inner

    return _CE_RE.sub(_ce, body)


def _convert_latex_markers(text: str) -> str:
    """LaTeX 公式分隔符 → Streamlit markdown 可渲染格式。

    \\(...\\) → $...$（行内公式），\\[...\\] → $$...$$（块公式）。
    Streamlit 的 markdown 内置 KaTeX 渲染 $...$/$$...$$（含 \\text{中文}、
    \\xrightarrow 等）；不转换时反斜杠被 markdown 吃掉，公式代码裸露。
    块公式的 $$ 必须独立成行且无前导空格（否则被当缩进代码块），
    内容压缩为单段（KaTeX 块公式内不能有空行）。公式内的 \\ce{...}
    同步转为 KaTeX 兼容语法。
    """
    # 空 \text{}（LLM 常在文本与 TikZ 之间留下的残留分隔符）无渲染意义，移除；
    # 常伴随 LaTeX 换行命令 \\ 与空白一起输出，一并清除（合法命令不受影响）
    text = re.sub(r"\\text\{\s*\}(?:\\\\|\s)*", "", text)

    def _disp(m):
        body = _convert_ce_math(m.group(1))
        body = re.sub(r"[ \t]+", " ", body).strip()
        body = re.sub(r"\n\s*\n+", "\n", body)
        return "\n$$\n" + body + "\n$$\n"

    text = _LATEX_DISPLAY_RE.sub(_disp, text)
    text = _LATEX_INLINE_RE.sub(
        lambda m: "$" + _convert_ce_math(m.group(1)).strip() + "$", text)
    # 裸 \\ce{...}（未用 \\(...\\)/\\[...\\] 包裹）→ 行内公式
    text = _CE_RE.sub(lambda m: "$" + _convert_ce_math("\\ce{" + m.group(1) + "}").strip() + "$", text)
    return text


def _summarize_title(question: str) -> str:
    """用 LLM 给对话生成简短标题（仅首条提问，极简 prompt 省 token）。

    失败（未配置 / 返回空 / 异常）时回退到问题前 _TITLE_MAX_LEN 字截断。
    """
    try:
        from core.llm_client import ask_llm
        title = ask_llm(f"提问：{question}\n对话标题：",
                        system_prompt=_TITLE_SYSTEM, max_tokens=64,
                        thinking="disabled")
        if title:
            # 清理：逐行去引号，取首个非空行，截断
            title = title.strip()
            title = next(
                (ln.strip('"“”\'').strip() for ln in title.splitlines()
                 if ln.strip().strip('"“”\'')),
                "")
            title = title[:_TITLE_MAX_LEN]
            if title:
                return title
    except Exception as e:
        print(f"[streamlit] 标题生成失败，回退截断: {e}")
    return question[:_TITLE_MAX_LEN]


def _build_history(cur: dict) -> list:
    """多轮历史：assistant 剥离渲染代码；空 content（图片理解失败残留）跳过。"""
    return [
        {"role": m["role"],
         "content": _strip_render_code(m["content"])
         if m["role"] == "assistant" else m["content"]}
        for m in cur["messages"][-_MAX_HISTORY:]
        if (m.get("content") or "").strip()
    ]


def _render_user_bubble(msg: dict, image_bytes: bytes | None = None) -> None:
    """渲染用户气泡：图片（全宽 + 下载按钮）+ 展示文本（display 优先，解 9c/10c）。"""
    with _chat_ctx("user"):
        b64 = msg.get("image")
        data = image_bytes if image_bytes is not None else (
            base64.b64decode(b64) if b64 else None)
        if data:
            st.image(data, width="stretch")   # 与文字气泡同宽（解 9b）
            st.download_button("下载图片", data=data,
                               file_name=msg.get("image_name") or "image.png",
                               key=f"dl_{uuid.uuid4().hex[:10]}")
        text = msg.get("display") if msg.get("display") is not None \
            else msg.get("content", "")
        if (text or "").strip():
            st.markdown(_convert_latex_markers(text))


def _append_user_msg(cur: dict, sessions: list, *, content: str = "",
                     display: str | None = None,
                     image_bytes: bytes | None = None,
                     image_name: str = "") -> dict:
    """用户消息入列（display 与 content 分离、image 以 base64 持久化，解 9d/10d）
    并立即渲染气泡。"""
    msg = {"role": "user",
           "content": content or (display or ""),
           "display": display,
           "image": (base64.b64encode(image_bytes).decode()
                     if image_bytes else None),
           "image_name": image_name or None}
    cur["messages"].append(msg)
    _save_sessions(sessions)
    _render_user_bubble(msg, image_bytes=image_bytes)
    return msg


def _generate_answer(sessions: list, cur: dict, question: str,
                     history: list, is_first: bool, status=None) -> None:
    """状态框 + 草稿流 + process_question + 标题 + 入列持久化 + 重跑。
    status 已存在时复用（图片流：视觉理解阶段已创建）。

    诊断：process_question 的 responses（原始标记文本，不含渲染后 TikZ）与
    diagnostics（每轮校验/渲染失败）在生成后写入 data/diagnostics.jsonl。"""
    diag = []          # diagnostics 收集（每轮失败）
    resp = []          # responses 收集（各阶段原始标记文本）
    if hasattr(st, "status"):
        if status is None:
            status = st.status("正在思考并绘制化学图示…", expanded=False)
        draft_box = st.empty()
        try:
            answer = process_question(
                question, history=history,
                progress_callback=_progress_updater(draft_box),
                correction_callback=lambda: status.update(
                    label="正在修正回答…", state="running"),
                diagnostics=diag, responses=resp)
            failed = (not answer) or answer.startswith("（LLM 调用失败")
        except Exception as e:
            answer = f"（生成异常：{e}）"
            failed = True
        draft_box.empty()
        status.update(
            label="完成" if not failed else "生成失败",
            state="complete" if not failed else "error",
        )
    else:
        with st.spinner("思考中（LLM 生成 + 渲染）..."):
            answer = process_question(question, history=history,
                                      diagnostics=diag, responses=resp)

    if is_first:
        cur["title"] = _summarize_title(question)
    cur["messages"].append({"role": "assistant", "content": answer})
    _save_sessions(sessions)
    _append_diagnostic(cur["id"], question, resp, diag)
    _rerun()


def _ask(sessions: list, session_id: str, question: str) -> None:
    """纯文字提问：消息先入列立即可见，再生成。"""
    cur = next((s for s in sessions if s["id"] == session_id), None)
    if cur is None:
        return
    is_first = not cur["messages"]
    history = _build_history(cur)
    _append_user_msg(cur, sessions, content=question)
    _generate_answer(sessions, cur, question, history, is_first)


@st.cache_data(show_spinner=False)
def _render_code_png(code: str) -> bytes | None:
    """编译 TikZ/chemfig 片段为 PNG（按代码内容缓存，重跑不重编）。"""
    return compile_tikz_to_png(code)


@st.cache_data(show_spinner=False)
def _detect_backends_cached() -> dict:
    """缓存 LaTeX 引擎探测结果（detect_backends 内部 subprocess，重复探测卡顿）。"""
    return detect_backends()


def _render_answer(text: str) -> None:
    """拆段渲染 assistant 回答（文本先行，图片编译后回填占位）。

    st.empty() 占位可在同一轮脚本内稍后回填：先按阅读顺序输出全部文本与
    「LaTeX 源码」下拉框（用户立即可读），再逐段编译 TikZ 并用 PNG 回填占位
    ——编译期间用户已在阅读文本，不再被"渲染图示中"整体阻塞。
    """
    segments = split_segments(text)
    code_segs = [c for k, c in segments if k == "code"]
    has_engine = False
    if code_segs:
        has_engine = bool(_detect_backends_cached().get("latex_engine"))
        if not has_engine:
            st.caption(
                "⚠️ 未检测到 LaTeX 引擎，图示以代码形式显示。"
                "安装 TeX Live / MiKTeX 后即可自动渲染为图片。"
            )

    slots = []  # (占位, TikZ 代码)：全部文本输出完毕后统一编译回填
    for kind, content in segments:
        if kind == "code":
            if not has_engine:
                st.code(content, language="latex")
                continue
            slot = st.empty()
            slot.caption("图示渲染中（LaTeX 编译，首次较慢）…")
            with st.expander("LaTeX 源码（复制到 Overleaf）"):
                st.code(content, language="latex")
            slots.append((slot, content))
        elif content.strip():
            st.markdown(_convert_latex_markers(content))

    for slot, code in slots:
        png = _render_code_png(code)
        if png:
            slot.image(png)
        else:
            slot.caption("⚠️ 图示编译失败，请展开下方 LaTeX 源码查看。")


def split_segments(text: str):
    """把含内联 TikZ 的文本拆成 [("text"|"code", 内容), ...] 段落序列。"""
    segments = []
    last = 0
    for m in _CODE_RE.finditer(text):
        if m.start() > last:
            segments.append(("text", text[last:m.start()]))
        segments.append(("code", m.group(0)))
        last = m.end()
    if last < len(text):
        segments.append(("text", text[last:]))
    return segments


@contextmanager
def _chat_ctx(role: str):
    """对话气泡容器；streamlit < 1.23 无 st.chat_message 时降级为普通 markdown。"""
    if hasattr(st, "chat_message"):
        with st.chat_message(role):
            yield
    else:
        st.markdown(f"**{'🧑 你' if role == 'user' else '🤖 助手'}**")
        yield


def _copy_iframe(srcdoc: str, height: int = 40) -> None:
    """内嵌 HTML iframe（st.iframe data-URL / components.v1.html 版本兼容）。

    st.iframe 只收 URL（1.59 无 srcdoc）→ HTML 经 URL 编码成 data: URL；
    脚本在 iframe 内可执行，navigator.clipboard 在 data: 源不可用时
    自动走按钮内的 textarea + execCommand 降级。
    """
    if hasattr(st, "iframe"):
        # charset 必写：data: URL 缺省 US-ASCII，中文 Windows 浏览器回退 GBK
        # 会把 UTF-8 中文渲染成乱码
        st.iframe("data:text/html;charset=utf-8,"
                  + urllib.parse.quote(srcdoc), height=height)
    else:
        import streamlit.components.v1 as st_components
        st_components.html(srcdoc, height=height)


def _copy_md_button(text: str) -> None:
    """每条回答下方的一键复制按钮（复制原始 Markdown）。

    Streamlit 无原生复制按钮，用 iframe 内嵌按钮实现：
    优先 navigator.clipboard.writeText，iframe 权限受限时降级
    textarea + execCommand。payload 经 JSON 转义并转义 `</`，
    防止回答内容中的 `</script>` 提前闭合脚本块。
    """
    payload = json.dumps(text, ensure_ascii=False).replace("</", "<\\/")
    # body margin:0 + overflow:hidden：按钮高度贴边，避免 iframe 溢出出现
    # 滚动条（Windows 经典滚动条在右缘显示为 ▲➖▼）
    _copy_iframe(
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='margin:0;overflow:hidden'>"
        "<button id='cp' style='padding:2px 10px;border:1px solid #bbb;"
        "border-radius:6px;background:transparent;color:inherit;"
        "cursor:pointer;font-size:13px;'>复制 Markdown</button>"
        "<script>"
        f"const text = {payload};"
        "const btn = document.getElementById('cp');"
        "btn.addEventListener('click', async () => {"
        "  try { await navigator.clipboard.writeText(text); }"
        "  catch (e) {"
        "    const ta = document.createElement('textarea');"
        "    ta.value = text; document.body.appendChild(ta);"
        "    ta.select(); document.execCommand('copy'); ta.remove();"
        "  }"
        "  btn.textContent = '已复制 ✓';"
        "  setTimeout(() => { btn.textContent = '复制 Markdown'; }, 1500);"
        "});"
        "</script></body></html>",
        height=40,
    )


# ---------------- 会话操作（侧边栏） ----------------

def _apply_rename(sid: str) -> None:
    """原位重命名：读取 text_input 值并保存（回车/✓ 触发）。"""
    s = next((x for x in st.session_state.sessions if x["id"] == sid), None)
    if s is None:
        return
    name = (st.session_state.get(f"rn_{sid}", "") or "").strip()
    s["title"] = (name or "新对话")[:30]
    _save_sessions(st.session_state.sessions)
    st.session_state.pop("renaming_id", None)
    _rerun()  # 立即恢复标题显示（输入框 → 标题）


def _delete_session(sid: str) -> None:
    """删除会话；若删的是当前会话则切到最近的。"""
    sessions = st.session_state.sessions
    sessions[:] = [s for s in sessions if s["id"] != sid]
    if st.session_state.get("current_id") == sid:
        st.session_state.current_id = sessions[-1]["id"] if sessions else None
    _save_sessions(sessions)
    st.session_state.pop("renaming_id", None)
    st.session_state.pop("pending_delete", None)
    _rerun()  # 立即重绘会话列表


def _session_menu(s: dict) -> None:
    """会话行右侧 ⋯ 菜单（重命名/删除）；旧版 streamlit 降级为展开。

    删除是**两步**的：先点「🗑 删除」进入待确认态（显示条数与"无法恢复"），
    再点「确认删除」才真的删。此前是一次点击即永久丢失，且与网页端
    （有确认弹窗）行为不一致。
    """
    def body() -> None:
        if st.session_state.get("pending_delete") == s["id"]:
            st.caption(f"删除「{s['title']}」？该会话的 "
                       f"{len(s.get('messages') or [])} 条消息将被永久删除，"
                       f"无法恢复。")
            c1, c2 = st.columns(2)
            if c1.button("确认删除", key=f"md_yes_{s['id']}",
                         width="stretch"):
                _delete_session(s["id"])
            if c2.button("取消", key=f"md_no_{s['id']}", width="stretch"):
                st.session_state.pop("pending_delete", None)
                _rerun()
            return
        if st.button("✏️ 重命名", key=f"mr_{s['id']}", width="stretch"):
            st.session_state.renaming_id = s["id"]
            _rerun()  # 立即重绘该行为输入框（原位编辑）
        if st.button("🗑 删除", key=f"md_{s['id']}", width="stretch"):
            st.session_state.pending_delete = s["id"]   # 只进入待确认，不删
            _rerun()

    if hasattr(st, "popover"):
        with st.popover("⋯", key=f"menu_{s['id']}"):
            body()
    else:
        with st.expander("⋯", key=f"menu_{s['id']}"):
            body()


# ---------------- 输入区（底部固定窗格） ----------------

def _chat_input_supports_file() -> bool:
    """st.chat_input 是否支持 accept_file 附件（streamlit ≥ 1.46）。"""
    if not hasattr(st, "chat_input"):
        return False
    try:
        return "accept_file" in inspect.signature(st.chat_input).parameters
    except (TypeError, ValueError):
        return False


def _on_prompt_change() -> None:
    """text_input 回车触发：暂存问题，由主流程统一处理（避免回调内 rerun）。

    仅旧版 streamlit（无 st.chat_input）回退路径使用。
    """
    q = (st.session_state.get("prompt_input") or "").strip()
    if q:
        st.session_state.pending_question = q
        st.session_state["prompt_input"] = ""


def _attachment_popover() -> None:
    """＋ 附件弹层：旧版 streamlit（chat_input 无附件能力）的上传入口。"""
    if hasattr(st, "popover"):
        with st.popover("＋", key="attach"):
            st.caption("上传图片（PNG/JPG）")
            st.file_uploader("上传图片", type=["png", "jpg", "jpeg"],
                             key="uploaded", label_visibility="collapsed")
    else:
        st.button("＋", key="attach")


def _describe_image_bytes(data: bytes, name: str) -> dict | None:
    """图片字节 → 临时文件 → 视觉理解 → 清理临时文件。"""
    suffix = "." + (name or "x.png").rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        from utils.ocr_utils import describe_image
        return describe_image(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _merge_image_question(question_text: str, desc: dict | None) -> str:
    """用户文字 + 图片描述合并为发给 LLM 的完整问题（仅提示词层使用，
    不进展示气泡——气泡由 display 字段承载，解 9c/10c）。

    desc 为空（图片识别失败）：空内容 + 用户文字照常传给主 LLM，并明确
    告知"图片识别失败"（视觉重试已耗尽，仍继续问答流程）。
    """
    if not desc or not desc.get("content"):
        tip = ("（用户上传了一张图片，但图片识别失败：视觉模型多次尝试仍"
               "无法获取图片内容。请基于文字内容作答，并提示用户重新上传"
               "图片或改用文字描述）")
        return f"{question_text}\n{tip}".strip() if question_text.strip() else tip
    if question_text.strip():
        return f"{question_text}\n（附图内容（{desc['type']}）：{desc['content']}）"
    return (f"用户上传了一张图片，图片内容（{desc['type']}）如下：\n"
            f"{desc['content']}\n请解答或分析其中的化学内容。")


def _ask_with_image(sessions: list, session_id: str, *, name: str,
                    data: bytes, question_text: str = "") -> None:
    """图文/纯图提问：图片+文字先入列立即可见（解 9e），再视觉理解、
    合并问题生成。display 与 content 分离：气泡只显示图片与用户文字。"""
    cur = next((s for s in sessions if s["id"] == session_id), None)
    if cur is None:
        return
    is_first = not cur["messages"]
    history = _build_history(cur)
    # 1. 入列：content 先放用户文字（描述成功后回填完整提示词）
    msg = _append_user_msg(cur, sessions, content=question_text,
                           display=question_text,
                           image_bytes=data, image_name=name)
    # 2. 视觉理解（消息已可见；status 先占"理解中"再转"生成中"）
    status = (st.status("正在理解图片内容（视觉模型）…", expanded=False)
              if hasattr(st, "status") else None)
    desc = _describe_image_bytes(data, name)
    if not desc or not desc.get("content"):
        # 识别失败：不中止——空内容 + 用户文字照常传给主 LLM，并明确
        # 告知"图片识别失败"（视觉模型已按 describe_image 内部重试耗尽）
        if status:
            status.update(label="图片理解失败（已重试），改用文字继续…",
                          state="warning")
        st.warning("图片识别失败：视觉模型多次尝试仍无法理解图片。"
                   "已基于文字内容继续生成，如需图片内容请重新上传或改用文字描述。")
        question = _merge_image_question(question_text, None)
        msg["content"] = question       # 回填完整提示词（供多轮历史沿用；不进气泡）
        _save_sessions(sessions)
        if status:
            status.update(label="正在思考并绘制化学图示…", state="running")
        _generate_answer(sessions, cur, question, history, is_first,
                         status=status)
        return
    question = _merge_image_question(question_text, desc)
    msg["content"] = question       # 回填完整提示词（供多轮历史沿用；不进气泡）
    _save_sessions(sessions)
    if status:
        status.update(label="正在思考并绘制化学图示…", state="running")
    _generate_answer(sessions, cur, question, history, is_first,
                     status=status)


def _handle_uploaded(uploaded) -> None:
    """已选附件（旧版 file_uploader 路径）：预览 + 识别并分析（统一图片流）。"""
    if uploaded is None:
        return
    st.image(uploaded, caption="已上传图片", width=200)
    if st.button("识别并分析", key="ocr_btn"):
        _ask_with_image(st.session_state.sessions,
                        st.session_state.current_id,
                        name=uploaded.name, data=uploaded.getvalue())


# ---------------- 页面 ----------------

st.set_page_config(page_title="有机化学知识智能体", page_icon="🧪", layout="wide")
# ★ 只隐藏页脚，**不隐藏右上角的 ⋮ 主菜单**：Streamlit 的「Settings → Theme
#   （Light / Dark / Use system setting）」就在那个菜单里（1.60 的前端包里
#   确实带 "Use system setting"），把它藏掉等于**顺手把深色模式也藏了**。
#   其余 CSS 只调宽度/间距，不含任何颜色 —— 两种主题下都自适应。
st.markdown(
    "<style>"
    "footer {visibility: hidden;}"
    "[data-testid='stChatInput']{"
    "width: max(280px, calc((100vw - 460px) * 0.8)) !important;"
    "max-width: 100% !important;"
    "margin-left: auto !important; margin-right: auto !important;}"
    "[data-testid='stMainBlockContainer']{"
    "width: max(440px, calc((100vw - 460px) * 0.8 + 160px)) !important;"
    "max-width: 100% !important;"
    "margin-left: auto !important; margin-right: auto !important;}"
    "@media (max-width: 863.98px){"
    "[data-testid='stMainBlockContainer']{"
    "width: max(312px, calc((100vw - 460px) * 0.8 + 32px)) !important;}}"
    "</style>",
    unsafe_allow_html=True,
)

# ---- 输入框：整页最先渲染的可见组分（固定窗格） ----
# st.chat_input 由前端钉在视口底部（[data-testid="stBottom"]：sticky，自带主题
# 层级与不透明背景，不被其他组分遮盖），与脚本位置无关；提到最前调用使其
# delta 最先到达前端——长历史重渲染 / 图片编译回填期间输入框也立即可见可用。
# 上方 CSS 把可见输入盒（[data-testid="stChatInput"]，无宽度声明、撑满父级，
# 是长度的真正控制点）固定为原长的 80%：(100vw - 侧边栏300 - 主区padding160)
# × 0.8，按视口计算与侧边栏状态无关——收起时长度不变、位置随居中左移；
# 主区内容列（stMainBlockContainer，border-box）取同宽 + 主区 padding 并居中，
# 使标题/消息/图片/按钮整列与输入框左右边缘对齐（padding 在 864px 断点变档
# 5rem↔1rem，内容列宽度随之补偿）。
_CHAT_FILE_OK = _chat_input_supports_file()
_CHAT_INPUT_OK = hasattr(st, "chat_input")
_submitted = None      # 现代路径（含附件）的提交值，页面末尾统一处理
_legacy_prompt = None  # 中间路径（无附件能力）的提交值
if _CHAT_FILE_OK:
    _submitted = st.chat_input(
        "输入化学问题，可基于上文连续追问；点 ＋ 可上传图片…",
        accept_file=True, file_type=["png", "jpg", "jpeg"])
elif _CHAT_INPUT_OK:
    _legacy_prompt = st.chat_input("输入化学问题，可基于上文连续追问…")

st.title("🧪 有机化学知识智能体")

# 会话初始化：加载 / 迁移旧历史 / 兜底新建
if "sessions" not in st.session_state:
    st.session_state.sessions = _load_sessions() or _migrate_legacy_history()
if not st.session_state.sessions:
    st.session_state.sessions = [_new_session([])]
    _save_sessions(st.session_state.sessions)
if "current_id" not in st.session_state or \
        st.session_state.current_id not in [s["id"] for s in st.session_state.sessions]:
    st.session_state.current_id = st.session_state.sessions[-1]["id"]

sessions = st.session_state.sessions
current_id = st.session_state.current_id

# 每次启动清空一次 diagnostics.jsonl（幂等），之后提问逐条 append
_flush_diagnostics_file()

# ---- 侧边栏：会话管理 ----
with st.sidebar:
    st.markdown("### 💬 对话")
    if st.button("＋ 新对话", width="stretch"):
        s = _new_session(sessions)
        st.session_state.current_id = s["id"]
        _save_sessions(sessions)
        _rerun()

    st.divider()
    for s in sessions:
        renaming = st.session_state.get("renaming_id") == s["id"]
        row = st.columns([1.0, 0.38])
        with row[0]:
            if renaming:
                # 原位重命名：输入框替换标题，回车或 ✓ 保存
                st.text_input(
                    "会话名", value=s["title"], key=f"rn_{s['id']}",
                    label_visibility="collapsed",
                    on_change=lambda sid=s["id"]: _apply_rename(sid))
            else:
                is_cur = s["id"] == current_id
                if st.button(
                        ("▶ " if is_cur else "") + s["title"],
                        key=f"sel_{s['id']}", width="stretch",
                        type="primary" if is_cur else "secondary"):
                    st.session_state.current_id = s["id"]
                    _rerun()
        with row[1]:
            if renaming:
                if st.button("✓", key=f"rn_ok_{s['id']}", help="保存"):
                    _apply_rename(s["id"])
            else:
                _session_menu(s)

    st.divider()
    st.caption(f"共 {len(sessions)} 个对话")

# ---- 主区：当前会话消息流 ----
cur = next((s for s in sessions if s["id"] == current_id), None)
if cur is not None:
    for msg in cur["messages"]:
        if msg["role"] == "assistant":
            with _chat_ctx("assistant"):
                _render_answer(msg["content"])
                _copy_md_button(msg["content"])
        else:
            _render_user_bubble(msg)

    # ---- 输入处理与附件区（输入框本体已在页面顶部渲染） ----
    if _CHAT_FILE_OK:
        if _submitted:
            text = (_submitted.text or "").strip()
            if _submitted.files:
                # 图文/纯图统一：提交后直接处理——消息先入列立即可见（解 9e），
                # 再视觉理解与生成，不再二次点击"识别并分析"（解 9a）
                f = _submitted.files[0]
                _ask_with_image(sessions, cur["id"],
                                name=f.name, data=f.getvalue(),
                                question_text=text)
            elif text:
                _ask(sessions, cur["id"], text)
    elif _CHAT_INPUT_OK:
        # chat_input 无附件能力（<1.46）：保留 ＋ 弹层上传
        _attachment_popover()
        _handle_uploaded(st.session_state.get("uploaded"))
        if _legacy_prompt and _legacy_prompt.strip():
            _ask(sessions, cur["id"], _legacy_prompt.strip())
    else:
        # 远古 streamlit（<1.24）：原 text_input 布局（随内容滚动）
        st.divider()
        cols = st.columns([0.08, 1.0])
        with cols[0]:
            _attachment_popover()
        with cols[1]:
            st.text_input(
                "输入化学问题，可基于上文连续追问…",
                key="prompt_input", label_visibility="collapsed",
                on_change=_on_prompt_change)

        # 待处理问题（text_input 回车暂存）→ 统一走 _ask
        if st.session_state.get("pending_question"):
            q = st.session_state.pop("pending_question")
            _ask(sessions, cur["id"], q)

        # 已选附件处理（上传图片预览 + 识别）
        _handle_uploaded(st.session_state.get("uploaded"))
