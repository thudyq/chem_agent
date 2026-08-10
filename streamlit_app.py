# -*- coding: utf-8 -*-
r"""streamlit_app.py — 本地 Web 界面（本地测试用；清小搭接入用 FastAPI 层）。

启动: streamlit run streamlit_app.py

对话式 AI 界面（参考 DeepSeek 布局）：
- 侧边栏会话管理：新对话 + 会话列表（悬停 ⋯ 三点菜单：重命名/删除；
  重命名为原位编辑，回车或 ✓ 保存）；
- 主区当前会话消息流（user 问题 + assistant 图文回答）；
- 底部输入区：st.chat_input 固定窗格（钉在视口底部，向主流大模型聊天界面
  看齐），整页最先渲染的可见组分（调用提到脚本最前，delta 最先到达前端），
  CSS 最高 z-index + 不透明底色保证不被其他组分遮盖；
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
import re
import tempfile
import time
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
    \\rightleftharpoons）；电荷（元素/括号后的 数字?+/- → 上标，Fe2+→Fe^{2+}）；
    数字下标（元素/括号后的数字 → 下标，C6→C_6）。
    """
    def _ce(m):
        inner = m.group(1)
        inner = re.sub(r"->\[([^\]]*)\]", r"\\xrightarrow{\1}", inner)
        inner = inner.replace("<=>", r"\rightleftharpoons")
        inner = inner.replace("->", r"\rightarrow")
        inner = re.sub(r"([A-Za-z\)])(\d*)([+-])", r"\1^{\2\3}", inner)
        inner = re.sub(r"([A-Za-z\)])(\d+)", r"\1_\2", inner)
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
                        system_prompt=_TITLE_SYSTEM, max_tokens=64)
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


def _ask(sessions: list, session_id: str, question: str) -> None:
    """带多轮历史与生成进度反馈的提问；追加到指定会话并持久化。"""
    cur = next((s for s in sessions if s["id"] == session_id), None)
    if cur is None:
        return
    history = [
        {"role": m["role"],
         "content": _strip_render_code(m["content"])
         if m["role"] == "assistant" else m["content"]}
        for m in cur["messages"][-_MAX_HISTORY:]
    ]
    if hasattr(st, "status"):
        status = st.status("正在思考并绘制化学图示…", expanded=False)
        draft_box = st.empty()
        try:
            answer = process_question(
                question, history=history,
                progress_callback=_progress_updater(draft_box))
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
            answer = process_question(question, history=history)

    if not cur["messages"]:
        cur["title"] = _summarize_title(question)
    cur["messages"].append({"role": "user", "content": question})
    cur["messages"].append({"role": "assistant", "content": answer})
    _save_sessions(sessions)
    _rerun()


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
    _rerun()  # 立即重绘会话列表


def _session_menu(s: dict) -> None:
    """会话行右侧 ⋯ 菜单（重命名/删除）；旧版 streamlit 降级为展开。"""
    if hasattr(st, "popover"):
        with st.popover("⋯", key=f"menu_{s['id']}"):
            if st.button("✏️ 重命名", key=f"mr_{s['id']}", use_container_width=True):
                st.session_state.renaming_id = s["id"]
                _rerun()  # 立即重绘该行为输入框（原位编辑）
            if st.button("🗑 删除", key=f"md_{s['id']}", use_container_width=True):
                _delete_session(s["id"])
    else:
        with st.expander("⋯", key=f"menu_{s['id']}"):
            if st.button("✏️ 重命名", key=f"mr_{s['id']}"):
                st.session_state.renaming_id = s["id"]
                _rerun()
            if st.button("🗑 删除", key=f"md_{s['id']}"):
                _delete_session(s["id"])


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


def _ocr_and_ask(name: str, data: bytes) -> None:
    """图片字节 → 临时文件 → 视觉识别 SMILES → 自动提问。"""
    suffix = "." + name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    with st.spinner("识别结构式（视觉模型）..."):
        from utils.ocr_utils import image_to_smiles
        smiles = image_to_smiles(tmp_path)
    if smiles:
        st.success(f"识别到 SMILES：`{smiles}`")
        cur = next((s for s in st.session_state.sessions
                    if s["id"] == st.session_state.current_id), None)
        if cur:
            _ask(st.session_state.sessions, cur["id"],
                 f"这个化合物的 SMILES 是 {smiles}，请分析其结构特征、官能团和基本化学性质。")
    else:
        st.error("识别失败。请在 .env 中配置 VISION_MODEL 为支持视觉的模型（如 GLM-4V / Qwen2-VL）。")


def _handle_pending_upload() -> None:
    """chat_input 附件（提交时暂存的字节）：预览 + 识别并分析 / 移除。"""
    pending = st.session_state.get("pending_upload")
    if not pending:
        return
    name, data = pending
    st.image(data, caption=f"已上传图片：{name}", width=200)
    cols = st.columns([0.2, 0.2, 1.0])
    with cols[0]:
        if st.button("识别并分析", key="ocr_btn"):
            st.session_state.pop("pending_upload", None)
            _ocr_and_ask(name, data)
    with cols[1]:
        if st.button("移除", key="ocr_rm"):
            st.session_state.pop("pending_upload", None)
            _rerun()


def _handle_uploaded(uploaded) -> None:
    """已选附件（旧版 file_uploader 路径）：预览 + 识别并分析。"""
    if uploaded is None:
        return
    st.image(uploaded, caption="已上传图片", width=200)
    if st.button("识别并分析", key="ocr_btn"):
        _ocr_and_ask(uploaded.name, uploaded.getvalue())


# ---------------- 页面 ----------------

st.set_page_config(page_title="有机化学知识智能体", page_icon="🧪", layout="wide")
st.markdown(
    "<style>"
    "footer {visibility: hidden;}"
    "#MainMenu {visibility: hidden;}"
    ".stChatMessage {max-width: 780px; margin: 0 auto;}"
    ".stChatFloatingInputContainer,[data-testid='stChatInput']"
    "{z-index: 999999; background-color: var(--background-color);}"
    "</style>",
    unsafe_allow_html=True,
)

# ---- 输入框：整页最先渲染的可见组分（固定窗格，默认最前不被遮盖） ----
# st.chat_input 由前端钉在视口底部，与脚本位置无关；提到最前调用使其 delta
# 最先到达前端——长历史重渲染 / 图片编译回填期间输入框也立即可见可用；
# 上方 CSS 赋最高 z-index 与不透明底色，其他组分任何时候都不遮盖它。
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

# ---- 侧边栏：会话管理 ----
with st.sidebar:
    st.markdown("### 💬 对话")
    if st.button("＋ 新对话", use_container_width=True):
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
                        key=f"sel_{s['id']}", use_container_width=True,
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
        with _chat_ctx(msg["role"]):
            if msg["role"] == "assistant":
                _render_answer(msg["content"])
            elif msg["content"].strip():
                st.markdown(_convert_latex_markers(msg["content"]))

    # ---- 输入处理与附件区（输入框本体已在页面顶部渲染） ----
    if _CHAT_FILE_OK:
        _handle_pending_upload()
        if _submitted:
            if _submitted.files:
                f = _submitted.files[0]
                st.session_state.pending_upload = (f.name, f.getvalue())
            text = (_submitted.text or "").strip()
            if text:
                _ask(sessions, cur["id"], text)
            elif _submitted.files:
                _rerun()  # 仅附件无文字：重跑以展示附件预览
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
