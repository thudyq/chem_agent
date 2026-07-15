# -*- coding: utf-8 -*-
"""streamlit_app.py — 本地 Web 界面（P3，本地测试用；清小搭接入用 FastAPI 层）。

启动: streamlit run streamlit_app.py

将 process_question 的输出（文本 + 内联 TikZ）拆段渲染：文本走 markdown，
TikZ/chemfig 代码段优先编译成 PNG 用 st.image 展示（检测到 LaTeX 引擎时），
未装 LaTeX 则回退为 st.code(language="latex")（点右上角复制到 Overleaf）。
"""

import re
import tempfile

import streamlit as st

from app import process_question
from utils.latex_compile import compile_tikz_to_png, detect_backends

# 代码段：tikzpicture 整块 | 单个 \chemfig{...}（兼容一级括号嵌套如 \mcfcringle{1.03}）
_CODE_RE = re.compile(
    r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"
    r"|\\chemfig\{(?:[^{}]|\{[^{}]*\})*\}",
    re.DOTALL,
)


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


@st.cache_data(show_spinner=False)
def _render_code_png(code: str) -> bytes | None:
    """编译 TikZ/chemfig 片段为 PNG（按代码内容缓存，重跑不重编）。"""
    return compile_tikz_to_png(code)


st.set_page_config(page_title="有机化学知识智能体", page_icon="🧪", layout="centered")
st.title("🧪 有机化学知识智能体")
st.caption("输入化学问题，获取图文混合回答。TikZ 代码块点右上角复制按钮即可粘到 Overleaf。")

if "result" not in st.session_state:
    st.session_state.result = None

user_q = st.text_input("你的问题", placeholder="如：画出苯的结构式 / 苯的硝化反应 / SN2 的势能面")

if st.button("提问", type="primary"):
    q = user_q.strip()
    if not q:
        st.warning("请输入问题。")
    else:
        with st.spinner("思考中（LLM 生成 + 渲染）..."):
            st.session_state.result = process_question(q)

# ---- 图片上传（R3：图片→SMILES→分析）----
st.divider()
st.markdown("#### 📷 或上传结构式图片")
uploaded = st.file_uploader("上传结构式图片（PNG/JPG）", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.image(uploaded, caption="已上传图片", width=200)
    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    if st.button("识别并分析", key="ocr_btn"):
        with st.spinner("识别结构式（视觉模型）..."):
            from utils.ocr_utils import image_to_smiles
            smiles = image_to_smiles(tmp_path)
        if smiles:
            st.success(f"识别到 SMILES：`{smiles}`")
            with st.spinner("分析中..."):
                st.session_state.result = process_question(
                    f"这个化合物的 SMILES 是 {smiles}，请分析其结构特征、官能团和基本化学性质。"
                )
        else:
            st.error("识别失败。请在 .env 中配置 VISION_MODEL 为支持视觉的模型（如 GLM-4V / Qwen2-VL）。")

result = st.session_state.result
if result:
    st.divider()
    segments = split_segments(result)
    # 预编译所有代码段：成功 st.image，失败回退 st.code
    code_segs = [c for k, c in segments if k == "code"]
    compiled = {}
    if code_segs:
        backends = detect_backends()
        if backends.get("latex_engine"):
            with st.spinner("渲染图示中（LaTeX 编译，首次较慢）..."):
                for c in code_segs:
                    compiled[c] = _render_code_png(c)
        else:
            st.caption(
                "⚠️ 未检测到 LaTeX 引擎，图示以代码形式显示。"
                "安装 TeX Live / MiKTeX 后即可自动渲染为图片。"
            )

    for kind, content in segments:
        if kind == "code":
            png = compiled.get(content)
            if png:
                st.image(png)
                with st.expander("LaTeX 源码（复制到 Overleaf）"):
                    st.code(content, language="latex")
            else:
                st.code(content, language="latex")
        elif content.strip():
            st.markdown(content)
