# -*- coding: utf-8 -*-
"""streamlit_app.py — 本地 Web 界面（P3，本地测试用；清小搭接入用 FastAPI 层）。

启动: streamlit run streamlit_app.py

将 process_question 的输出（文本 + 内联 TikZ）拆段渲染：文本走 markdown，
TikZ 走 st.code(language="latex")（自带复制按钮，点右上角复制到 Overleaf）。
"""

import re

import streamlit as st

from app import process_question

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

result = st.session_state.result
if result:
    st.divider()
    for kind, content in split_segments(result):
        if kind == "code":
            st.code(content, language="latex")
        elif content.strip():
            st.markdown(content)
