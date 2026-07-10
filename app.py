# -*- coding: utf-8 -*-
"""
app.py
======
有机化学知识智能体 - Streamlit Web 界面。

启动:
    streamlit run app.py

功能:
    - 输入化学名称/问题 -> main_process 全链路处理
    - 展示中文解答、chemfig 结构式代码、（可选）RDKit PNG 缩略图
"""

from io import BytesIO

import streamlit as st

from chem_agent import main_process


def render_png(smiles: str):
    """用 rdkit.Chem.Draw 生成 PNG 缩略图作为视觉辅助。

    rdkit 不可用或绘图失败时返回 None，不影响主流程（AGENT.md 标注为可选功能）。
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=(320, 320))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


st.set_page_config(page_title="有机化学知识智能体", page_icon="🧪", layout="centered")

st.title("🧪 有机化学知识智能体")
st.caption("输入化学名称或问题，获取结构式代码与中文解答（支持中英文，如：苯酚 / aspirin / caffeine）")

# session_state 持久化结果，避免 Streamlit 重跑时丢失
if "result" not in st.session_state:
    st.session_state.result = None

user_input = st.text_input(
    "化学名称 / 问题",
    placeholder="例如：苯酚、阿司匹林、对乙酰氨基酚、caffeine ...",
)

col_btn, col_hint = st.columns([1, 4])
if col_btn.button("查询", type="primary"):
    query = user_input.strip()
    if not query:
        st.warning("请输入化学名称或问题。")
    else:
        with st.spinner("正在解析名称、查询知识库、渲染结构式并生成回答 ..."):
            st.session_state.result = main_process(query)

result = st.session_state.result

if result is not None:
    st.divider()

    # ---- 结构式：PNG 缩略图 + chemfig 代码 ----
    if result["smiles"]:
        st.markdown("### 🧬 结构式")
        png = render_png(result["smiles"])
        if png:
            st.image(png, caption="结构式预览（RDKit 渲染）")
        else:
            st.caption("（RDKit 图像不可用，请参考下方 chemfig 代码）")

    if result["chemfig"]:
        st.markdown(
            "复制下方代码到 [Overleaf](https://www.overleaf.com) 编译"
            "（需在导言区加 `\\usepackage{chemfig}` 与 `\\usepackage{mol2chemfig}`）"
        )
        st.code(result["chemfig"], language="latex")

    # ---- 文字说明 ----
    st.markdown("### 📝 说明")
    if result["answer"]:
        st.markdown(result["answer"])
    else:
        st.error("LLM 调用失败。请检查 .env 中的 API_KEY / BASE_URL / MODEL 配置与网络。")

    # ---- 详细信息 ----
    with st.expander("🔧 详细信息"):
        st.write(f"- **输入**：{result['input']}")
        st.write(f"- **SMILES**：`{result['smiles'] or '（未解析）'}`")
        props = result["properties"]
        if props:
            st.write(f"- **数据来源**：本地知识库")
            st.write(f"- **分子式**：{props.get('molecular_formula', '—')}")
            st.write(f"- **分子量**：{props.get('mol_weight', '—')}")
            if props.get("iupac_name"):
                st.write(f"- **IUPAC 名**：{props.get('iupac_name')}")
            if props.get("cas"):
                st.write(f"- **CAS 号**：{props.get('cas')}")
        else:
            st.write("- **数据来源**：PubChem 在线解析（不在本地知识库）")
