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

import base64
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


def _png_to_data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def render_reaction_view(result):
    """渲染反应方程式视图：flexbox 布局（竖直居中）+ chemfig 代码 + LLM 分析。"""
    st.markdown("### 🧪 反应方程式")
    reactants = result["reactants"]
    products = result["products"]
    arrow = "⇌" if result.get("reversible") else "→"

    # 单元格序列：图片与分隔符(+, →)交替，便于横向布局
    cells = []
    for i, smi in enumerate(reactants):
        cells.append(("img", smi))
        if i < len(reactants) - 1:
            cells.append(("sep", "+"))
    cells.append(("sep", arrow))
    for i, smi in enumerate(products):
        cells.append(("img", smi))
        if i < len(products) - 1:
            cells.append(("sep", "+"))

    # flexbox：align-items:center 让 + / → 与结构图竖直居中对齐
    items = []
    for kind, val in cells:
        if kind == "img":
            png = render_png(val)
            if png:
                url = _png_to_data_url(png)
                items.append(f'<img src="{url}" style="height:140px;">')
            else:
                items.append(f'<span style="font-family:monospace">{val}</span>')
        else:
            items.append(
                f'<span style="font-size:2em;font-weight:bold;color:#333;padding:0 4px">{val}</span>'
            )
    html = (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'gap:10px;flex-wrap:wrap;padding:12px 0;">'
        + "".join(items)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    if result.get("conditions"):
        st.caption(f"反应条件：{result['conditions']}")

    if result.get("equation_chemfig"):
        st.markdown(
            "复制下方代码到 [Overleaf](https://www.overleaf.com) 编译"
            "（需 `\\usepackage{chemfig}` + `amsmath`，可逆带条件还需 `mathtools`）"
        )
        st.code(result["equation_chemfig"], language="latex")

    st.markdown("### 📝 反应分析")
    if result["answer"]:
        st.markdown(result["answer"])
    else:
        st.error("LLM 调用失败。请检查 .env 中的 API_KEY / BASE_URL / MODEL 配置与网络。")

    with st.expander("🔧 详细信息"):
        st.write(f"- **输入**：{result['input']}")
        st.write(f"- **反应物**：{', '.join(reactants)}")
        st.write(f"- **产物**：{', '.join(products)}")
        st.write(f"- **条件**：{result.get('conditions') or '未标注'}")
        thermo = result.get("thermo")
        if thermo is None:
            st.write("- **热力学参数**：调用失败")
        elif thermo:
            st.write("- **热力学参数（LLM 估算，仅供参考）**：")
            for k, v in thermo.items():
                st.write(f"  - {k}：{v}")
        else:
            st.write("- **热力学参数**：数据不足（LLM 无法估算）")


st.set_page_config(page_title="有机化学知识智能体", page_icon="🧪", layout="centered")

st.title("🧪 有机化学知识智能体")
st.caption("输入化学名称/问题，或 reaction SMILES（如 CC(=O)O.CCO>>CC(=O)OCC.O）。支持中英文：苯酚 / aspirin / caffeine")

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
    if result.get("type") == "reaction":
        render_reaction_view(result)
    else:
        is_question = result.get("is_question", False)

        def render_structure():
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

        def render_answer():
            # 问题型用"解答"（回答是主体），裸名型用"说明"（结构是主体）
            st.markdown(f"### {'📝 解答' if is_question else '📝 说明'}")
            if result["answer"]:
                st.markdown(result["answer"])
            else:
                st.error("LLM 调用失败。请检查 .env 中的 API_KEY / BASE_URL / MODEL 配置与网络。")

        # 问题型：解答在前、结构在后；裸名型：结构在前、说明在后
        if is_question:
            render_answer()
            render_structure()
        else:
            render_structure()
            render_answer()

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
                st.write(f"- **熔点**：{props.get('melting_point') or '—'}")
                st.write(f"- **沸点**：{props.get('boiling_point') or '—'}")
                st.write(f"- **密度**：{props.get('density') or '—'}")
                st.write(f"- **XLogP**：{props.get('xlogp') or '—'}")
                st.write(f"- **CAS 号**：{props.get('cas') or '—'}")
            else:
                st.write("- **数据来源**：PubChem 在线解析（不在本地知识库）")
