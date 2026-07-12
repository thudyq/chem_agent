# -*- coding: utf-8 -*-
"""renderers/structure.py — [STRUCT] 标记渲染器：SMILES → chemfig TikZ。

提取自旧 utils/structure_render.py 的 smiles_to_tikz，重命名为 render_structure，
并新增 label 参数（Day 8-9 增强 label 定位）。
"""


def render_structure(smiles: str, label: str = None) -> str:
    """SMILES → \\chemfig{...} 代码；可选在下方加 label 标注。

    依赖 mol2chemfigPy3（内部用 epam.indigo，非 rdkit）。
    任何失败返回空串 ""，不抛异常。

    参数:
        smiles: 合法 SMILES，如 "c1ccccc1"（苯）。
        label: 可选结构标注名称，置于结构下方。

    返回:
        \\chemfig{...} 代码字符串（含 label 时附标注）；失败返回 ""。
    """
    if not smiles or not isinstance(smiles, str):
        print("[render_structure] 输入 SMILES 为空或类型错误。")
        return ""

    try:
        from mol2chemfigPy3 import mol2chemfig
    except ImportError:
        print("[render_structure] mol2chemfigPy3 未安装，无法渲染结构式。")
        return ""

    try:
        # inline=True 才返回字符串；默认 inline=False 只打印到 stdout 返回 None
        result = mol2chemfig(smiles, inline=True)
    except Exception as e:
        print(f"[render_structure] mol2chemfig 调用异常: {e}")
        return ""

    # 失败时 mol2chemfig 返回错误描述字符串（非 None），需校验确实是 chemfig 代码
    if not isinstance(result, str) or not result.startswith("\\chemfig"):
        print(f"[render_structure] 渲染失败，返回值非 chemfig 代码: {result!r}")
        return ""

    if label:
        # 在结构下方加居中加粗标注（Day 8-9 会改为 \node 精确定位）
        result = result + "\n\\\\\n" + f"\\centerline{{\\textbf{{{label}}}}}"
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("render_structure 测试：c1ccccc1 / CC(=O)O,label=乙酸")
    print("=" * 60)
    print("[1] 苯:")
    print(render_structure("c1ccccc1") or "(渲染失败)")
    print("\n[2] 乙酸(带label):")
    print(render_structure("CC(=O)O", label="乙酸") or "(渲染失败)")
