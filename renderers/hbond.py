# -*- coding: utf-8 -*-
"""renderers/hbond.py — 顶层 [HBOND] 渲染器（已弃用，保留文件供参考）。

2026-08-15 语义分离：HBOND 从顶层标记移除，仅支持容器内
HBOND:idA:a#k>idB:b（氢由 XH 负责绘制，HBOND 只画 H···Y 点状虚线，
见 renderers/composite.py 的 _parse_hbond_specs 与渲染段）。
本文件的顶层 render_hbond（SMILES + from-to，自带 XH 逻辑）不再注册使用。
"""


def render_hbond(smiles: str, pairs_str: str = "") -> str:
    """[HBOND] 渲染（已弃用）：顶层标记已移除，此函数仅供历史参考。

    请改用容器内 HBOND:idA:a#k>idB:b（见 composite.py）。
    """
    return ("（氢键渲染失败：HBOND 已改为容器内标记 "
            "供体组件:原子#第k个H>受体组件:原子，氢原子请先用 XH 画出）")


if __name__ == "__main__":
    print("HBOND 顶层渲染器已弃用（2026-08-15 语义分离）。")
    print("当前用法：容器内 HBOND:idA:a#k>idB:b，氢由 XH 画出。")
