"""Generic adaptive PNG sizing tests independent of service adapters."""

def test_adaptive_dpi_ladder(monkeypatch):
    """compile_tikz_to_png_adaptive：超阈值按阶梯降 DPI，取第一个达标者。"""
    import utils.latex_compile as lc

    by_dpi = {300: b"A" * 5000, 200: b"A" * 3000, 150: b"A" * 1500,
              100: b"A" * 800, 75: b"A" * 400}
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex",
                        lambda engine, tex, flags: tex + ".pdf")
    monkeypatch.setattr(lc, "_pdf_to_png_pymupdf",
                        lambda pdf, dpi: by_dpi.get(dpi))
    code = "\\begin{tikzpicture}\\end{tikzpicture}"
    assert lc.compile_tikz_to_png_adaptive(code, max_kb=1) == by_dpi[100]
    # 无阈值：300dpi 快路径（零行为变化）
    monkeypatch.setattr(lc, "compile_tikz_to_png",
                        lambda c, title="", **kw: by_dpi[300])
    assert lc.compile_tikz_to_png_adaptive(code) == by_dpi[300]


def test_adaptive_dpi_floor(monkeypatch):
    """阶梯耗尽仍超限 → 返回最小（最低 DPI）PNG，宁降清晰度保显示。"""
    import utils.latex_compile as lc

    by_dpi = {300: b"A" * 5000, 200: b"A" * 4000, 150: b"A" * 3000,
              100: b"A" * 2000, 75: b"A" * 1500}
    monkeypatch.setattr(lc, "_find_latex_engine", lambda: "xelatex")
    monkeypatch.setattr(lc, "_run_latex",
                        lambda engine, tex, flags: tex + ".pdf")
    monkeypatch.setattr(lc, "_pdf_to_png_pymupdf",
                        lambda pdf, dpi: by_dpi.get(dpi))
    out = lc.compile_tikz_to_png_adaptive(
        "\\begin{tikzpicture}\\end{tikzpicture}", max_kb=1)   # 全部 > 1KB
    assert out == by_dpi[75]


