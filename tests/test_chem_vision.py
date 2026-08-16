# -*- coding: utf-8 -*-
"""utils/chem_vision.py 版面分析/裁剪/路由单元测试（mock 视觉与 subprocess）。

不调用真实视觉模型与 MolScribe/RxnScribe（测试环境无网络/独立环境），
验证：版面 JSON 容错解析、按 bbox 裁剪、类型路由与降级（识别不可用 →
回退领域描述）、molscribe 结果解析。
"""

import json
from pathlib import Path

import pytest

import utils.chem_vision as cv


@pytest.fixture
def tmp_workdir():
    """项目内临时目录（沙箱拒绝系统 Temp 写入，pytest tmp_path 不可用）。"""
    d = Path(__file__).resolve().parent / "_cv_tmp"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _layout_json():
    """模拟 GLM-4.6V 版面分析输出（与 fast_test/bbox_test_truth 同构）。"""
    return ("```json\n"
            '[{"type": "文字", "bbox": [0.02, 0.05, 0.4, 0.1]}, '
            '{"type": "结构式", "bbox": [0.18, 0.2, 0.34, 0.44]}, '
            '{"type": "反应式", "bbox": [0.52, 0.27, 0.96, 0.38]}, '
            '{"type": "势能面", "bbox": [0.17, 0.56, 0.34, 0.8]}, '
            '{"type": "纽曼投影", "bbox": [0.67, 0.57, 0.84, 0.77]}]'
            "\n```")


def test_parse_blocks():
    """版面 JSON 容错解析：代码块/前言/非法。"""
    blocks = cv._parse_blocks(_layout_json())
    assert len(blocks) == 5
    assert blocks[1]["type"] == "结构式"
    assert blocks[1]["bbox"] == [0.18, 0.2, 0.34, 0.44]
    # 前言文字 + 裸 JSON
    assert len(cv._parse_blocks("内容如下：[{\"type\": \"反应式\", "
                                "\"bbox\": [0.5, 0.2, 0.9, 0.4]}]")) == 1
    # 无 JSON / 越界 bbox 丢弃
    assert cv._parse_blocks("无法识别") == []
    assert cv._parse_blocks('[{"type": "x", "bbox": [0.1, 0.1, 0.5, 0.5], '
                            '"bbox": [0, 0, 2, 2]}]') == []


def test_crop_blocks(tmp_workdir):
    """按归一化 bbox 裁剪，外扩 pad、越界钳制、像素坐标正确。"""
    from PIL import Image
    p = tmp_workdir / "t.png"
    Image.new("RGB", (1000, 800), "white").save(p)
    out = cv.crop_blocks(str(p), [{"type": "结构式", "bbox": [0.1, 0.1, 0.5, 0.5]}],
                         pad=0.0)
    assert len(out) == 1
    assert out[0]["bbox_px"] == [100, 80, 500, 400]
    assert out[0]["image"].size == (400, 320)
    # 越界 bbox 钳制
    out2 = cv.crop_blocks(str(p), [{"type": "x", "bbox": [0.9, 0.9, 1.1, 1.1]}],
                          pad=0.05)
    assert out2[0]["bbox_px"][2] == 1000 and out2[0]["bbox_px"][3] == 800


def test_env_python(monkeypatch):
    """CHEM_VISION_PYTHON 优先于探测；未配置探测 my-rdkit-env。"""
    monkeypatch.setenv("CHEM_VISION_PYTHON", "C:/fake/python.exe")
    assert cv._env_python() == "C:/fake/python.exe"
    monkeypatch.delenv("CHEM_VISION_PYTHON")
    # 探测路径存在性依赖真实环境，只验证返回 str 或 None
    p = cv._env_python()
    assert p is None or isinstance(p, str)


def test_predict_molscribe(monkeypatch):
    """MolScribe subprocess 输出解析；失败/不可用返回 None。"""
    monkeypatch.setattr(cv, "_run_python", lambda *a, **k: '{"smiles": "c1ccccc1"}')
    assert cv.predict_molscribe("x.png") == "c1ccccc1"
    monkeypatch.setattr(cv, "_run_python", lambda *a, **k: None)
    assert cv.predict_molscribe("x.png") is None
    monkeypatch.setattr(cv, "_run_python", lambda *a, **k: "not json")
    assert cv.predict_molscribe("x.png") is None


def test_process_image_routing(monkeypatch, tmp_workdir):
    """主流程：版面分析 → 裁剪 → 类型路由；识别不可用回退领域描述。"""
    from PIL import Image
    img = Image.new("RGB", (1000, 800), "white")
    p = tmp_workdir / "mix.png"
    img.save(p)

    monkeypatch.setattr(cv, "_call_vision", lambda *a, **k: _layout_json())
    # MolScribe/RxnScribe 不可用 → 回退 describe_block（mock 文本）
    monkeypatch.setattr(cv, "_run_python", lambda *a, **k: None)
    monkeypatch.setattr(cv, "describe_block",
                        lambda t, im: f"描述:{t}")

    res = cv.process_image(str(p), tmpdir=str(tmp_workdir / "tmp"))
    assert res is not None
    types = [b["type"] for b in res["blocks"]]
    assert "结构式" in types and "反应式" in types
    assert "势能面" in types and "纽曼投影" in types
    # 结构式走了 molscribe（不可用）→ 回退描述
    for b in res["blocks"]:
        assert b["bbox_px"][0] <= b["bbox_px"][2]
        assert b["text"]  # 非空

    # 识别可用：结构式走 MolScribe → SMILES
    monkeypatch.setattr(cv, "_run_python",
                        lambda *a, **k: '{"smiles": "c1ccccc1"}')
    res2 = cv.process_image(str(p), tmpdir=str(tmp_workdir / "tmp2"))
    s = next(b for b in res2["blocks"] if b["type"] == "结构式")
    assert "c1ccccc1" in s["text"]

    # 版面分析失败 → None
    monkeypatch.setattr(cv, "_call_vision", lambda *a, **k: None)
    assert cv.process_image(str(p), tmpdir=str(tmp_workdir / "tmp3")) is None
