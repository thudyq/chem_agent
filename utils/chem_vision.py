# -*- coding: utf-8 -*-
"""utils/chem_vision.py — 化学视觉管线：版面分析 → 裁剪 → 类型路由。

用户上传的图片常是"图文混排"（文字 + 多个化学图）。本模块把整图交给
视觉模型做版面分析（每块内容的类型 + 归一化 bbox，20260816 实测 GLM-4.6V
可用：9/9 全匹配、类型 100%、平均中心距 0.041），按框裁剪后按类型路由：

    结构式  → MolScribe（本地 my-rdkit-env subprocess）→ SMILES
    反应式  → RxnScribe（同上）→ 反应 SMILES
    势能面/纽曼/其他 → 视觉 LLM + 化学领域提示词 → 结构化描述

MolScribe/RxnScribe 通过独立 Python 环境（CHEM_VISION_PYTHON，如 conda
my-rdkit-env）subprocess 调用——主 venv 零污染；未配置/超时/失败一律
静默回退视觉 LLM 描述（仿 rxn_balancer 的 best-effort 原则）。

用法:
    from utils.chem_vision import process_image
    result = process_image("upload.png")
    # → {"blocks": [{"type": "结构式", "text": "c1ccccc1", "bbox_px": [...]}, ...]}
"""

import base64
import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import requests

from core.config import settings

# ---------------------------------------------------------------- 版面分析

_LAYOUT_PROMPT = """这张图包含多块独立内容（标题文字、化学结构式、反应方程式、势能面图、纽曼投影图、说明文字等）。
请找出图中每一块内容，对每块输出：
- type: 内容类型，取值为 文字 / 结构式 / 反应式 / 势能面 / 纽曼投影 / 其他
- bbox: 该块在整图中的位置，归一化坐标 [x1, y1, x2, y2]（0~1，原点左上角，x 向右、y 向下）

严格只输出 JSON 数组，不要任何其他文字、不要思考过程或解释：
[{"type": "文字", "bbox": [0.05, 0.05, 0.4, 0.1]}, {"type": "结构式", "bbox": [0.1, 0.15, 0.4, 0.5]}]"""


def _call_vision(prompt: str, image_path: str, max_tokens: int = 2000) -> str | None:
    """调视觉模型（OpenAI 兼容），返回文本。

    关闭深度思考（thinking disabled）让模型直接输出 content——实测
    glm-4.6v 思考模式会把回答吞进 reasoning_content 致 content 为空
    （20260816）；content 仍空时回退 reasoning_content 尝试提取。
    """
    config = settings.vision
    if not config.is_configured:
        return None
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = str(image_path).rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    data_url = f"data:{mime};base64,{b64}"
    url = f"{config.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        msg = resp.json()["choices"][0]["message"]
    except (KeyError, ValueError):
        return None
    content = msg.get("content") or ""
    if not content.strip():
        reasoning = msg.get("reasoning_content") or ""
        if reasoning.strip():
            content = reasoning
        else:
            return None
    return content.strip() or None


def _parse_blocks(text: str) -> list[dict]:
    """容错解析视觉模型输出为 [{type, bbox:[x1,y1,x2,y2]}]。"""
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            data = json.loads(m.group(0).replace("'", '"'))
        except json.JSONDecodeError:
            return []
    blocks = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict) or "bbox" not in d:
            continue
        bb = d["bbox"]
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            bb = [float(x) for x in bb]
        except (TypeError, ValueError):
            continue
        if not all(0.0 <= v <= 1.0 for v in bb) or bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        blocks.append({"type": str(d.get("type", "其他")), "bbox": bb})
    return blocks


def analyze_layout(image_path: str) -> list[dict] | None:
    """版面分析：整图 → 每块 {type, bbox(归一化)}；失败返回 None。"""
    text = _call_vision(_LAYOUT_PROMPT, image_path)
    if not text:
        return None
    blocks = _parse_blocks(text)
    return blocks or None


def crop_blocks(image_path: str, blocks: list[dict], pad: float = 0.02) -> list[dict]:
    """按归一化 bbox 裁剪（PIL），外扩 pad（相对图宽高），返回带像素 bbox 的块。

    每块: {type, bbox(归一化), bbox_px, image: PIL.Image}。裁剪越界自动钳制。
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    out = []
    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        px1 = max(0, int((x1 - pad) * w))
        py1 = max(0, int((y1 - pad) * h))
        px2 = min(w, int((x2 + pad) * w))
        py2 = min(h, int((y2 + pad) * h))
        if px2 <= px1 or py2 <= py1:
            continue
        out.append({
            "type": b["type"],
            "bbox": b["bbox"],
            "bbox_px": [px1, py1, px2, py2],
            "image": img.crop((px1, py1, px2, py2)),
        })
    return out


# --------------------------------------------------- MolScribe / RxnScribe

_MOLSCRIBE_CODE = (
    "import sys, json\n"
    "from molscribe import MolScribe\n"
    "m = MolScribe()\n"
    "smi = m.predict_image_file(sys.argv[1])\n"
    "print(json.dumps({'smiles': smi}))\n"
)

_RXNSCRIBE_CODE = (
    "import sys, json\n"
    "from rxnscribe import RxnScribe\n"
    "m = RxnScribe()\n"
    "outs = m.predict_image_file(sys.argv[1])\n"
    "print(json.dumps(outs, ensure_ascii=False))\n"
)


def _env_python() -> str | None:
    """CHEM_VISION_PYTHON 配置优先；否则探测常见 conda 环境。

    候选环境名：chem-vision（MolScribe/RxnScribe 独立 3.11 环境，推荐）、
    my-rdkit-env（早期约定）。MolScribe/RxnScribe 需 Python 3.11，
    my-rdkit-env 若是 3.12+ 会导入失败——届时显式配置
    CHEM_VISION_PYTHON 指向 chem-vision 环境的 python.exe。
    """
    cfg = os.environ.get("CHEM_VISION_PYTHON", "").strip()
    if cfg:
        return cfg
    home = Path.home()
    for env_name in ("chem-vision", "molscribe", "my-rdkit-env"):
        for conda in (home / "anaconda3", home / "miniconda3",
                      home / "mambaforge"):
            for p in (conda / "envs" / env_name / "python.exe",
                      conda / "envs" / env_name / "bin" / "python"):
                if p.exists():
                    return str(p)
    return None


def _run_python(code: str, image_path: str, timeout: int) -> str | None:
    """在独立环境跑识别脚本（子进程隔离 + 超时）；任何失败返回 None。"""
    py = _env_python()
    if not py:
        return None
    try:
        r = subprocess.run([py, "-c", code, str(image_path)],
                           capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out or None


def predict_molscribe(image_path: str) -> str | None:
    """MolScribe 单分子识别 → SMILES；失败/不可用返回 None。"""
    out = _run_python(_MOLSCRIBE_CODE, image_path,
                      settings.chem_vision.molscribe_timeout)
    if not out:
        return None
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return None
    smi = d.get("smiles") if isinstance(d, dict) else None
    if not smi:
        return None
    return str(smi).strip() or None


def predict_rxnscribe(image_path: str) -> str | None:
    """RxnScribe 反应式识别 → 反应信息文本；失败/不可用返回 None。

    输出为 dict 列表（每个反应含 reactants/reagents/products/condition
    SMILES 等），序列化为 JSON 文本返回，调用方/用户核对格式。
    """
    out = _run_python(_RXNSCRIBE_CODE, image_path,
                      settings.chem_vision.rxnscribe_timeout)
    return out or None


# ------------------------------------------------------------ 领域描述

# 各图类型的结构化描述提示词：输出可直接映射到项目标记参数
_DESCRIBE_PROMPTS = {
    "结构式": """这是化学结构式图片。请识别分子结构：能确定时给出 SMILES；不能确定时用文字描述（如"苯环连一个硝基"）。""",
    "反应式": """这是化学反应方程式图片。请列出：①反应物 ②产物 ③反应条件/试剂（分子尽量用 SMILES 或规范化学式）。若含可逆/等号箭头请说明。""",
    "势能面": """这是化学反应势能面（能量-反应坐标）图。请输出：
①曲线类型（单步/多步反应）
②各驻点按横轴顺序列出：角色（反应物/过渡态/中间体/产物）+ 相对能量高低（如"过渡态=最高点"）
③若轴上有数值标注请转录。输出格式：驻点序列: [角色=能量描述], ...""",
    "纽曼投影": """这是纽曼投影图（大圆=远端碳，中心点=近端碳，各碳上三根键呈 120° 放射）。请输出：
①近端碳三根键上各连什么基团（H/CH3/Cl 等）
②远端碳三根键上各连什么基团
③两组键的相对角度（近端键与远端键的夹角：交叉式≈60°、重叠式≈0°）
④构象名称（交叉式/重叠式/邻位交叉式等）""",
    "其他": """请描述这张化学相关图片的内容：图类型、关键结构/数值/标注。""",
}


def describe_block(block_type: str, image: object) -> str:
    """视觉 LLM 按领域提示词描述单个裁剪块；返回文本（失败空串）。

    image: PIL.Image 或 文件路径。裁剪块用临时文件传入（视觉 API 需路径）。
    """
    prompt = _DESCRIBE_PROMPTS.get(
        block_type, _DESCRIBE_PROMPTS["其他"])
    tmp = None
    try:
        from PIL import Image as _I
        if hasattr(image, "save"):   # PIL.Image
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.save(tmp)
            path = tmp
        else:
            path = str(image)
        text = _call_vision(prompt, path, max_tokens=1200)
        return (text or "").strip()
    except Exception:
        return ""
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


# ------------------------------------------------------------- 主入口

# 类型 → 处理函数（可插拔：MolScribe/RxnScribe 未装时回退 describe_block）
def _handle_structure(block: dict) -> str:
    smi = predict_molscribe(block["_tmp_path"])
    if smi:
        return f"结构式 SMILES: {smi}"
    return describe_block("结构式", block["image"])


def _handle_reaction(block: dict) -> str:
    out = predict_rxnscribe(block["_tmp_path"])
    if out:
        return f"反应式识别: {out}"
    return describe_block("反应式", block["image"])


_HANDLERS = {
    "结构式": _handle_structure,
    "反应式": _handle_reaction,
    "势能面": lambda b: describe_block("势能面", b["image"]),
    "纽曼投影": lambda b: describe_block("纽曼投影", b["image"]),
}


def process_image(image_path: str, pad: float = 0.02, tmpdir: str | None = None) -> dict | None:
    """图文混排图片 → 版面分析 + 裁剪 + 类型路由 → 结构化内容块。

    返回 {"blocks": [{"type", "text", "bbox_px"}...], "note": 降级说明}
    版面分析失败（视觉模型不可用/无输出）返回 None——调用方沿用原逻辑。
    tmpdir: 裁剪块落盘目录（MolScribe/RxnScribe subprocess 需文件路径）；
        缺省用系统临时目录。测试环境（受限）可传入项目内目录。
    """
    blocks = analyze_layout(image_path)
    if not blocks:
        return None
    cropped = crop_blocks(image_path, blocks, pad=pad)
    if not cropped:
        return None
    import tempfile
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="chemvision_")
    else:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    results = []
    for b in cropped:
        handler = _HANDLERS.get(b["type"])
        if handler is None:      # 文字/其他 → 通用描述
            text = describe_block("其他", b["image"]) if b["type"] != "文字" \
                else describe_block("文字", b["image"])
            results.append({"type": b["type"], "text": text,
                            "bbox_px": b["bbox_px"]})
            continue
        tmp = os.path.join(tmpdir, f"block_{len(results)}.png")
        try:
            b["image"].save(tmp)
            b["_tmp_path"] = tmp
        except Exception:
            b["_tmp_path"] = tmp
        text = handler(b)
        results.append({"type": b["type"], "text": text,
                        "bbox_px": b["bbox_px"]})
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
    return {"blocks": results}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("用法: python -m utils.chem_vision <图片路径>")
        raise SystemExit(1)
    res = process_image(sys.argv[1])
    if not res:
        print("版面分析失败（视觉模型不可用/无输出）")
        raise SystemExit(1)
    for b in res["blocks"]:
        print(f"- [{b['type']}] {b['text'][:120]}")
