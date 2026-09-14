# -*- coding: utf-8 -*-
"""core/config.py — 统一配置入口。

设计原则
--------
* 单一入口：所有环境变量 / 默认常量统一在此读取。
* 无额外依赖：使用标准库 dataclasses 与 pathlib，python-dotenv 缺失时手动解析兜底。
* 只读配置：实例化后不可变，避免运行时被意外修改。
* 向后兼容：保留 `MODEL` 作为 `MODEL_NAME` 的 fallback。

用法
----
    from core.config import settings

    print(settings.llm.api_key)
    print(settings.vision.model_name)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """把项目根目录下的 .env 加载进 os.environ（python-dotenv 优先，缺失时手动解析兜底）。"""
    env_paths = [_PROJECT_ROOT / ".env", Path(".env")]
    try:
        from dotenv import load_dotenv
        for p in env_paths:
            if p.exists():
                load_dotenv(p)
                return
        return
    except ImportError:
        pass

    for p in env_paths:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip("'").strip('"')
            os.environ.setdefault(key.strip(), val)
        return


# 模块导入时只加载一次 .env；后续所有配置读取都基于 os.environ 的当前状态。
_load_dotenv()


def _get_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_str_fallback(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


# ---------------------------------------------------------------- 思考参数的取值与归一（模型与思考参数重构 §4.4 / §4.5.3）

# 思考开关
THINKING_ON = "on"
THINKING_OFF = "off"
THINKING_CHOICES = (THINKING_ON, THINKING_OFF)

# 思考强度（低 → 高）；端点别名归一化映射：
#   minimal→low / xhigh→high / ultra→max（medium/high/max 为本名原样）
EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX = "low", "medium", "high", "max"
EFFORT_CHOICES = (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX)
_EFFORT_ALIASES = {
    "minimal": EFFORT_LOW, "low": EFFORT_LOW,
    "medium": EFFORT_MEDIUM, "high": EFFORT_HIGH,
    "xhigh": EFFORT_HIGH, "max": EFFORT_MAX, "ultra": EFFORT_MAX,
}
# 强度由低到高（降级链与"就近换档"都依赖这个顺序）
EFFORT_ORDER = (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX)

# 面向用户的档位标签（前端与提示文案共用，避免各处自己写映射）
EFFORT_LABELS = {EFFORT_LOW: "低", EFFORT_MEDIUM: "中",
                 EFFORT_HIGH: "高", EFFORT_MAX: "最大"}
THINKING_LABELS = {THINKING_ON: "开", THINKING_OFF: "关"}

_DEPRECATION_WARNED: set = set()


def _warn_deprecated(old: str, new: str) -> None:
    """废弃变量提示（每项进程内只提示一次，避免刷屏）。"""
    if old in _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED.add(old)
    print(f"[config] {old} 已废弃，请改用 {new}（本次仍按旧值生效）")


def normalize_thinking(value: str, default: str = THINKING_ON) -> str:
    """思考开关归一：on/off（大小写、true/false、enabled/disabled 均接受）。"""
    v = (value or "").strip().lower()
    if v in ("on", "true", "1", "yes", "enabled", "enable"):
        return THINKING_ON
    if v in ("off", "false", "0", "no", "disabled", "disable"):
        return THINKING_OFF
    return default


def normalize_effort(value: str, default: str = EFFORT_LOW) -> str:
    """思考强度归一（含官方别名 minimal/medium/xhigh/ultra）；无法识别用 default。"""
    v = (value or "").strip().lower()
    return _EFFORT_ALIASES.get(v, default)


def _thinking_default_env() -> str:
    """`THINKING_DEFAULT`，带**过渡期兼容**读取旧 `THINKING_MODE`。

    理由（重构方案 §4.1）：`THINKING_MODE=disabled` 的用户若被静默改成"开思考"，
    会变慢变贵且没有任何提示——直接影响成本，因此给一条迁移桥 + 废弃告警。
    """
    new = _get_str("THINKING_DEFAULT")
    if new:
        return normalize_thinking(new)
    old = _get_str("THINKING_MODE")
    if old:
        _warn_deprecated("THINKING_MODE", "THINKING_DEFAULT")
        return normalize_thinking(old)
    return THINKING_ON


def _effort_default_env() -> str:
    """`EFFORT_DEFAULT`，带过渡期兼容读取旧 `REASONING_EFFORT`。"""
    new = _get_str("EFFORT_DEFAULT")
    if new:
        return normalize_effort(new)
    old = _get_str("REASONING_EFFORT")
    if old:
        _warn_deprecated("REASONING_EFFORT", "EFFORT_DEFAULT")
        return normalize_effort(old)
    return EFFORT_LOW


@dataclass(frozen=True)
class LLMConfig:
    """主 LLM 配置（OpenAI 兼容接口）。

    「模型与思考参数重构」后的形态：**只有一个模型**，思考由"开关 + 强度"两个
    正交参数表达；不再有 fallback / upgrade / 关键词路由（见
    instructions/Model-Config-Refactor.md）。
    """

    api_key: str = field(default_factory=lambda: _get_str("API_KEY"))
    base_url: str = field(default_factory=lambda: _get_str("BASE_URL").rstrip("/"))
    model_name: str = field(default_factory=lambda: _get_str_fallback("MODEL_NAME", "MODEL"))
    # 思考开关（THINKING_DEFAULT）：on / off。仅当调用方（网页请求头）未指定时生效。
    thinking_default: str = field(default_factory=_thinking_default_env)
    # 思考强度（EFFORT_DEFAULT）：low / medium / high / max。
    effort_default: str = field(default_factory=_effort_default_env)
    # 最大输出上限（MAX_TOKENS）：**上限不是预留**，按实际用量计费。
    # 思考与正式回答共享该额度，设小会导致截断（§4.2.1）。
    max_tokens: int = field(default_factory=lambda: int(_get_str("MAX_TOKENS") or 32768))
    # 并发 LLM 调用上限（MAX_CONCURRENT_LLM，默认 4）：按**凭证指纹**分桶，
    # 同一把 Key 最多同时进行这么多调用，避免打爆上游限流。
    max_concurrent: int = field(
        default_factory=lambda: int(_get_str("MAX_CONCURRENT_LLM") or 4)
    )
    temperature: float = 0.2
    # 生成 32768 tokens 需要数分钟；SSE 下 timeout 只作用于"两块数据间隔"，
    # 因此 180s 是"多久没吐字算超时"，而非总时长限制。
    timeout: int = 180
    retries: int = 3

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model_name)

    @property
    def thinking_on(self) -> bool:
        return normalize_thinking(self.thinking_default) == THINKING_ON


@dataclass(frozen=True)
class VisionConfig:
    """视觉模型配置（图片识别）。

    定位（重构 §4.9）：视觉是**可能完全独立**的一个模型（另一厂商、另一把 Key、
    另一套思考参数）。三项都留空 = 用主模型识图；只填模型名 = 该模型 + 主模型端点/Key。
    原生多模态模型（deepseek-flash / Gemini / GLM 视觉版）自带视觉，通常整组留空即可。
    """

    api_key: str = field(default_factory=lambda: _get_str("VISION_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _get_str("VISION_BASE_URL").rstrip("/"))
    model_name: str = field(default_factory=lambda: _get_str("VISION_MODEL"))
    # 视觉的思考参数（VISION_THINKING / VISION_EFFORT）：
    # **留空 = 自动** —— 视觉与主模型是同一个（host+model+key 全同）时复用主模型设置，
    # 否则用 off（识图要快、要省，不需要长思考）。
    thinking: str = field(default_factory=lambda: _get_str("VISION_THINKING"))
    effort: str = field(default_factory=lambda: _get_str("VISION_EFFORT"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model_name)


@dataclass(frozen=True)
class CompToxConfig:
    """EPA CompTox API 配置。"""

    api_key: str = field(default_factory=lambda: _get_str("COMPTOX_API_KEY"))
    base_url: str = "https://comptox.epa.gov/ctx-api"
    timeout: int = 20

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ServiceConfig:
    """清小搭接入服务（api.py）配置。

    api_key: 服务端 Bearer 密钥（环境变量 SERVICE_API_KEY），
    接入清小搭向导时在「API 密钥」处填同一个值。
    public_base_url: 服务公网地址（环境变量 PUBLIC_BASE_URL，如
    https://your.host），用于拼接附件下载 URL；缺省用请求 Host 推导。
    attachment_dir: 图片附件存放目录。
    attachment_max_bytes: 附件目录总字节上限，超过时删最旧图片
    （默认 2GB；0/负数=不按大小限制）。附件生命周期语义见
    core/attachments.py 模块 docstring（不按时间 TTL 硬删，超配额才删最旧）。
    attachment_max_files: 附件目录最多保留的图片数量（默认 50000；0/负数=不限）。
    """

    api_key: str = field(default_factory=lambda: _get_str("SERVICE_API_KEY"))
    public_base_url: str = field(
        default_factory=lambda: _get_str("PUBLIC_BASE_URL").rstrip("/")
    )
    attachment_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "attachments"
    )
    attachment_max_bytes: int = 2 * 1024 * 1024 * 1024   # 2GB
    attachment_max_files: int = 50000

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class PromptConfig:
    """System prompt 文件路径配置。"""

    system_prompt_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "prompts" / "system_prompt.txt"
    )


@dataclass(frozen=True)
class Settings:
    """项目全局配置集合。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    comptox: CompToxConfig = field(default_factory=CompToxConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)


# 模块级单例。任何导入 core.config 的模块共享同一套配置实例。
settings = Settings()


if __name__ == "__main__":
    print(f"project_root: {settings.project_root}")
    print(f"llm configured: {settings.llm.is_configured}")
    print(f"model_name: {settings.llm.model_name}")
    print(f"thinking: {settings.llm.thinking_default} / effort: {settings.llm.effort_default}")
    print(f"max_tokens: {settings.llm.max_tokens}")
    print(f"vision configured: {settings.vision.is_configured} "
          f"(model={settings.vision.model_name or '（空=用主模型）'})")
    print(f"prompt path: {settings.prompt.system_prompt_path}")
    print(f"comptox configured: {settings.comptox.is_configured}")
