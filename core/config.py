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


@dataclass(frozen=True)
class LLMConfig:
    """主 LLM 配置（OpenAI 兼容接口）。"""

    api_key: str = field(default_factory=lambda: _get_str("API_KEY"))
    base_url: str = field(default_factory=lambda: _get_str("BASE_URL").rstrip("/"))
    model_name: str = field(default_factory=lambda: _get_str_fallback("MODEL_NAME", "MODEL"))
    temperature: float = 0.2
    # 推理模型（deepseek-reasoner 等）的 reasoning + content 共享 max_tokens 预算；
    # 2048 会被长思考链吃光导致 content 为空/截断，提到 8192。
    max_tokens: int = 8192
    # 生成 8192 tokens 需要 1~3 分钟，60s 会误杀正常生成（实测 Read timed out）。
    timeout: int = 180
    retries: int = 3

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model_name)


@dataclass(frozen=True)
class VisionConfig:
    """视觉模型配置（图片识别结构式）。未独立配置时回退到主 LLM。"""

    api_key: str = field(
        default_factory=lambda: _get_str_fallback("VISION_API_KEY", "API_KEY")
    )
    base_url: str = field(
        default_factory=lambda: _get_str_fallback("VISION_BASE_URL", "BASE_URL").rstrip("/")
    )
    model_name: str = field(
        default_factory=lambda: _get_str_fallback(
            "VISION_MODEL", "MODEL_NAME", "MODEL"
        )
    )

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
    attachment_dir: 图片附件存放目录（清小搭拉取后可清理）。
    attachment_ttl: 附件保留秒数（清小搭会立即转存到自己 OSS，默认 1 小时足够）。
    """

    api_key: str = field(default_factory=lambda: _get_str("SERVICE_API_KEY"))
    public_base_url: str = field(
        default_factory=lambda: _get_str("PUBLIC_BASE_URL").rstrip("/")
    )
    attachment_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "attachments"
    )
    attachment_ttl: int = 3600

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
    print(f"vision configured: {settings.vision.is_configured}")
    print(f"vision model_name: {settings.vision.model_name}")
    print(f"prompt path: {settings.prompt.system_prompt_path}")
    print(f"comptox configured: {settings.comptox.is_configured}")
