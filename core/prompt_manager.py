# -*- coding: utf-8 -*-
"""core/prompt_manager.py — System Prompt 加载与管理。"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    """从 prompts/system_prompt.txt 加载系统提示文本。

    文件不存在时返回空串，调用方可据此降级。
    """
    if not _PROMPT_PATH.exists():
        return ""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_full_prompt(user_question: str) -> str:
    """组装 system + user 为完整提示串。

    用于非 chat 接口或调试；chat 接口应将 system 与 user 分作两条消息。
    """
    sys_prompt = load_system_prompt()
    if not sys_prompt:
        return user_question
    return f"{sys_prompt}\n\n---\n用户问题：{user_question}"


if __name__ == "__main__":
    sp = load_system_prompt()
    print(f"system_prompt 长度: {len(sp)} 字符")
    print(f"前 200 字符预览:\n{sp[:200]}...")
    print("\n--- build_full_prompt 测试 ---")
    full = build_full_prompt("画出苯")
    print(f"完整提示长度: {len(full)} 字符")
