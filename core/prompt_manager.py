# -*- coding: utf-8 -*-
"""core/prompt_manager.py — System Prompt 加载与管理。"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
_ARROW_INSTRUCTION_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "Instruction-for-Arrows.md"
)
_STRUCTURE_INSTRUCTION_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "Instruction-for-Structure.md"
)
_SMILES_INSTRUCTION_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "Instruction-for-SMILES.md"
)


def _load_instruction(path: Path) -> str:
    """加载独立 instruction 文件；不存在时返回空串。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_arrow_instructions() -> str:
    """加载箭头使用规范。"""
    return _load_instruction(_ARROW_INSTRUCTION_PATH)


def load_structure_instructions() -> str:
    """加载结构式使用规范。"""
    return _load_instruction(_STRUCTURE_INSTRUCTION_PATH)


def load_smiles_instructions() -> str:
    """加载 SMILES 书写规范。"""
    return _load_instruction(_SMILES_INSTRUCTION_PATH)


def load_hbond_instructions() -> str:
    """氢键规范已并入主提示（20260821 重写：Instruction-for-Hbonds.md 删除）。

    保留函数签名供外部兼容；返回空串。
    """
    return ""


def load_system_prompt() -> str:
    """从 prompts/system_prompt.txt 加载系统提示，并追加箭头与结构式规范。

    文件不存在时返回空串，调用方可据此降级。箭头与结构式规范作为独立文件
    维护，避免系统提示过长且便于后续更新。
    """
    if not _PROMPT_PATH.exists():
        return ""
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    arrow_instructions = load_arrow_instructions()
    if arrow_instructions:
        system_prompt = (
            system_prompt
            + "\n\n"
            + "============================================================\n"
            + "箭头使用规范（来自 prompts/Instruction-for-Arrows.md）\n"
            + "============================================================\n\n"
            + arrow_instructions
        )

    structure_instructions = load_structure_instructions()
    if structure_instructions:
        system_prompt = (
            system_prompt
            + "\n\n"
            + "============================================================\n"
            + "结构式使用规范（来自 prompts/Instruction-for-Structure.md）\n"
            + "============================================================\n\n"
            + structure_instructions
        )

    smiles_instructions = load_smiles_instructions()
    if smiles_instructions:
        system_prompt = (
            system_prompt
            + "\n\n"
            + "============================================================\n"
            + "SMILES 书写规范（来自 prompts/Instruction-for-SMILES.md，权威参考）\n"
            + "============================================================\n\n"
            + smiles_instructions
        )

    return system_prompt


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
