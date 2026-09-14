# -*- coding: utf-8 -*-
"""tests/test_correction.py — 渲染反馈闭环测试：渲染/校验失败时回传 LLM 自动修正。

验证四个行为：
1. 无失败不触发重试（ask_llm 仅调用 1 次）；
2. 校验失败 → 修正重试 → 采用修正版；
3. 渲染失败 → 修正重试 → 采用修正版；
4. 修正仍失败 → 降级输出且不无限重试（最多 max_corrections 次）。
"""

import dataclasses
import types

import pytest

from app import _build_correction_prompt, process_question
from core import credentials
from core.config import settings as real_settings
from renderers import registry


@pytest.fixture
def base_env():
    """注入凭证层基准配置替身（模拟服务器 .env 的思考参数等）。"""

    def _apply(**llm_kw):
        base = dataclasses.replace(
            real_settings, llm=dataclasses.replace(real_settings.llm, **llm_kw))
        credentials.set_base_settings_for_tests(base)
        return base

    yield _apply
    credentials.set_base_settings_for_tests(None)
    credentials._reset_semaphores_for_tests()


class _R:
    """`ask_llm(return_result=True)` 的最小替身（把字符串包成结果对象）。

    辅助调用（重写/修正）仍按 `ask_llm(...)` 的**文本返回**使用其返回值，
    因此这里把若干字符串方法委托给 text，使同一个替身两种用法都成立。
    """

    def __init__(self, text):
        self.text = text
        self.error = None if text else "模拟失败"
        self.downgraded = False
        self.notice = ""
        self.effort_requested = ""
        self.effort_effective = None

    def __bool__(self):
        return bool(self.text)

    def __str__(self):
        return self.text or ""

    # --- 字符串协议委托（子串查找/包含/拼接）---
    def find(self, *a, **k):
        return self.text.find(*a, **k)

    def __contains__(self, item):
        return item in self.text

    def __add__(self, other):
        return self.text + other

    def __radd__(self, other):
        return other + self.text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item]

    def startswith(self, *a, **k):
        return self.text.startswith(*a, **k)

    def strip(self, *a, **k):
        return self.text.strip(*a, **k)

    def __getattr__(self, name):
        """未定义的属性/方法一律委托给 text。

        `_R` 同时服务两类调用点：主生成读 `.text`（return_result=True），
        辅助调用（修正/重写）把返回值当字符串用（如 `re.sub`/`parse_tags`）。
        委托后两种用法都成立。
        """
        attr = getattr(self.text, name)
        return attr if callable(attr) else self.text


def _res(text):
    return _R(text)


@pytest.fixture
def flawed_renderers(monkeypatch):
    """STRUCT 渲染器：c1ccccc1 成功，其余（合法但模拟内部失败）返回失败串。"""

    def render_struct(smiles, label=None, mode="skeleton", subs="",
                      bond="", angle="", charge=""):
        if smiles == "c1ccccc1":
            return "RENDERED:c1ccccc1"
        return f"（结构渲染失败：无法为「{smiles}」生成结构式）"

    original = dict(registry.RENDERER_REGISTRY)
    registry.RENDERER_REGISTRY["STRUCT"] = render_struct
    yield
    registry.RENDERER_REGISTRY.clear()
    registry.RENDERER_REGISTRY.update(original)


def test_no_failure_no_retry(fake_rdkit, fake_renderers, no_aux_calls,
                              monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res("苯是 [STRUCT:c1ccccc1]。"))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 1, "无失败不应触发修正重试"
    assert "RENDERED:c1ccccc1" in result


def test_correction_after_validation_failure(fake_rdkit, fake_renderers,
                                             monkeypatch,
                                                no_aux_calls,):
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # 非法 SMILES → 校验拦截 → 触发修正
        "苯是 [STRUCT:c1ccccc1]。",     # 修正版
    ]
    # 禁用 PubChem 翻译（避免消耗 ask_llm 调用序列）
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res(answers.pop(0)))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 2, "应触发一次修正重试"
    assert "RENDERED:c1ccccc1" in result
    assert "无效 SMILES" not in result
    assert "[STRUCT:XYZABC]" not in result


def test_correction_after_render_failure(fake_rdkit, flawed_renderers,
                                         monkeypatch,
                                                no_aux_calls,):
    calls = []
    answers = [
        "看 [STRUCT:CCl]。",            # CCl 合法但渲染器失败 → 触发修正
        "看 [STRUCT:c1ccccc1]。",       # 修正版
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res(answers.pop(0)))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 2
    assert "RENDERED:c1ccccc1" in result
    assert "结构渲染失败" not in result


def test_correction_exhausted_degrades(fake_rdkit, fake_renderers,
                                       no_aux_calls, monkeypatch):
    """修正耗尽 → 降级（不无限重试）。

    本用例屏蔽了辅助重写路径（`no_aux_calls`），因此调用序列为
    "主生成 + 1 次常规修正"；含手术式重写的完整序列见
    `test_p3_escape_still_works_single_model` 与 tests/test_struct_rewrite.py。
    """
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res("苯是 [STRUCT:XYZABC]。"))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 2, "主生成 + 最多一次常规修正，不应无限重试"
    assert "图示无法渲染" in result


def test_diagnostics_resolved_after_correction(fake_rdkit, fake_renderers,
                                               monkeypatch,
                                                no_aux_calls,):
    """修正救回：diagnostics 记录失败轮（round=0），resolved=True。"""
    calls = []
    answers = ["苯是 [STRUCT:XYZABC]。", "苯是 [STRUCT:c1ccccc1]。"]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res(answers.pop(0)))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(diag) == 1
    assert diag[0]["round"] == 0
    assert diag[0]["resolved"] is True
    assert "无效 SMILES" in diag[0]["reason"]          # 技术细节留给后端
    assert diag[0]["friendly"].startswith("（结构式图示无法渲染，已省略")  # 前端友好版
    assert "RENDERED:c1ccccc1" in result


def test_translate_name_uses_user_model(monkeypatch):
    """★ 实测病例回归：辅助调用（中文化学名→英文翻译）**必须用用户的模型**。

    病例：网页填 `deepseek-flash`，但日志里出现 `.env` 的 `gemini-3.7-flash`
    ——根因是 `_build_correction_prompt → _fetch_pubchem_references →
    _translate_name_zh2en` 三层都没把 `model` 穿下去，翻译调用回退到了
    服务器 `.env` 的模型（用户既未授权、也可能无权访问）。

    本用例钉住整条链：修正 prompt 构造时，翻译调用收到的 `model` 必须是用户那个。
    """
    import types
    seen = {}

    def fake_ask(text, **kw):
        seen["model"] = kw.get("model")
        seen["user_scoped"] = kw.get("user_scoped")
        return _res("benzene")

    monkeypatch.setattr("app.ask_llm", fake_ask)
    monkeypatch.setattr("app.name_to_smiles", lambda name: "c1ccccc1",
                        raising=False)
    monkeypatch.setattr("utils.name_resolver.name_to_smiles",
                        lambda name: "c1ccccc1")
    # 隔离 PubChem 失败缓存（按凭证指纹分桶，避免跨用例串味）
    import app as app_mod
    app_mod._PUBCHEM_FAIL_CACHE.clear()

    failures = [(types.SimpleNamespace(type="STRUCT", args=["XYZABC", "苯"],
                                       raw="[STRUCT:XYZABC,label=苯]",
                                       start_pos=0), "无效 SMILES")]
    with credentials.user_credentials({"api_key": "sk-user",
                                       "model": "deepseek-flash"}):
        app_mod._build_correction_prompt("画出苯", "苯是 [STRUCT:XYZABC]。",
                                         failures, model=None)
    assert seen.get("model") is None, "传 None 时由凭证层解析为用户模型"
    assert seen.get("user_scoped") is True, "★ 必须标记为用户作用域（禁止回退服务器模型）"
    # 凭证层在该作用域下解析出的就是用户模型
    with credentials.user_credentials({"api_key": "sk-user",
                                       "model": "deepseek-flash"}):
        assert credentials.llm_config().model_name == "deepseek-flash"


def test_translate_name_never_falls_back_to_server_model(fake_rdkit,
                                                         monkeypatch):
    """★ 用户作用域内、又没有可用模型名时：**拒绝**用服务器模型（宁可不翻译）。"""
    import types
    import core.llm_client as lc

    calls = []

    def fake_stream(url, headers, payload, on_piece=None):
        calls.append(dict(payload))
        return "benzene", "stop", 0, False

    monkeypatch.setattr(lc, "_stream_chat", fake_stream)
    # 服务器配置：模型名留空 + 端点/密钥给全（模拟"凭证齐全但没有模型名"）
    base = credentials.base_settings()
    credentials.set_base_settings_for_tests(
        dataclasses.replace(
            base, llm=dataclasses.replace(base.llm, model_name="",
                                          base_url="http://x", api_key="k")))
    try:
        with credentials.user_credentials({"api_key": "sk-user"}):
            out = lc.ask_llm("苯", system_prompt="x", thinking="disabled",
                             user_scoped=True)
        assert out is None, "不应偷偷用服务器模型"
        assert not calls, "不应发起任何上游调用"
    finally:
        credentials.set_base_settings_for_tests(None)


def test_diagnostics_include_final_round_unresolved(fake_rdkit,
                                                    fake_renderers,
                                                    no_aux_calls,
                                                    monkeypatch):
    """修正救不回：diagnostics 含最后一轮（round=1，即 max），resolved=False
    ——后端拿到完整失败反馈（含最后一次），前端只见友好降级。

    （辅助重写路径被 `no_aux_calls` 屏蔽，故调用序列为 主生成 + 1 次常规修正。）
    """
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(a[0] if a else None) or _res("苯是 [STRUCT:XYZABC]。"))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(calls) == 2                       # 主生成 + 一次常规修正
    assert len(diag) == 2                       # 两轮失败都被记录（含最后一次）
    assert [d["round"] for d in diag] == [0, 1]
    assert all(d["resolved"] is False for d in diag)
    assert all(d["raw"] == "[STRUCT:XYZABC]" for d in diag)
    # 前端友好、后端拿技术细节
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result
    assert all("无效 SMILES" in d["reason"] for d in diag)


def test_correction_prompt_contains_failure_info():
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = "苯是 [STRUCT:XYZABC]。"
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    prompt = _build_correction_prompt("画苯", text, [(tag, vr.reason)])
    assert "[STRUCT:XYZABC]" in prompt
    assert "无效 SMILES" in prompt
    assert "画苯" in prompt
    assert "修正要求" in prompt


def test_correction_prompt_chem_guidance():
    """化学校验失败时修正 prompt 给出守恒修正提示（氧化/脱氢补物种）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("乙醇氧化：[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS]"            "[STRUCT:O,id=w][ARROW:type=single,Cu, Δ]"            "[STRUCT:CC=O,id=b][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert vr.reason.startswith("化学校验：")
    prompt = _build_correction_prompt("乙醇氧化成乙醛", text, [(tag, vr.reason)])
    assert "两侧原子" in prompt                      # 失败原因
    assert "补物种或调系数" in prompt               # 修正指导
    assert "辅助试剂" in prompt and "箭头条件" in prompt  # 辅助试剂归位规则
    assert "[O]" in prompt and "[H]" in prompt        # 占位符禁止提示
    assert "氧化剂" in prompt                        # 氧化配平规则


def test_correction_prompt_invalid_smiles_guidance():
    """无效 SMILES 时修正 prompt 给出具体改法（配离子
    走分子式轨道——[Ag(NH3)2]+ 直写合法，不再建议拆分组分）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CC=O,id=a][PLUS]"
            "[STRUCT:[Ag(NH3)2OH],id=b][ARROW:type=single,Δ]"
            "[STRUCT:CC(=O)[O-],id=c][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason
    prompt = _build_correction_prompt("银镜反应", text, [(tag, vr.reason)])
    assert "[Ag(NH3)2OH]" in prompt                  # 失败标记原文
    assert "[Ag(NH3)2]+" in prompt                   # 配离子分子式改法
    assert "分子式" in prompt                         # 分子式轨道指引
    assert "省略或文字描述" in prompt                 # 降级策略


def test_correction_prompt_inorganic_salt_guidance():
    """无机盐/含氧酸盐 SMILES 非法时修正 prompt 给出分子式/离子式改法
    （KMnO4 写成 K[Mn](=O)(=O)=O——金属与中心原子无直接键）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS]"
            "[STRUCT:K[Mn](=O)(=O)=O,id=k][ARROW:type=single]"
            "[STRUCT:CC=O,id=b][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason or "化学校验" in vr.reason
    prompt = _build_correction_prompt("乙醇被高锰酸钾氧化", text,
                                      [(tag, vr.reason)])
    assert "K[Mn](=O)(=O)=O" in prompt              # 失败标记原文
    assert "[K+].[O-][Mn](=O)(=O)=O" in prompt      # 离子式改法
    assert "分子式" in prompt                        # 分子式轨道指引


def test_retry_succeeds_after_smiles_fix(fake_rdkit, fake_renderers,
                                         monkeypatch,
                                                no_aux_calls,):
    """修正重试成功闭环：首次无效 SMILES 校验拦截 → 修正版渲染成功。

    注：fake_rdkit 的 _FakeMol 无元素计数，ARROW/REACTION 的守恒校验
    在 fake 下必然拦截——用 [STRUCT:XYZABC]（无效 SMILES，校验明确
    拦截）触发首次失败，验证"改简单标记后重试成功"的闭环。"""
    calls = []
    answers = [
        # 首次：无效 SMILES（配离子/复杂物种写错的一类）→ 校验拦截
        "乙醛氧化：[STRUCT:XYZABC]",
        # 修正版：简单结构渲染成功
        "乙醛氧化：[STRUCT:CCO]",
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or _res(answers.pop(0)))
    result = process_question("写出乙醛发生银镜反应的化学方程式。", max_corrections=1)
    assert len(calls) == 2, "首次失败应触发一次修正重试"
    assert "RENDERED:CCO" in result                # 修正版已渲染
    assert "无法渲染" not in result                # 无降级提示


# ---------------------------------------------------------------- 单模型：不再有"升级/回退模型"路由

def test_single_model_used_for_all_calls(fake_rdkit, fake_renderers, base_env,
                                         no_aux_calls, monkeypatch):
    """全程单模型：主生成与后续修正**都用同一个模型**，且不显式覆盖 model。"""
    base_env(model_name="deepseek-flash")
    calls = []
    answers = ["苯是 [STRUCT:XYZABC]。", "[STRUCT:c1ccccc1]"]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _res(answers.pop(0)))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 2
    # 两次调用都不覆盖 model（由凭证层统一解析为同一模型）
    assert all(c.get("model") is None for c in calls)
    # 修正调用是辅助调用：关思考 + 收紧的 max_tokens
    assert calls[1].get("thinking") == "disabled"
    assert calls[1].get("max_tokens") == 2048
    assert "RENDERED:c1ccccc1" in result


def test_mechanism_question_does_not_switch_model(fake_rdkit, fake_renderers,
                                                  base_env, monkeypatch):
    """★ 用户病例回归：机理题**不得**被换成另一个模型。

    旧实现里"机理"命中关键词路由 → 把 .env 的升级模型发到上游
    （用户既未授权、也可能无权访问该模型名）。单模型后必须彻底消失。
    """
    base_env(model_name="server-model")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(k) or _res("机理是 [STRUCT:c1ccccc1]。"))
    with credentials.user_credentials({"api_key": "sk-user",
                                       "model": "deepseek-flash"}):
        result = process_question("介绍 SN1 反应的机理", max_corrections=1)
    assert len(calls) == 1
    assert all(c.get("model") is None for c in calls), "不再有显式模型切换"
    assert "RENDERED:c1ccccc1" in result


def test_user_model_resolved_through_credentials(base_env):
    """凭证层：用户模型优先于服务器 .env 的模型。"""
    base_env(model_name="server-model")
    with credentials.user_credentials({"api_key": "sk-user",
                                       "model": "deepseek-flash"}):
        assert credentials.user_model() == "deepseek-flash"
        assert credentials.llm_config().model_name == "deepseek-flash"


def test_thinking_params_forwarded_to_main_generation(fake_rdkit,
                                                      fake_renderers,
                                                      monkeypatch):
    """网页传入的思考开关/强度/最大输出必须透传到主生成调用。"""
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _res("苯是 [STRUCT:c1ccccc1]。"))
    process_question("画苯", max_corrections=1, thinking="off",
                     effort="high", max_tokens=12345)
    assert calls[0].get("thinking") == "off"
    assert calls[0].get("effort") == "high"
    assert calls[0].get("max_tokens") == 12345


def test_main_generation_uses_call_site_max_tokens(fake_rdkit, fake_renderers,
                                                   monkeypatch):
    """未指定 max_tokens 时用调用点常量（32768），而不是继承配置的旧默认。"""
    from core.llm_client import MAX_TOKENS_MAIN
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _res("苯是 [STRUCT:c1ccccc1]。"))
    process_question("画苯", max_corrections=1)
    assert calls[0].get("max_tokens") == MAX_TOKENS_MAIN


def test_p3_escape_still_works_single_model(fake_rdkit, fake_renderers,
                                            monkeypatch):
    """单模型下逃生仍生效：同错误重犯 → 不再烧满修正轮次。"""
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",       # 首跑失败
        "（手术重写失败：无有效标记）",    # 手术式结构重写尝试（仍坏）
        "[STRUCT:XYZABC]",               # 常规修正原样重犯（同 fingerprint）
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _res(answers.pop(0)))
    result = process_question("画苯", max_corrections=2)
    assert len(calls) == 3
    assert all(c.get("model") is None for c in calls)
    assert "RENDERED" not in result                  # 未救回 → 降级文本


def test_correction_continues_on_new_error(fake_rdkit, fake_renderers,
                                           monkeypatch):
    """修正后失败原因**变化**（fingerprint 不同）→ 不触发逃生，继续修正救回。"""
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # 首跑失败
        "[STRUCT:XYZABD]",              # 第 1 次修正仍失败（不同原因串）
        "[STRUCT:c1ccccc1]",            # 第 2 次修正成功
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _res(answers.pop(0)))
    result = process_question("画苯", max_corrections=2)
    assert len(calls) == 3
    assert "RENDERED:c1ccccc1" in result


def test_unresolved_failure_diagnostics_single_stage(fake_rdkit,
                                                     fake_renderers,
                                                     monkeypatch):
    """修正救不回 → 降级；诊断的 stage 现在恒为 main（无 upgrade 阶段）。"""
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(k) or _res("苯是 [STRUCT:XYZABC]。"))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result               # 前端友好
    assert diag
    assert {d["stage"] for d in diag if d.get("type") != "NOTICE"} == {"main"}
    assert all(d["resolved"] is False for d in diag if d.get("type") != "NOTICE")


# ---------------------------------------------------------------- 思考档位被静默改写时的诚实上报（§4.5.5）

def test_effort_notice_appended_once(fake_rdkit, fake_renderers, monkeypatch):
    """端点强制思考时：回答末尾追加一次提示，且同会话不重复。"""
    class _Res:
        text = "苯是 [STRUCT:c1ccccc1]。"
        downgraded = True
        notice = "（该模型始终思考，无法关闭；本次按「低」执行）"

    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: calls.append(k) or _Res())
    diag = []
    r1 = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert _Res.notice in r1
    assert any(d.get("type") == "NOTICE" for d in diag)
    # 第二次提问：不再追加（同会话去重）
    r2 = process_question("画苯", max_corrections=1)
    assert _Res.notice not in r2


def test_no_notice_when_not_downgraded(fake_rdkit, fake_renderers, monkeypatch):
    class _Res:
        text = "苯是 [STRUCT:c1ccccc1]。"
        downgraded = False
        notice = ""

    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: _Res())
    r = process_question("画苯", max_corrections=1)
    assert "无法关闭" not in r


# ---------------------------------------------------------------- 部分降级：COMPOSITE 仅 MECHARROW 报错时剔除箭头保留分子

_BAD_MECH_COMPOSITE = (
    "[COMPOSITE:reaction]"
    "[STRUCT:CCl,id=r0,label=CH3Cl][ARROW:type=single]"
    "[STRUCT:CO,id=p0,label=CH3OH]"
    "[MECHARROW:r0:9>r0:0]"   # 9 越界（该分子只有 2 个重原子）→ 仅 MECHARROW 报错
    "[/COMPOSITE]"
)


def test_partial_render_without_mecharrows():
    """部分降级：仅 MECHARROW 报错的 COMPOSITE 剔除箭头后仍渲染出分子图。"""
    pytest.importorskip("rdkit")
    from app import _partial_render_composite_without_mecharrows
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tags

    tags = parse_tags(_BAD_MECH_COMPOSITE)
    _, invalid = validate_tags(tags)
    assert len(invalid) == 1
    assert "MECHARROW" in invalid[0].reason
    out = _partial_render_composite_without_mecharrows(invalid[0].tag)
    assert out is not None
    assert out.startswith("\\begin{tikzpicture}")
    assert "\\begin{scope}" in out          # 分子组件保留
    assert "Stealth[length=2.5mm]" in out  # 主反应箭头保留


def test_partial_render_not_for_non_mecharrow():
    """非 MECHARROW 错误（STRUCT 无效）不触发部分渲染（整体降级）。"""
    pytest.importorskip("rdkit")
    from app import _partial_render_composite_without_mecharrows
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tags

    text = ("[COMPOSITE:reaction][STRUCT:XYZABC,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0][/COMPOSITE]")
    tags = parse_tags(text)
    _, invalid = validate_tags(tags)
    assert "MECHARROW" not in invalid[0].reason
    assert _partial_render_composite_without_mecharrows(invalid[0].tag) is None


def test_mecharrow_only_degrades_partially(monkeypatch):
    """集成：仅 MECHARROW 报错且修正耗尽 → 输出分子图 + 「反应箭头无法渲染」
    提示（而非整图省略）。"""
    pytest.importorskip("rdkit")
    bad = f"机理：{_BAD_MECH_COMPOSITE}"
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: bad)
    result = process_question("画机理", max_corrections=1)
    assert "反应箭头无法渲染，已省略" in result
    assert "\\begin{tikzpicture}" in result           # 分子图保留
    assert "（复合图图示无法渲染" not in result        # 未整体降级


def test_non_mecharrow_still_whole_degrade(monkeypatch):
    """非 MECHARROW 错误修正耗尽 → 仍整体降级（行为不变）。"""
    pytest.importorskip("rdkit")
    bad = "苯是 [STRUCT:XYZABC]。"
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: bad)
    result = process_question("画苯", max_corrections=1)
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result                # 前端友好（无技术细节）


def test_autofix_endpoint_skips_llm_correction(monkeypatch):
    """端到端：唯一候选端点自动修复——主生成后直接渲染，零修正调用。"""
    pytest.importorskip("rdkit")
    calls = []
    bad_tag = ("[COMPOSITE:reaction]"
               "[STRUCT:O=[N+]([O-])C([H])1C=CC=C[CH+]1,id=sg]"
               "[PLUS][STRUCT:O=[N+]([O-])[O-],id=no3]"
               "[ARROW:type=single][STRUCT:O=[N+]([O-])C1=CC=CC=C1,id=nb]"
               "[PLUS][STRUCT:O=[N+]([O-])O,id=hno3]"
               "[MECHARROW:no3:2>sg:4,sg:3-6>sg:3-9][/COMPOSITE]")  # 3-6 应为 3-4
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or f"机理：{bad_tag}")
    result = process_question("磺化脱质子", max_corrections=2)
    assert len(calls) == 1                     # 端点自动修复，无 LLM 修正
    assert "渲染失败" not in result and "无法渲染" not in result
    assert "tikzpicture" in result


def test_correction_prompt_dynamic_sections():
    """修正要求按失败类型裁剪——mech 失败只给端点指引，不掺守恒讲座。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CC(C)(C)[OH2+],id=p]"
            "[MECHARROW:p:3-4>p:4][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "MECHARROW" in vr.reason
    prompt = _build_correction_prompt("脱质子", text, [(tag, vr.reason)])
    assert "直接改写" in prompt                  # mech 指引在场
    assert "禁用 [O]/[H] 占位符" not in prompt      # 守恒段落被裁掉
    assert "配体括号写全" not in prompt             # SMILES 段落被裁掉


def test_patch_rejects_unrelated_rewrite():
    """身份闸门：修正输出是"另一个图"（类型不同或
    id/label 零交集）→ 拒绝该处替换——替换等于删除原图（删内容保合法）；
    同身份修正（保留 id/label）正常替换。"""
    from app import _apply_patch_corrections
    from core.tag_parser import parse_tags
    original = ("共振式：[COMPOSITE:row][BLOCK][STRUCT:XYZABC,id=c1,"
                "label=式 I][/BLOCK][/COMPOSITE]")
    tag = parse_tags(original)[0]
    # 修正输出是毫不相干的另一个 COMPOSITE（id/label 零交集）→ 拒绝替换
    fixed = "[COMPOSITE:row][STRUCT:Cc1ccccc1,id=x,label=甲苯][/COMPOSITE]"
    patched = _apply_patch_corrections(original, [(tag, "无效 SMILES")], fixed)
    assert patched is not None and "XYZABC" in patched
    # 类型不同 → 拒绝替换
    fixed2 = "[STRUCT:Cc1ccccc1]"
    patched2 = _apply_patch_corrections(original, [(tag, "无效 SMILES")], fixed2)
    assert patched2 is not None and "XYZABC" in patched2
    # 同一身份（保留 id/label）→ 正常替换
    fixed3 = ("[COMPOSITE:row][BLOCK][STRUCT:Cc1ccccc1,id=c1,label=式 I]"
              "[/BLOCK][/COMPOSITE]")
    patched3 = _apply_patch_corrections(original, [(tag, "无效 SMILES")], fixed3)
    assert "XYZABC" not in patched3 and "Cc1ccccc1" in patched3


def test_correction_content_loss_flagged(fake_rdkit, fake_renderers,
                                         monkeypatch):
    """内容完整性对账：修正/重写把标记改没了
    （校验全过但内容缺失）→ 按未解决记账 + 回答末尾显式提示，不允许
    "删内容保合法"无声通过。"""
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: "苯是 [STRUCT:c1ccccc1]。[STRUCT:XYZABC]")
    # 结构重写失控：返回无标记文本（原标记被替换没了）
    monkeypatch.setattr("app._rewrite_struct_smiles",
                        lambda *a, **k: "（该图省略）")
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert "未能保留" in result                 # 显式提示用户
    assert "RENDERED:c1ccccc1" in result        # 其余内容正常
    assert any(d["resolved"] is False and d["type"] == "STRUCT"
               and "缺失" in d["reason"] for d in diag), diag


def test_correction_no_loss_no_note(fake_rdkit, fake_renderers, monkeypatch):
    """G2 对照：正常修正（标记数量不变）不出现缺失提示、不产生缺失诊断。"""
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # 非法 SMILES → 触发修正
        "苯是 [STRUCT:c1ccccc1]。",     # 修正版（1 个标记换 1 个标记）
    ]
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: answers.pop(0))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert "未能保留" not in result
    assert not any("缺失" in d["reason"] for d in diag), diag


def test_correction_prompt_special_species_guidance():
    """无效 SMILES 失败时修正 prompt 附特殊
    物种写法词典（酰基正离子 C[C+]=O / sp2 碳负离子 [CH-] / 氧鎓显式 H）；
    离子+自由基簿记冲突（other 类）单独注入 [CH-] 写法。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    # 酰基正离子病例：CC(=O)[Cl-] 无效 SMILES
    text = ("[COMPOSITE:reaction][STRUCT:CC(=O)Cl,id=ac]"
            "[ARROW:type=single][STRUCT:CC(=O)[Cl-],id=acy][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason
    prompt = _build_correction_prompt("傅克酰化", text, [(tag, vr.reason)])
    assert "R[C+]=O" in prompt          # 酰基正离子通式
    assert "丙酰基 CC[C+]=O" in prompt   # 不同碳数示例（防照抄不换 R）
    assert "[CH-]" in prompt            # sp2 碳负离子写法
    assert "[O+]([H])" in prompt        # 氧鎓显式 H 写法
    # 簿记冲突病例（other 类失败）：2 键裸 [C-]
    text2 = "[STRUCT:C1[C-]C=CC=C1,label=苯基负离子]"
    tag2 = parse_tags(text2)[0]
    vr2 = validate_tag(tag2)
    assert "自由基" in vr2.reason and "电荷" in vr2.reason
    prompt2 = _build_correction_prompt("苯基负离子", text2, [(tag2, vr2.reason)])
    assert "[CH-]" in prompt2 and "[C-]" in prompt2


def test_autofix_stereo_skips_llm_correction(monkeypatch):
    """立体枚举修正端到端：CIP 写反 → 自动修正后直接
    渲染，零 LLM 修正调用。"""
    pytest.importorskip("rdkit")
    calls = []
    bad_tag = ("[STRUCT:C[C@H](Cl)[C@@H](Cl)CC,mode=stereo,"
               "label=(2R,3S)-2,3-二氯戊烷]")
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or f"构型：{bad_tag}")
    result = process_question("立体构型测试", max_corrections=2)
    assert len(calls) == 1                     # 自动修正，无 LLM 修正调用
    assert "无法渲染" not in result
    assert "tikzpicture" in result


def test_autofix_balance_gap_skips_llm_correction(monkeypatch):
    """守恒缺口自动补足端到端：漏写 Na+ 反离子 →
    自动补齐后直接渲染，零 LLM 修正调用。"""
    pytest.importorskip("rdkit")
    calls = []
    bad_tag = ("[COMPOSITE:reaction][STRUCT:C#C,label=乙炔,id=b1][PLUS]"
               "[STRUCT:[Na+].[NH2-],label=氨基钠,id=r1][ARROW:type=single]"
               "[STRUCT:[C-]#C,label=乙炔钠,id=b2][PLUS][STRUCT:N,label=氨]"
               "[/COMPOSITE]")
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or f"脱质子：{bad_tag}")
    result = process_question("炔钠制备测试", max_corrections=2)
    assert len(calls) == 1                     # 自动补足，无 LLM 修正调用
    assert "无法渲染" not in result
    assert "tikzpicture" in result
