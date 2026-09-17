# Chem_Agent 测试指南

> 公开版测试入口汇总。覆盖：环境准备、单元测试、渲染器离线 demo、归因工具、
> 基线评测与端到端自测。全部命令在项目根目录执行。

---

## 1. 环境准备

```bash
pip install -r requirements.txt
cp .env.example .env     # 编辑填入 API_KEY / BASE_URL / MODEL_NAME
```

`.env` 核心三件套（完整模板见 `.env.example`）：

```bash
API_KEY=your-api-key
BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-flash
THINKING_DEFAULT=on      # 思考开关 on / off
EFFORT_DEFAULT=low       # 思考强度 low / medium / high / max
MAX_TOKENS=32768         # 上限而非预留，按实际用量计费
# 视觉可选：原生多模态主模型留空即用主模型识图；主模型纯文本时才需要
# VISION_MODEL / VISION_BASE_URL / VISION_API_KEY 三件套
```

LaTeX 工具链（仅当需要 PNG 附件/图片输出时）：
`xelatex`（Debian/Ubuntu: `texlive-xetex texlive-latex-extra texlive-lang-chinese`）+ `poppler-utils`。

> 包内模块统一相对导入：运行包内脚本用 `python -m 包名.模块名`
> （如 `python -m renderers.structure`），不要 `python renderers/structure.py`。

---

## 2. 单元测试

```bash
python -m pytest tests/ -v                    # 全部（900+ 项）
python -m pytest tests/test_parsing.py -v     # 单文件
python -m pytest tests/ -k validator -v       # 按关键词筛选
```

测试文件按用途分组：

- 解析/校验：`test_parsing.py`、`test_validator.py`、`test_bad_inputs.py`、`test_corpus.py`
- 渲染器：`test_composite.py`、`test_mech_arrows.py`、`test_layout.py`、`test_mol_labels.py`、
  `test_mol_widths.py`、`test_bond_segments.py`、`test_lp_rules.py`、`test_energy_roles.py`、
  `test_newman.py`、`test_chair.py`、`test_free_hydrogen.py`、`test_collide.py`
- 管线/服务：`test_smoke.py`、`test_correction.py`、`test_api.py`、`test_web_api.py`、
  `test_attachments.py`、`test_llm_client.py`、`test_electron_sim.py`、`test_arrow_rewrite.py`
- 示例审计：`test_examples_audit.py`（见 §6）

> `pytest.ini` + `tests/conftest.py` 覆盖了内置 `tmp_path`：系统 temp 可写时沿用
> pytest 原语义，不可写（容器/CI 沙箱）时回退到工作区内
> （`CHEM_AGENT_TEST_TMP` 可指定别处）。

---

## 3. 渲染器离线 demo（看真实 TikZ 输出）

每个渲染器都有 `__main__` 入口，打印正例（完整 TikZ）与反例（错误提示），
可直接复制到 Overleaf 编译核对：

```bash
python -m renderers.structure      # STRUCT（苯+label / 乙酸 / 无效 SMILES）
python -m renderers.composite      # COMPOSITE（SN2 / energy / row / 共振式 / 错误）
python -m renderers.energy         # ENERGY（SN2 3 点 / 两步 5 点 / 无效）
python -m renderers.lewis          # LEWIS（水 / 氨 / 甲醇 / 无效）
python -m renderers.stereo         # STEREO（R-乳酸 / S-氨基丁酸 / 无手性）
python -m renderers.chair          # CHAIR（纯骨架 / Br ax / CH3 eq / flip 镜像 / 非环己烷）
python -m renderers.newman         # NEWMAN（60° 交叉 / 0° 重叠）
python -m renderers.layout         # 布局引擎（layout_row 组件序列）
```

校验 + 渲染单段标记文本（离线重放管线判定，逐标记给出校验结果 + 语义规则
trace + 渲染状态，退出码 0/1）：

```bash
python -m core.replay "乙烷：[STRUCT:CC,label=乙烷]"   # 直接传文本
python -m core.replay 某文件.md                         # 传文件
python -m core.replay --full "含标记的文本"              # 附完整 TikZ
python app.py "问题" | python -m core.replay -          # 管道 stdin
```

---

## 4. 基线评测（`core/metrics.py`）

```bash
python -m core.metrics --questions-file fast_test/questions.txt
python -m core.metrics --questions-file fast_test/questions.txt --detail-file detail.txt
python -m core.metrics --questions-file fast_test/questions.txt --json-out corpus.json
```

- `fast_test/questions.txt`：基线问题集（每行一题，`#` 注释行）；每种标记类型至少
  1 题，遇到新失败模式就把对应问题追加进去固化。
- `--json-out`：把原始统计（含 LLM 输出全文）导出 JSON，供规则归因统计复用。

校验规则归因统计（哪条规则拦了多少、多少本可确定性自动修复）：

```bash
python -m core.rule_stats corpus.json --out report_rule_stats.txt
```

---

## 5. 端到端自测

```bash
# Streamlit 本地界面
streamlit run streamlit_app.py

# HTTP 服务（OpenAI 兼容 /v1 + 公开网页 BYOK 同一进程）
uvicorn api:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000/chat
```

`/v1` 自测 checklist（需 `.env` 配置 `SERVICE_API_KEY`）：

```bash
KEY="你的 SERVICE_API_KEY"

# 1) 连通性 + 凭证（期望 200；错误凭证 401）
curl -i http://localhost:8000/v1/models -H "Authorization: Bearer $KEY"

# 2) 非流式（期望 JSON 含 choices[0].message.content 与 usage）
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"画出苯的结构式"}]}'

# 3) 流式（期望多帧 data: 且以 data: [DONE] 结尾）
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"stream":true,"max_tokens":1,"messages":[{"role":"user","content":"你好"}]}'
```

检查要点：Bearer 鉴权、SSE 帧序（role → reasoning → content → stop+[DONE]）、
`x_soda.attachments` 图片附件（非流式挂响应顶层、流式挂 stop 帧，`fileUrl`
可直接 GET 下载 PNG）。部署（Nginx/HTTPS/systemd 与验收清单）见 `docs/deploy.md`。

---

## 6. 示例一致性审计（`tests/test_examples_audit.py`）

把 `prompts/*`、`README.md`、`docs/`、`renderers/core/utils` 源码 docstring 与
`__main__` demo 中的**全部标记示例**提取出来，逐个过真实契约校验 + 化学校验，
防止"自带示例写错"。

```bash
python -m pytest tests/test_examples_audit.py -v
```

挂了优先修示例本身；若被挂的是**格式占位说明**或**故意错误示例**（反面教材），
扩充测试文件中的占位符正则 `_PLACEHOLDER_RE` 或上下文关键词 `_SKIP_CTX_RE`
豁免——过滤原则是"宁可漏报、不可误报"。
