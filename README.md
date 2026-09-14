# Chem_Agent：给 LLM 装上化学的“眼睛”和“画笔”

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 项目简介

**Chem_Agent** 是一个为 LLM（大语言模型）设计的化学可视化增强引擎。它让 LLM 不仅能“思考”化学，更能“画出”化学。

当前，LLM 已能流畅解答各学科的专业问题，但在化学领域存在一个核心痛点：**输出端无法精准绘制反应机理、电子转移、势能面等图示内容**。Chem_Agent 通过一套轻量级的“渲染标记语言（Tag-based Rendering Language）”，让 LLM 在输出文字回答的同时，自动生成对应的化学结构式、反应式、机理图等可视化内容，实现真正的“图文并茂”。

> **核心理念**：LLM 是大脑（理解、推理、决策），Chem_Agent 是画笔（解析指令、渲染图像），也是校验尺（**生成 → 校验 → 修正 → 交付的可靠性闭环**——不让错误图示静默到达用户）。

---

## 🎯 项目目标

项目本体是一条**化学问答管线**：用户输入化学问题（可含文字、结构式图片），
系统返回**逻辑清晰、图文并茂**的回答。管线末端可接多种形态，当前实现三种：

| 末端 | 形态 | 入口 |
| :--- | :--- | :--- |
| **HTTP 服务（OpenAI 兼容）** | 接入清小搭等 AI 平台（`/v1/*`，Bearer 鉴权，SSE 流式，图片附件） | `api.py` |
| **公开网页（BYOK）** | 任意访客填自己的 API Key 即用（`/chat`，密钥只存浏览器） | `api.py` + `web/index.html` |
| **本地 Streamlit 界面** | 开发调试与本地使用 | `streamlit_app.py` |

三种形态共用同一条管线（LLM → 标记解析 → 契约校验 → 修正 → 渲染 → 注入），
互不影响。

---

## 🔍 解决的问题

| 维度 | 现有 LLM 的局限 | Chem_Agent 的解决方案 |
| :--- | :------------- | :------------------- |
| **结构式** | 无法绘制键线式、Lewis 结构、楔形式、椅式构象、Newman 投影 | `[STRUCT:SMILES]` + mode 绘制全部画法变体 |
| **反应式/机理** | 无法绘制配平的反应式、电子转移弯箭头、多步序列 | `[COMPOSITE]` 容器：组件 + 箭头 + 机理弯箭头统一坐标系组装 |
| **共振与立体** | 无法绘制共振块、构象翻转对比 | `[BLOCK]` 共振块、row 布局横向排列 |
| **能量变化** | 无法绘制势能面、驻点结构 | `[ENERGY:点序列]` + energy 布局驻点挂载 |
| **推理过程** | 黑盒输出，复杂机理直接写标记易错 | `[REASONING]...[/REASONING]` 思考规划空间：复杂图先在此规划步骤、数清原子编号（内容不进入最终回答） |
| **可靠性** | 化学图示错了也静默通过 | 标记契约校验层（SMILES/守恒/引用）+ 失败自动回传修正 |

---

## 🧠 实现方式

### 三阶段策略

**第一阶段：拆分**——所有化学图示拆解为最小单元（结构组件、连接符、机理箭头、标注），符号的选用和组装完全由 LLM 自主决策。

**第二阶段：标记**——通过精心设计的 System Prompt，引导 LLM 将化学符号以标准标记语法嵌入文本：

```text
[STRUCT:c1ccccc1,label=苯]               → 绘制带标注的苯环（芳香小写画圈）
[STRUCT:CC,mode=newman,bond=0-1,angle=60] → 绘制乙烷交叉式纽曼投影
[COMPOSITE:reaction]                      → 苯的硝化反应式：
  [STRUCT:C1=CC=CC=C1,label=苯][PLUS][STRUCT:O=[N+]([O-])O,label=硝酸]
  [ARROW:type=single,浓H2SO4, Δ]
  [STRUCT:O=[N+]([O-])C1=CC=CC=C1,label=硝基苯][PLUS][STRUCT:O,label=水]
[/COMPOSITE]
[ENERGY:0,108,-20]                        → 绘制三点势能面图
[REASONING]亲电取代机理...[/REASONING]     → 思考规划空间（不进入最终回答）
```

**第三阶段：渲染**——渲染引擎解析标记，基于 RDKit 计算分子几何，自绘 TikZ（键线式、孤对电子、弯箭头、布局防重叠），最终注入回答输出。

### 可靠性闭环（核心差异化）

```mermaid
graph LR
    U["用户输入化学问题"] --> L["LLM 推理<br>System Prompt 驱动"]
    L --> T["输出含标记的回答"]
    T --> P["标记解析器"]
    P --> V{"契约校验<br>SMILES / 守恒 / 引用"}
    V -- 通过 --> R["渲染器<br>生成 TikZ"]
    V -- 失败 --> C["回传 LLM 修正<br>（附失败原因）"]
    C --> P
    R --> O["图文混合回答<br>+ PNG 附件"]
```

- **契约校验层**：渲染前拦截非法 SMILES、原子/电荷不守恒、越界引用、格式错误；
- **失败回传修正**：失败标记连同原因回传 LLM 部分修正（不重写全文），修正耗尽则友好降级（宁可不画，不画错）；
- **单模型 + 思考档位**：整个服务只用一个模型（DeepSeek V4.1 Flash 已在性能/成本上全面超越 V4 Pro）；
  成本与延迟由用户可见的两个正交开关控制——**思考开关**（开/关）与**思考强度**（低/中/高/最大）；
  端点不配合（强制思考、不认某档位）时自动摘字段适配并如实告知，不静默照做。

---

## 🏗️ 技术架构

```text
/chem_agent
├── api.py                     # 清小搭接入服务（FastAPI，OpenAI 兼容协议）
├── app.py                     # 端到端管线：LLM → 解析 → 校验 → 修正 → 渲染 → 注入
├── streamlit_app.py           # Streamlit 网页入口
├── core/
│   ├── config.py              # 统一配置入口（模型 + 思考开关/强度/最大输出 + 视觉组）
│   ├── credentials.py         # 每请求凭证与参数（BYOK）：请求头 → contextvar
│   ├── capabilities.py        # 端点能力表（思考参数的学习与记忆，按 端点+模型 分键）
│   ├── llm_client.py          # LLM API 调用封装（流式、思考档位映射与降级、字段自适应）
│   ├── tag_parser.py          # 标记解析器
│   ├── tag_validator.py       # 标记契约校验层
│   ├── tag_injector.py        # 标记→TikZ 注入替换
│   ├── prompt_manager.py      # System Prompt 管理
│   ├── electron_sim.py        # 电子流模拟器（机理箭头自洽性深层校验）
│   ├── attachments.py         # TikZ→PNG 附件构建（编译、托管、超配额回收）
│   ├── answer_cache.py        # 多轮对话标记恢复（渲染后文本 → 原始标记）
│   ├── web_api.py             # 公开网页后端（/api/chat、BYOK 凭证、限流、会话附件）
│   ├── replay.py              # 离线重放工具（标记管线归因：校验 trace + 渲染状态）
│   ├── diaglog.py             # 诊断日志（失败标记与原始输出落盘，0600 滚动）
│   └── metrics.py             # 基线评测（标记遵循率 / 端到端管线）
├── renderers/
│   ├── registry.py            # 标记调度表
│   ├── layout.py              # 统一坐标布局引擎
│   ├── mol_primitives.py      # 共享分子绘制原语（键线/孤对电子/弯箭头/标签）
│   ├── composite.py           # [COMPOSITE] 容器式复合标记渲染器（核心）
│   ├── structure.py           # [STRUCT] 渲染器（分子家族 mode 分派）
│   ├── lewis.py / stereo.py / chair.py / newman.py   # 画法变体
│   ├── energy.py              # [ENERGY] 势能面渲染器
│   ├── hbond.py               # 氢键渲染器
│   └── collide.py             # 占据注册表（布局避让）
├── utils/
│   ├── rdkit_utils.py         # RDKit 验证工具
│   ├── latex_compile.py       # TikZ 编译为 PNG
│   ├── ocr_utils.py           # 图片多模态理解（视觉模型）
│   ├── name_resolver.py       # 化学名称 → SMILES（PubChem 兜底）
│   └── tempdir.py             # 临时目录探测（沙箱/只读 /tmp 环境兜底）
├── prompts/
│   ├── system_prompt.txt      # 核心 System Prompt
│   └── Instruction-for-SMILES.md   # SMILES 书写规范
├── web/
│   └── index.html             # 公开网页（BYOK，自包含单文件，无需构建）
├── deploy/                    # 部署模板（systemd 单元 + Nginx 反代）
├── tests/                     # 单元测试（700+ 项，含示例一致性审计）
├── legacy/                    # MVP 旧代码（存档保留）
└── requirements.txt
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Conda（推荐）或 pip
- LaTeX 工具链（仅当需要 PNG 附件/图片输出时）：
  `xelatex`（Debian/Ubuntu: `texlive-xetex texlive-latex-extra texlive-lang-chinese`）+ `poppler-utils`

### 安装步骤

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd chem_agent

# 2. 创建并激活 Conda 环境
conda create -n chem_agent python=3.9
conda activate chem_agent

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API_KEY / BASE_URL / MODEL_NAME
#   MODEL_NAME=deepseek-flash      ← 整个服务只用一个模型（旧名 v4-flash/v4-pro 已退役）
#   THINKING_DEFAULT=on            ← 思考开关 on/off（网页用户可各自覆盖）
#   EFFORT_DEFAULT=low             ← 思考强度 low/medium/high/max
#   MAX_TOKENS=32768               ← 上限而非预留，按实际用量计费
# 可选：VISION_*（仅当主模型不支持图片识别时才需要；原生多模态模型留空即可）
# 可选：CHEM_AGENT_TMPDIR（只读 /tmp 的容器里指定可写临时目录；见 docs/deploy.md §2.2）
# 接入清小搭时还需设置 SERVICE_API_KEY（服务端密钥）
# 已废弃：FALLBACK_MODEL_NAME / UPGRADE_MODEL_NAME / UPGRADE_KEYWORDS（代码不再读取）；
#         THINKING_MODE / REASONING_EFFORT 仅作过渡期兜底——新变量未设置时才读取
#         并打废弃告警，建议尽快改名

# 5. 验证安装
python -c "from utils.rdkit_utils import validate_smiles; print(validate_smiles('C'))"
# 预期输出: True
```

### 本地测试

```bash
# 运行 Streamlit 界面（本地调试）
streamlit run streamlit_app.py

# 或运行清小搭接入服务（OpenAI 兼容协议）
uvicorn api:app --host 0.0.0.0 --port 8000
```

### API 调用示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer 你的SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "画出苯的结构式，并说明其分子式"}],
    "stream": false
  }'
```

服务特性：Bearer 鉴权、SSE 流式、多模态图片/音频/文件输入（OCR 理解结构式图片）、`x_soda.attachments` 图片附件输出（TikZ→PNG 编译 + 托管下载）。

### 公开网页（BYOK：用户填自己的 API Key）

除清小搭接入外，服务同时提供**面向任意用户的公开网页**——不需要服务器提供任何
密钥，访问者填自己的 OpenAI 兼容凭证即可使用，由他自己的账号计费：

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# 浏览器打开：http://localhost:8000/chat
```

- **界面**：自包含单页（`web/index.html`，零构建、零 CDN 依赖）——设置面板填
  `API Key / 接口地址 / 模型` 与两个思考控件 `思考（开/关）`、`思考强度（低/中/高/最大）`，
  可选 `最大输出`、`视觉模型`（**留空即用主模型识图**）、`视觉端点/Key/思考参数`；
  「测试连接」会顺带**探测该端点的思考能力**（能否关闭、是否接受档位）并显示结论。
  对话流式输出，图示以 PNG 内联显示，支持上传结构式图片、多轮追问、会话本地保留。
- **密钥处理**：只存在浏览器 `sessionStorage`（关标签页即失效），随请求头
  `X-Chem-*` 一次性发给服务端，**服务端不写日志、不落盘、不回显**（日志只打
  `sk-abc…f3d2` 形式的指纹）。凭证经 `core/credentials.py` 放进请求作用域的
  contextvar，整条管线（生成 → 校验 → 修正 → 渲染）全程只用该用户的凭证。
- **隐私**：为了排查化学图示渲染/校验失败，服务端会把**提问原文**（带图时含模型
  对图片的识别结果）与**模型原始输出**记入 `data/diagnostics.jsonl`（权限 0600、
  8MB 滚动覆盖、仅运维可读）——**不含 API Key**。页面设置面板里有对应的「隐私说明」
  告知。对话正文本身只保存在访客自己的浏览器里（IndexedDB）。
- **安全**：用户自定义的接口地址做 SSRF 校验（拒绝内网/回环/云元数据地址）、
  每 IP 限流（默认 20 次/分钟，客户端身份取 uvicorn 净化后的对端地址，
  **不采信客户端自报的 `X-Forwarded-For`**）、问题长度与图片数上限；BYOK 路径
  **绝不继承**服务器 `.env` 的模型/端点/视觉配置（不会把服务器配置静默施加到用户自己的 key）。
  出站请求**不跟随重定向**（3xx 一律当失败）——否则 302 可以绕过上面那道 SSRF
  校验；因此「接口地址」要填**最终的**完整地址。LaTeX 编译另有文件访问限制与
  源码闸门，见 `docs/deploy.md` §2.6。
- **与清小搭互不影响**：`/v1/*` 契约与鉴权一字未改，两类调用方共用同一条
  化学渲染与契约校验管线。
- 端点：`GET /chat`（或 `/web`）页面、`POST /api/chat`（`stream=true` 走 SSE）、
  `GET /api/web-config`、`GET /api/session/{会话}/{uuid}.png`。

部署（Nginx/HTTPS/systemd 与验收清单）见 `docs/deploy.md`。

---

## 📦 核心依赖

| 库 | 用途 |
| :- | :--- |
| `rdkit` | 化学信息学核心（SMILES 解析验证、分子几何计算） |
| `fastapi` + `uvicorn` | HTTP 服务（清小搭接入） |
| `streamlit` | 本地 Web 界面（调试用） |
| `requests` | LLM API 调用（OpenAI 兼容，SSE 流式） |
| `python-dotenv` | 环境变量管理 |
| `PyMuPDF` | PDF → PNG（图片附件） |
| `curl_cffi` | PubChem 访问（名称→SMILES 兜底） |

---

## 🛣️ 开发路线图

| 阶段 | 目标 | 状态 |
| :--- | :--- | :--- |
| **MVP** | 名称→SMILES→TikZ 结构式渲染 | ✅ 已完成 |
| **Phase 1** | 转型为 LLM 驱动 + 标记解析器 | ✅ 已完成 |
| **Phase 2** | FastAPI 适配 + 清小搭接入 | ✅ 已完成 |
| **Phase 3** | 标记面收敛（STRUCT 家族 + COMPOSITE）+ 可靠性工程（契约校验 / 修正闭环 / 模型路由） | ✅ 已完成 |
| **Phase 4** | 化学正确性增强（确定性化学规则校验、LLM 复审、机理箭头样式优化） | 🚧 进行中 |
| **Phase 5** | 开放使用形态：公开网页 BYOK（用户自带 API Key）+ 每请求凭证隔离 + 部署模板 | ✅ 已完成 |
| **Phase 6** | 模型与思考参数重构：单模型（删 fallback/upgrade/关键词路由）+ 思考开关×强度暴露给用户 + 端点能力表与参数自适应 + 视觉组独立配置 | ✅ 已完成 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。如有疑问，请联系项目维护者。

---

## 📄 许可证

[MIT](LICENSE)

---

## 🙏 致谢

- [RDKit](https://www.rdkit.org/) - 化学信息学工具包
- [清小搭](https://www.xiaoda.tsinghua.edu.cn) - AI 智能体平台
