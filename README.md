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

最终交付一个可接入清小搭等 AI 平台的 HTTP 服务，用户输入一个化学问题（可包含文字、结构式图片），系统返回一个**逻辑清晰、图文并茂**的回答。

---

## 🔍 解决的问题

| 维度 | 现有 LLM 的局限 | Chem_Agent 的解决方案 |
| :--- | :------------- | :------------------- |
| **结构式** | 无法绘制键线式、Lewis 结构、楔形式、椅式构象、Newman 投影 | `[STRUCT:SMILES]` + mode 绘制全部画法变体 |
| **反应式/机理** | 无法绘制配平的反应式、电子转移弯箭头、多步序列 | `[COMPOSITE]` 容器：组件 + 箭头 + 机理弯箭头统一坐标系组装 |
| **共振与立体** | 无法绘制共振块、构象翻转对比 | `[BLOCK]` 共振块、row 布局横向排列 |
| **能量变化** | 无法绘制势能面、驻点结构 | `[ENERGY:点序列]` + energy 布局驻点挂载 |
| **推理过程** | 黑盒输出，复杂机理直接写标记易错 | `[REASONING]...[/REASONING]` 思考规划空间：复杂图先在此规划步骤、数清原子编号，前端折叠展示 |
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
[REASONING]亲电取代机理...[/REASONING]     → 折叠展示推理过程
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
- **模型路由**：简单题走 flash、难题关键词直 pro、flash 失败自动升级 pro 修正，控制成本。

---

## 🏗️ 技术架构

```text
/chem_agent
├── api.py                     # 清小搭接入服务（FastAPI，OpenAI 兼容协议）
├── app.py                     # 端到端管线：LLM → 解析 → 校验 → 修正 → 渲染 → 注入
├── streamlit_app.py           # Streamlit 网页入口
├── core/
│   ├── config.py              # 统一配置入口
│   ├── llm_client.py          # LLM API 调用封装（流式、思考参数降级、回退链）
│   ├── tag_parser.py          # 标记解析器
│   ├── tag_validator.py       # 标记契约校验层
│   ├── tag_injector.py        # 标记→TikZ 注入替换
│   ├── prompt_manager.py      # System Prompt 管理
│   ├── attachments.py         # TikZ→PNG 附件构建（编译、托管、TTL 清理）
│   └── metrics.py             # 基线评测（标记遵循率 / 路由评估）
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
│   └── name_resolver.py       # 化学名称 → SMILES（PubChem 兜底）
├── prompts/
│   ├── system_prompt.txt      # 核心 System Prompt
│   └── Instruction-for-SMILES.md   # SMILES 书写规范
├── tests/                     # 单元测试（500+ 项，含示例一致性审计）
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
# 编辑 .env，填入你的 API_KEY / BASE_URL / MODEL_NAME
# 可选：FALLBACK_MODEL_NAME（回退模型）、UPGRADE_MODEL_NAME（升级模型，路由用）
# 接入清小搭时还需设置 SERVICE_API_KEY（服务端密钥）

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
