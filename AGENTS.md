# AGENTS.md — 写给 Agent 的工作指南

## 项目是什么

Chem_Agent 是 LLM 的化学可视化增强引擎：LLM 在回答中嵌入渲染标记
（`[STRUCT]` / `[COMPOSITE]` / `[ENERGY]` / `[REASONING]`），管线负责
**解析 → 契约校验 → 失败回传修正 → 渲染 TikZ → 注入回答**。项目目标与
架构见 `README.md`，测试入口见 `docs/testing.md`，部署见 `docs/deploy.md`。

## 怎么工作

1. **先跑通再改动**：`pip install -r requirements.txt` 后
   `python -m pytest tests/ -q` 应全绿（个别环境性用例除外）。
2. **复杂工作列 Todos 或分派子任务**，并实时更新进度。
3. **交付 = 修改的文件 + 验证方法**。改动完成后必须验证生效：
   - 代码改动：`python -m pytest tests/ -q`（至少跑受影响文件的测试）；
   - 渲染器改动：另跑 `python -m renderers.<改动模块>` 查看真实 TikZ 输出
     （可复制到 Overleaf 编译核对），并跑
     `python -m pytest tests/test_examples_audit.py` 确认自带示例仍合法；
   - 标记语法 / prompt 改动：必须过示例一致性审计（同上）。
4. 工作中产生的临时测试文件可以写，但**完成后清除**。

## 代码约定

- **最小改动**：不碰与任务无关的代码；发现其他问题先提出，不顺手改。
- **注释 = 当前契约 + 为什么**：不写日期、不写工单号、不写"修复"编年史
  （历史在 git 里）；引用文档时指名文档。
- **术语**（指运行时行为，与模型路由无关——项目早已取消"回退模型"）：
  `降级`=用户可见的质量退让（友好降级提示、思考档位降级链）；
  `回退`=失败后切换备选路径（解析失败回退纯化学式、回退默认实现）；
  `兜底`=接住漏网之鱼的安全网（校验已拦截时渲染兜底跳过、异常兜底为
  error 帧）。
- 包内模块统一相对导入，运行包内脚本用 `python -m 包名.模块名`。
- 不引入类型错误抑制（`as any`、`@ts-ignore` 之类），不留空 `except`。

## 不可破坏的契约

- **`/v1/*` 的 OpenAI 兼容协议与 Bearer 鉴权**（清小搭等平台接入方依赖：
  SSE 帧序、`x_soda.attachments` 位置、`finish_reason` 白名单值）。
- **BYOK 安全边界**：用户凭证只走请求头、只存请求作用域 contextvar，
  不落盘不回显；SSRF 校验、出站不跟随重定向、限流、LaTeX 源码闸门与
  文件访问限制——这些闸门**不许绕过**，改动必须同步更新对应安全测试。
- **标记语法向后兼容**：`prompts/` 是 LLM 看到的契约，改语法必须同步改
  `core/tag_parser.py` / `core/tag_validator.py` / 渲染器 / 示例审计。
- 不提交密钥与内网地址；`.env`、真实端点、服务器信息永远不进仓库。
