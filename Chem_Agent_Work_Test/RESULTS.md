# 实验记录：2026-10-17

## 当前结论

本地 RDKit 生成和图片工具读取已通过。用户回复“是的”，确认当前聊天实际
显示了“文字 → 苯图 → 文字 → 硝基苯图 → 文字”，因此 A、B 在当前 Codex
界面通过。随后 C 复用现有核心生成两张机理 PNG；用户再次确认实际顺序为
“文字 → 图片 → 文字 → 图片 → 文字”，因此 C 的本地调用与当前聊天混排也通过。
此后用户在讨论 Local Work 原样复测后报告“ABC均完成”，因此目标 Local Work
的 A、B、C 记为通过（依据用户确认，未另行取得该界面的截图或运行日志）。

| 环节 | 状态 | 证据或缺口 |
| --- | --- | --- |
| 分支 | 已确认 | `experiment/work-rendering`，开始时工作树干净 |
| 执行环境 | 已确认 | 本次会话提供 Codex 本地工具；未独立验证目标 Local Work 界面 |
| RDKit 运行 | 成功 | 项目 `venv/Scripts/python.exe`，RDKit 2026.03.5 |
| PNG 生成 | 成功 | 苯、硝基苯各一张，600×400，真实 RDKit 结构图 |
| PNG 读取 | 成功 | Pillow 完整解码、非空检查；图片工具分别打开并目视核对结构 |
| A：聊天正文展示 | 当前 Codex 成功 | 用户确认两张本地 PNG 出现在聊天正文 |
| B：图文交错 | 当前 Codex 成功 | 用户确认三段文字与两张图片按指定顺序显示 |
| C：Chem_Agent 核心复用 | 当前 Codex 成功 | 现有标记经过解析、校验、渲染和安全编译，产出两张机理 PNG；用户确认最终聊天按指定顺序混排 |
| 目标 Local Work | 用户确认通过 | 用户在 Local Work 复测说明后报告“ABC均完成”；该界面日志未另行采集 |

两张结构图分别展示三组交替双键的苯环，以及带 N 正电荷、O 负电荷的硝基苯。
两图均不含电子流箭头，不能用作完整机理图的证据。

## 验证

在项目根目录运行：

```powershell
& .\venv\Scripts\python.exe -m pip check
& .\venv\Scripts\python.exe -m pytest tests/ -q
& .\venv\Scripts\python.exe .\Chem_Agent_Work_Test\render_test.py
& .\venv\Scripts\python.exe .\Chem_Agent_Work_Test\render_test.py --verify-only
```

- 依赖检查：无冲突。
- 项目基线：1030 passed，1 skipped，68.04 秒；有一条 pytest 缓存目录写入权限警告。
- 生成与只读核验：退出码均为 0；尺寸、PNG 解码和 SHA-256 匹配。
- 图片目视检查：苯环、硝基连接与形式电荷可见，无明显裁切。
- 不调用 LLM API、不使用图像生成模型或网络化学图片。
- 改动包括本独立目录中的脚本、实验文档与可复查输出、新增回归测试及测试指南入口；现有服务及渲染核心未改。

## 客户端观察栏

- 当前已确认界面：Codex。
- A：苯和硝基苯都直接显示在聊天正文，用户确认。
- B：“文字 → 苯 → 文字 → 硝基苯 → 文字”，用户确认。
- C：用户确认两张机理图按“文字 → 图片 → 文字 → 图片 → 文字”显示。

Codex 中的直接验证与 Local Work 中的用户确认分别记录；最终报告见 `REPORT.md`。

## C 实测细节

- 新增 `chem_render_cli.py`，使用 `python -m Chem_Agent_Work_Test.chem_render_cli` 执行。
- 输入 `03_attack.json`、`04_rearomatization.json` 来自现有 prompt 示例，无语法修改。
- 调用链：`parse_tags` → `validate_tag` → `render_tag` → `compile_tikz_to_png`。
- 输出 `03_attack.png`（1777×655）及 `04_rearomatization.png`（1744×655），附原始 TikZ 与哈希报告。
- 两图均经图片工具打开：第一步两根红色弯箭头，第二步一根恢复环内键的弯箭头，中文标签、形式电荷可见，无明显裁切。
- 第二步沿用项目的简化脱质子图：没有显式碱及其进攻 H 的箭头，只显示 C—H 键电子重建 π 键和形式离去的 H⁺。
- 首次 PNG 编译失败：MiKTeX 在受限环境下访问用户运行数据目录时报 Windows API error 5；同一入口获准正常权限运行后成功，未绕过源码闸门或改动编译器。
- 新增回归测试覆盖非法输入在编译前被拒绝，以及两个机理样例的现有契约校验和箭头数量；与示例审计合计 62 项通过。
- 全量回归：1038 passed，1 skipped，63.84 秒（关闭 pytest 缓存写入以避免环境权限警告）。
- 两张 PNG 及对应 TikZ 的实际文件哈希均与报告一致；A、B 原有图片只读核验仍通过。

## 本次聊天复测（2026-10-17）

- 界面：当前 Codex 聊天；没有独立确认目标 Local Work 界面。
- A、B：只读打开已有苯、硝基苯 PNG，并在聊天正文依次嵌入；用户回复“是的”，确认“文字 → 苯图 → 文字 → 硝基苯图 → 文字”。展示后的 `render_test.py --verify-only` 通过，原图未被替换。
- C：使用项目 `venv/Scripts/python.exe` 分别调用 `Chem_Agent_Work_Test.chem_render_cli` 处理两份既有 JSON。首次受限执行均退出 1，报 `PNG compilation failed; check existing LaTeX backend diagnostics`；该输出没有提供更细的失败原因。获准正常本地权限后，同一命令均退出 0；未修改或绕过解析、校验、渲染与安全编译流程。
- 生成并核验 `03_attack.png`（1777×655）和 `04_rearomatization.png`（1744×655）及各自 `.tikz`、`.json`。PNG 完整解码、尺寸、PNG 与 TikZ 的 SHA-256 均与报告一致；图片工具读取成功，中文标签和电子流箭头可见。
- 第二份输入产生 `not removing hydrogen atom without neighbors` 警告，仍成功生成 PNG；简化脱质子图没有显示碱及其夺取质子的箭头。
- 指定验证：`tests/test_work_rendering_cli.py` 与 `tests/test_examples_audit.py` 合计 62 passed；再次只读核验 A、B 图片通过。
- C 聊天展示：两张机理图在回答正文按“文字 → 亲电进攻图 → 文字 → 恢复芳香性图 → 文字”嵌入后，用户回复“是的，正常显示。”，确认实际可见性与顺序。本次当前聊天的 A、B、C 全部通过；此结论包含用户界面观察，不仅依据文件生成或图片工具读取。
- 本次更新限于实验输出及本记录，没有修改 MCP、服务或渲染器，没有调用图像生成模型。
