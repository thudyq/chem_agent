# Work × Chem_Agent 最小展示实验

本目录是独立实验材料；不修改服务、MCP、标记语法或渲染核心。
A、B 的固定输入为苯与硝基苯结构图；C 使用现有 prompt 的两步机理图样例。

## 待办与判定

- [x] 本地生成并核对 PNG。
- [x] A：用户确认当前 Codex 聊天正文中直接展示两张已有 PNG。
- [x] B：用户确认当前 Codex 中观察到“文字 → 苯 → 文字 → 硝基苯 → 文字”。
- [x] C：复用现有严格解析、校验及渲染核心生成两步机理 PNG，用户确认当前聊天图文交错显示。
- [x] 在目标 Local Work 中复测 A、B、C（用户报告“ABC均完成”；未另行采集目标界面日志）。

生成、图片工具读取、最终聊天显示是不同的验证环节。Codex 中的结果不能自动
算作 Local Work 通过；回答中的 Markdown 顺序也不能替代实际界面观察。
没有观察到图片时，记录“未确认”，不要直接归因为渲染失败。

## 复现（在项目根目录执行）

```powershell
& .\venv\Scripts\python.exe .\Chem_Agent_Work_Test\render_test.py
& .\venv\Scripts\python.exe .\Chem_Agent_Work_Test\render_test.py --verify-only
```

第一条命令会重新绘图；A、B 阶段只运行第二条只读命令。
`output/manifest.json` 记录生成器、版本、SMILES、尺寸和文件 SHA-256。
复测时在 Local Work 打开本目录，使用已安装 RDKit 的项目 Python；不需要安装
整个 Chem_Agent 或配置任何模型 API。

## A 提示词

请读取当前目录 output/01_benzene.png 与 output/02_nitrobenzene.png，
在最终聊天正文中直接展示，分别标注“苯”和“硝基苯”。不要重新绘图、不要
调用图像生成模型、不要使用网络图片。如果只能提供链接，请如实说明。

## B 提示词

请只使用已有的 output/01_benzene.png 与 output/02_nitrobenzene.png，
按下列顺序在最终聊天回答中讲解苯的硝化：

1. 第一段介绍苯的芳香性和亲电芳香取代反应。
2. 直接展示苯的 PNG。
3. 第二段解释硝鎓离子的生成、苯环进攻形成 σ 络合物、去质子化恢复芳香性。
4. 直接展示硝基苯的 PNG。
5. 第三段总结产物和机理要点。

禁止重新绘图。两图仅展示反应物和产物，不能冒充电子流箭头机理图。
展示后再次运行 --verify-only，确认文件未被替换。

## 结果记录

见 `RESULTS.md`。请记录目标聊天所处界面、图片是否可见、实际顺序以及文件
校验结果。当前 Codex 的 A、B、C 已验证，Local Work 的 A、B、C 已由用户确认完成。
实验结论、验证范围与限制见 `REPORT.md`。

## C：现有渲染核心的命令行入口

在项目根目录运行：

```powershell
& .\venv\Scripts\python.exe -m Chem_Agent_Work_Test.chem_render_cli --input Chem_Agent_Work_Test/03_attack.json --output Chem_Agent_Work_Test/output/03_attack.png
& .\venv\Scripts\python.exe -m Chem_Agent_Work_Test.chem_render_cli --input Chem_Agent_Work_Test/04_rearomatization.json --output Chem_Agent_Work_Test/output/04_rearomatization.png
& .\venv\Scripts\python.exe -m pytest tests/test_work_rendering_cli.py tests/test_examples_audit.py -q -p no:cacheprovider
```

JSON 只接受 `tag` 字段，内容为一个完整的现有渲染标记。两个样例来自
`prompts/system_prompt.txt` 示例 11 的亲电进攻和脱质子步骤，箭头端点原样保留。
封装按顺序调用现有解析器、校验器、渲染注册表和安全 PNG 编译器；校验失败
即停止，不接受直接传入 LaTeX。编译失败时返回非零退出码。

每张 PNG 旁保存渲染器生成的 `.tikz` 和包含标记、调用链、尺寸、哈希的 `.json`，
便于复查。输出会覆盖同名实验产物，不会自动启动服务或调用模型。
需要正常可用的本地 LaTeX 工具链；本机受限执行环境阻止 MiKTeX 访问其用户
运行数据，使用获准的正常本地执行权限后成功，编译安全流程保持原样。

第二张图是简化脱质子示意：保留 C—H 到环内键的电子流，以 H⁺ 表示离去质子，
没有展示夺取质子的碱及对应箭头，不应称作完整的酸碱步骤。

官方参考：[ChatGPT Work local security](https://learn.chatgpt.com/docs/enterprise/chatgpt-work-local-security)。
本地执行能力取决于客户端、账户和权限；文档没有保证本地 PNG 的任意图文排序。
