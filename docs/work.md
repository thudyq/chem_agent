# Work 本地渲染

在项目根目录安装最小依赖，并准备 XeLaTeX（含 tikz、amsmath、ctex 宏包）：

```powershell
python -m pip install -r requirements/render.txt
python -m cli.render --input examples/work/attack.json --output data/work/attack.png
```

输入 JSON 只接受一个字符串字段 `tag`，内容为一个完整的既有渲染标记：

```json
{"tag": "[STRUCT:c1ccccc1,label=苯]"}
```

命令先解析和校验标记，再调用现有渲染器与安全编译器。非法输入或编译失败
返回非零退出码；成功时只保存 PNG，并打印其绝对路径。添加 `--debug` 可
额外保存同名 TikZ 源码。不启动服务、不调用模型 API，也不需要 `.env`。
编译权限与文件隔离要求见 [部署指南 §2.6](deploy.md)。

本地代理将输出 PNG 的绝对路径作为 Markdown 图片插入回答。当前命令一次
处理一个标记，多图的位置由代理安排；实际可见性取决于目标聊天界面。

`examples/work/attack.json` 与 `rearomatization.json` 展示苯硝化的两个步骤。
第二步为简化脱质子示意，没有显式绘制碱及其夺取质子的箭头。

![亲电进攻样例](../examples/work/attack.png)
