# 部署：把 Chem_Agent 发布成一个"任何用户都能用"的公开网页（BYOK）

> 适用版本：`core/web_api.py` + `web/index.html`（公开网页 / BYOK）落地之后。
> 目标形态：**用户打开网页 → 填自己的 API Key → 直接用**。服务端不提供、
> 不存储任何用户的密钥。

本文分三部分：**本地验证 → 服务器部署 → 验收清单**。全部命令都可复制执行。

---

## 0. 架构速览

```
浏览器（web/index.html，单文件）
   │  ① 用户在设置面板填自己的 Key / 端点 / 模型（存浏览器 sessionStorage）
   │  ② 每个请求把凭证放请求头 X-Chem-Api-Key / X-Chem-Base-Url / X-Chem-Model…
   ▼
FastAPI（api.py，同一进程同时服务两类调用方）
   ├── /v1/*    清小搭接入契约（Bearer = SERVICE_API_KEY，行为不变）
   ├── /api/chat      网页对话（SSE 流式 / JSON 非流式）
   ├── /api/web-config 前端公共配置（无敏感信息）
   ├── /api/session/<会话>/<uuid>.png  网页图示附件
   └── /files/<uuid>.png               清小搭图示附件（不变）
   ▼
core/credentials.py：把请求头里的凭证放进 contextvars，管线（app.process_question
及全部下游）在这一请求内**只用用户的凭证**；请求结束即失效。
```

要点：

* **两类调用方互不影响**。`/v1` 与网页走两套鉴权（服务端密钥 vs 用户自带密钥），
  共用同一条化学渲染/校验管线。
* **服务端不存用户密钥**：只在请求内存里存活，日志里只打指纹（`sk-abc…f3d2`）。
* 用户的 Key 若被前端写入浏览器 `sessionStorage`，关闭标签页即消失。

---

## 1. 本地验证（部署前必做）

```bash
# 1) 依赖（新增为零：只用已有的 fastapi/uvicorn）
pip install -r requirements.txt

# 2) 起服务（本地自测本机模型可加 WEB_ALLOW_PRIVATE_BASE_URL=1）
#    Windows PowerShell:
$env:WEB_ALLOW_PRIVATE_BASE_URL="1"; python -m uvicorn api:app --host 127.0.0.1 --port 8000
#    Linux/macOS:
WEB_ALLOW_PRIVATE_BASE_URL=1 python -m uvicorn api:app --host 127.0.0.1 --port 8000

# 3) 打开页面
#    http://127.0.0.1:8000/chat
#    在设置面板填你自己的 Key（端点默认 https://api.deepseek.com/v1），保存后提问：
#    "画出苯的结构式" / "用机理箭头说明 SN2 反应的电子流向"
```

单元测试（不联网、不花钱）：

```bash
python -m pytest tests/test_credentials.py tests/test_web_api.py -v
```

真实链路联调（不需要清小搭，也不需要浏览器）：见
`instructions/Test-Method.md` §20「公开网页（BYOK）验证」——里面有一个
mock LLM 端点 + 端到端脚本，可以确认"用户 Key 确实被用到了上游"。

---

## 2. 服务器部署（阿里云 / 任意 Linux）

现有清小搭服务已经跑在同一台机器上（`60.205.181.60`，systemd 单元
`chem_agent`，目录 `/var/www/chem_agent`）。**公开网页复用同一个服务进程**，
不需要新开端口、不需要第二个 systemd 单元——重启一次即可同时生效。

### 2.1 更新代码

```bash
ssh root@60.205.181.60
cd /var/www/chem_agent
git pull origin main                 # 本次改动：core/credentials.py、core/web_api.py、
                                     # web/index.html、api.py、core/llm_client.py 等
pip install -r requirements.txt      # 无新增依赖（fastapi/uvicorn 已有）
sudo systemctl restart chem_agent
sudo systemctl status chem_agent --no-pager     # 期望 active (running)
```

### 2.2 环境变量（`/var/www/chem_agent/.env`）

公开网页**不需要**新增任何服务器密钥。**不要**设置
`WEB_ALLOW_PRIVATE_BASE_URL`（那是本地自测开关，公网开启等于给 SSRF 开门）。
按需确认这几项：

```bash
SERVICE_API_KEY=...            # 清小搭用（保持不变）
PUBLIC_BASE_URL=https://你的域名  # 图示 PNG 的下载前缀；不填则按请求 Host 推导
                               # （Nginx 反代需 proxy_set_header Host $host;）
MODEL_NAME=deepseek-flash      # 单模型；旧名 v4-flash/v4-pro 已退役
THINKING_DEFAULT=on            # 思考开关（清小搭侧默认值；网页用户各自覆盖）
EFFORT_DEFAULT=low             # 思考强度 low/medium/high/max
MAX_TOKENS=32768               # 上限而非预留，按实际用量计费
# 视觉组留空即可（deepseek-flash 自带视觉）；仅当 MODEL_NAME 为纯文本模型时才填
```

**容器部署（只读 `/tmp` 的镜像）**：LaTeX 编译与 `/v1` 的请求级目录要有可写
落地处，指向一个挂载卷即可（不填 = 用系统临时目录）：

```bash
CHEM_AGENT_TMPDIR=/var/tmp/chem_agent
```

> 不设也不会崩（清理失败只记一行日志），但 **LaTeX 编译与图片落盘会失败**，
> 表现为回答里出现"（图示未能渲染）"。别指向 `tmpfs`——LaTeX 中间产物不小。

> 网页的图示 URL 形如 `https://你的域名/api/session/<会话>/<uuid>.png`，
> 由 `PUBLIC_BASE_URL` 或请求 Host 推导，二者与 `/files/*` 同一套逻辑。

### 2.3 Nginx 反向代理（若尚未配置）

关键点：**SSE 不能缓冲**，否则网页"打字机"效果变成一次性吐出；**转发真实 IP**，
否则限流把所有用户算作同一人（Nginx 的 IP）。

参考 `deploy/nginx-chem-agent.conf`（本仓库内，可直接改域名后使用）：

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;              # ★ SSE 必需
    proxy_cache off;
    proxy_read_timeout 300s;          # 机理题生成可达 1~3 分钟
    client_max_body_size 24m;         # 图片上传（前端每张限 8MB、最多 4 张）
}
```

改完 reload：`sudo nginx -t && sudo systemctl reload nginx`。

### 2.4 systemd 单元参考

`deploy/chem-agent.service`（本仓库内）。若沿用现有单元，确认两点即可：
`WorkingDirectory=/var/www/chem_agent`、`ExecStart=... -m uvicorn api:app --host 127.0.0.1 --port 8000`。

```bash
sudo cp deploy/chem-agent.service /etc/systemd/system/chem_agent.service   # 首次
sudo systemctl daemon-reload && sudo systemctl restart chem_agent
```

### 2.5 可选：把网页放到别的域名/静态托管

`web/index.html` 是**自包含单文件**（无构建、无 CDN 依赖），可以直接丢到任意
静态托管（OSS + CDN、GitHub Pages…），只要前端 `fetch` 的地址指向本服务并
允许跨域即可：

* 页面里两处端点写死为同源（`api/chat`、`api/web-config`）——放到异地托管时
  改成绝对地址，例如把 `fetch("api/chat"` 改为
  `fetch("https://你的域名/api/chat"`；
* 服务端已开 `CORSMiddleware(allow_origins=["*"])`，但**必须**允许自定义头，
  否则浏览器预检失败——当前配置 `allow_headers=["*"]` 已满足；
* 附件 URL 由服务端按 `PUBLIC_BASE_URL` 生成，与页面托管位置无关。

> 安全提示：把页面放到第三方域名时，用户的 Key 会经由那个域名的 JS 发出——
> 只托管你自己控制的页面。

---

## 3. 验收清单（部署后逐项确认）

```bash
KEY=<你的 SERVICE_API_KEY>          # 清小搭密钥；网页路径不需要它
BASE=https://你的域名

# ① 页面可打开，且是 HTML（不是 JSON）
curl -sI $BASE/chat | head -3

# ② 清小搭契约未受影响：无凭证 401，正确凭证 200
curl -s -o /dev/null -w '%{http_code}\n' $BASE/v1/models
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" $BASE/v1/models

# ③ 网页端点缺密钥 → 400 且提示怎么填
curl -s -X POST $BASE/api/chat -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"你好"}]}'

# ④ 内网端点必须被拒（安全）
curl -s -X POST $BASE/api/chat -H 'Content-Type: application/json' \
     -H 'X-Chem-Api-Key: sk-test' -H 'X-Chem-Base-Url: http://169.254.169.254/v1' \
     -d '{"messages":[{"role":"user","content":"你好"}]}'

# ⑤ 真实提问（用自己的 Key，会消耗你自己的 token）
curl -s -X POST $BASE/api/chat -H 'Content-Type: application/json' \
     -H 'X-Chem-Api-Key: 你的Key' -H 'X-Chem-Model: deepseek-flash' \
     -d '{"messages":[{"role":"user","content":"画出苯的结构式"}]}' | head -c 400
```

浏览器侧逐项检查：

- [ ] `/chat` 首次打开弹出设置面板；填入 Key 后"测试连接"返回正常。
- [ ] 提问"画出苯的结构式"→ 文字先出，化学图示 PNG 随后显示在正文中。
- [ ] 关闭标签页再打开：Key 已消失（sessionStorage），需重新填写。
- [ ] 上传一张结构式图片 → 若未填"视觉模型"，回答中会明确提示无法识别（不是静默失败）。
- [ ] 服务器日志里**看不到**任何完整密钥（只有 `sk-abc…f3d2` 形式的指纹）：
      `sudo journalctl -u chem_agent -n 50 --no-pager | grep '\[web\]'`
- [ ] 清小搭侧照旧可用（`/v1` 契约未变）。

---

## 4. 运维速查

| 事项 | 命令 / 说明 |
| :--- | :--- |
| 服务日志 | `sudo journalctl -u chem_agent -f` |
| 只看网页请求 | `sudo journalctl -u chem_agent --no-pager \| grep '\[web\]'` |
| 会话附件目录 | `data/web_sessions/<会话id>/`（**按全局配额回收**：2GB / 50000 个文件，**只在超配额时**删最旧的图，不按时间清；见 `core/web_api.py::prune_web_attachments`） |
| 清小搭附件目录 | `data/attachments/`（独立配额 2GB / 50000，长期保留，与网页互不挤占） |
| 限流 | 每 IP 每分钟 20 次（`core/web_api.py` 的 `RATE_LIMIT_PER_MINUTE`） |
| 用户密钥去向 | 只在请求内存；日志只打指纹；`data/diagnostics.jsonl` 里存的是标记文本与凭证指纹，**不含密钥** |

### 常见问题

**Q：网页报"请求过于频繁"？**
同 IP 每分钟超过 20 次。这是防脚本的默认值，可在 `core/web_api.py` 调整
`RATE_LIMIT_PER_MINUTE`（0 = 关闭）。Nginx 未转发 `X-Forwarded-For` 时
所有用户会共享同一配额，先修反代配置。

**Q：图示一直显示"（图示未能渲染）"？**
服务器缺 LaTeX 工具链。装：
`apt install texlive-xetex texlive-latex-extra texlive-lang-chinese poppler-utils`
（文字回答不受影响，只是不出图）。

**Q：用户填了自定义端点被拒（"端点地址不可用"）？**
服务端默认拒绝内网/本机/云元数据地址（SSRF 防护）。用户的公网 OpenAI 兼容
端点（DeepSeek / 智谱 / SiliconFlow / OpenRouter…）不受影响。

**Q：想让所有访问者白嫖服务器自己的 Key？**
那是另一种形态（共享密钥），本项目按需求做成了**纯 BYOK**：服务端不提供共享
密钥。若将来要放开，需要在 `core/web_api.py::_require_credentials` 里显式回退
到环境配置，并自行加上配额与鉴权——**不要**默默放开。

