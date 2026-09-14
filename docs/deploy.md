# Chem_Agent 自托管部署指南

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

# 2) 起服务（本地自测**默认就这样**，不需要任何额外变量）
python -m uvicorn api:app --host 127.0.0.1 --port 8000

#    ─────────────── ⚠️ 下面这个开关只给"本地连自己电脑上的模型"用 ───────────────
#    WEB_ALLOW_PRIVATE_BASE_URL=1 会让服务端的"内网/本机端点"拦截**整体失效**
#    。它同时关掉两道防护：
#      · R2 —— 视觉端点的 SSRF 校验
#      · R3 —— 出站请求"不跟随跳转"
#    也就是说：**线上开着它 = 把服务器变成任意内网地址的请求跳板**，
#    而且不会有任何日志提醒你。所以它**绝不能写进线上的 `.env` 或 systemd 单元**。
#
#    只在需要连本机模型时**临时**带一次（用完即弃，不要落盘）：
#      Windows PowerShell: $env:WEB_ALLOW_PRIVATE_BASE_URL="1"; python -m uvicorn api:app --host 127.0.0.1 --port 8000
#      Linux/macOS:        WEB_ALLOW_PRIVATE_BASE_URL=1 python -m uvicorn api:app --host 127.0.0.1 --port 8000
#    ─────────────────────────────────────────────────────────────────────────────

# 3) 打开页面
#    http://127.0.0.1:8000/chat
#    在设置面板填你自己的 Key（端点默认 https://api.deepseek.com/v1），保存后提问：
#    "画出苯的结构式" / "用机理箭头说明 SN2 反应的电子流向"
```

**线上随时可以核实这个开关没被打开**（`false` 才是对的）：

```bash
curl -s https://你的域名/api/web-config | grep -o '"allow_private_base_url":[a-z]*'
```

> 为什么用这条而不是翻文件：`.env` 是 `python-dotenv` 在**运行时**注入
> `os.environ` 的，`/proc/<pid>/environ` **看不到**它。上面这条走的是应用自己的
> 判断结果，是运行时事实。要查文件就查两处：项目 `.env` 与 systemd 单元
> （含 drop-in）：`systemctl cat chem_agent | grep -i allow_private`。

单元测试（不联网、不花钱）：

```bash
python -m pytest tests/test_credentials.py tests/test_web_api.py -v
```

真实链路联调（不需要清小搭，也不需要浏览器）：见
`instructions/Test-Method.md` §20「公开网页（BYOK）验证」——里面有一个
mock LLM 端点 + 端到端脚本，可以确认"用户 Key 确实被用到了上游"。

---

## 2. 服务器部署（任意 Linux + systemd + Nginx 反代）

以下假设一台 Linux 服务器：systemd 单元 `chem_agent`，项目目录
`/var/www/chem_agent`（均为示例值，按实际调整）。**公开网页复用同一个服务
进程**，不需要新开端口、不需要第二个 systemd 单元——重启一次即可同时生效。

### 2.1 更新代码

```bash
ssh <你的服务器>
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
否则限流把所有用户算作同一人（Nginx 的 IP）。★ 转发的写法必须是**覆盖**
（`$remote_addr`），**不能**用 `$proxy_add_x_forwarded_for` —— 后者是"追加"，
客户端自己伪造的 `X-Forwarded-For` 会被原样留在最前面，攻击者每个请求换一个假
IP 就得到一个新的限流桶，"每 IP 20 次/分钟"形同虚设。

参考 `deploy/nginx-chem-agent.conf`（本仓库内，可直接改域名后使用）：

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $remote_addr;   # ★ 覆盖，不是 $proxy_add_x_forwarded_for
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;              # ★ SSE 必需
    proxy_cache off;
    proxy_read_timeout 300s;          # 机理题生成可达 1~3 分钟
    client_max_body_size 24m;         # 图片上传（前端每张限 8MB、最多 4 张）
}
```

改完 reload：`sudo nginx -t && sudo systemctl reload nginx`。

> 服务端**自己不再解析**这个头：它只读 uvicorn 净化后的
> `request.client`。uvicorn 默认只信任来自 `127.0.0.1` 的代理，并按"从右往左
> 找第一个不受信地址"取真实客户端。**若你的反代不是从 `127.0.0.1` 连进来**
> （例如跑在同一台机器的 Docker 网络里、或另有一层 LB），必须给 uvicorn 加
> `--forwarded-allow-ips=<反代 IP 或网段>`，否则所有用户会共用一个限流桶
> （限流变严，属于安全一侧的失败）。

### 2.4 systemd 单元参考

`deploy/chem-agent.service`（本仓库内）。若沿用现有单元，确认三点即可：
`WorkingDirectory=/var/www/chem_agent`、
`ExecStart=... -m uvicorn api:app --host 127.0.0.1 --port 8000`、
**`--forwarded-allow-ips=127.0.0.1`**（限流取的是 uvicorn 净化后的
客户端 IP，uvicorn 只在直连对端是受信代理时才采信 `X-Forwarded-For`；反代不在
本机时改成它的 IP/网段）。

> 另注：限流桶是**进程内**内存，`--workers` 大于 1 会让有效额度成倍放大。
> 单 worker 足够（生成是 IO 密集），参考单元里就是 `--workers 1`。

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
* 服务端已开 `CORSMiddleware`，但**默认只放行本机调试地址与你自己的 `PUBLIC_BASE_URL`**
  （网页与接口同源时根本不需要 CORS）。放到异地托管时，必须把那个域名
  加进去，否则浏览器预检失败：`CORS_ALLOW_ORIGINS=https://你的静态域名`；
* 必须允许自定义头，否则浏览器预检失败——当前配置 `allow_headers=["*"]` 已满足；
* 附件 URL 由服务端按 `PUBLIC_BASE_URL` 生成，与页面托管位置无关。

> 安全提示：把页面放到第三方域名时，用户的 Key 会经由那个域名的 JS 发出——
> 只托管你自己控制的页面。

### 2.6 ★ 跑 LaTeX 的隔离要求

图示是靠**编译 LaTeX** 画出来的，而 `.tex` 内容来自大模型 —— 也就是**可以被用户
提问影响**。TeX 的 `\input`/`\openin`/`\read` 能读服务器上任意可读文件，并把内容
排版进图片**回传给访客**（实测见 `instructions/Security-Review.md` R1）。

代码里已经加了两层（源码闸门 + 引擎级加固），但**引擎级限制只有 TeX Live/kpathsea
认**（`--cnf-line=openin_any=p`），MiKTeX 不认。所以最后一道必须由部署形态保证：

```bash
# ① 先确认这台机器的引擎到底拦不拦得住（别假定）
python -m utils.latex_compile --security-check
#    file_read_restricted: true  → 引擎已拦住越界读文件
#    file_read_restricted: false → 拦不住，必须靠下面的隔离

# ② 无论结果如何，都让编译进程读不到密钥文件
chmod 600 /var/www/chem_agent/.env        # 只有服务属主可读
```

* **最省**：给编译单独开一个系统用户，`.env` 归另一个用户所有。
* **更稳**：把编译放进容器 / systemd 沙箱，只挂一个空的可写工作目录，例如
  `ProtectHome=yes`、`ReadOnlyPaths=/var/www/chem_agent`、
  `InaccessiblePaths=/var/www/chem_agent/.env`。
* 用了 `ProtectSystem`/`PrivateTmp` 之类的沙箱时，记得给 LaTeX 留一个可写的临时
  目录（配合 `CHEM_AGENT_TMPDIR`，见 §2.2），否则图示会全部渲染失败。

> 服务启动时会打一行 `[startup] LaTeX 加固：...`，把当前生效的加固选项写进日志；
> 引擎不支持文件访问限制时还会额外警告一次。

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

# ⑥ LaTeX 编译的文件访问限制——★ 必须跑，别假定
python -m utils.latex_compile --security-check
#    file_read_restricted: true  → 引擎已拦住越界读文件
#    file_read_restricted: false → 这台机器拦不住（如 MiKTeX），必须做 §2.6 的隔离
python -m utils.latex_compile --security-check --strict   # 没拦住则以退出码 1 结束
```

浏览器侧逐项检查：

- [ ] `/chat` 首次打开弹出设置面板；填入 Key 后"测试连接"返回正常。
- [ ] 提问"画出苯的结构式"→ 文字先出，化学图示 PNG 随后显示在正文中。
- [ ] 关闭标签页再打开：Key 已消失（sessionStorage），需重新填写。
- [ ] 上传一张结构式图片 → 若未填"视觉模型"，回答中会明确提示无法识别（不是静默失败）。
- [ ] 服务器日志里**看不到**任何完整密钥（只有 `sk-abc…f3d2` 形式的指纹）：
      `sudo journalctl -u chem_agent -n 50 --no-pager | grep '\[web\]'`
- [ ] 清小搭侧照旧可用（`/v1` 契约未变）。
- [ ] 设置面板里能看到「📝 隐私说明」（写明会记录提问原文、不含 Key），
      输入框下方也有常驻提示。
- [ ] 诊断日志权限是 0600：`ls -l data/diagnostics.jsonl`（应为 `-rw-------`）。

---

## 4. 运维速查

| 事项 | 命令 / 说明 |
| :--- | :--- |
| 服务日志 | `sudo journalctl -u chem_agent -f` |
| 只看网页请求 | `sudo journalctl -u chem_agent --no-pager \| grep '\[web\]'` |
| 会话附件目录 | `data/web_sessions/<会话id>/`（**按全局配额回收**：2GB / 50000 个文件，**只在超配额时**删最旧的图，不按时间清；见 `core/web_api.py::prune_web_attachments`） |
| 清小搭附件目录 | `data/attachments/`（独立配额 2GB / 50000，长期保留，与网页互不挤占） |
| 限流 | 每 IP 每分钟 20 次（`core/web_api.py` 的 `RATE_LIMIT_PER_MINUTE`）；客户端身份取 uvicorn 净化后的对端地址 |
| 在途上限 | 全局 16、单 IP 4。超了返回 503 + "服务器当前请求较多"（不是用户设置问题）。可用环境变量 `CHEM_AGENT_MAX_INFLIGHT` / `CHEM_AGENT_MAX_INFLIGHT_PER_IP` 调整，设 `0` = 关闭该道闸门 |
| 图片大小上限 | 单张 8MB、单次合计 16MB（**解码后**）；超限 400 并提示压缩/减少张数。反代另需 `client_max_body_size` 放行（见 §2.3） |
| 单个会话附件上限 | 64MB（`core/web_api.py` 的 `WEB_SESSION_MAX_BYTES`）；超了先回收**该会话内部**最旧的图，再走全局配额 |
| 浏览器来源（CORS） | 默认只放行本机调试地址 + `.env` 的 `PUBLIC_BASE_URL`（网页与接口同源，同源请求不需要 CORS）。把页面托管到**别的域名**时（§2.5），用 `CORS_ALLOW_ORIGINS=https://a.com,https://b.com` 显式列出 |
| 安全响应头 | 所有响应带 `X-Frame-Options: DENY`、CSP `frame-ancestors 'none'`、`nosniff`、`Referrer-Policy: no-referrer`（防 iframe 套框/点击劫持） |
| 用户密钥去向 | 只在请求内存；journald 日志只打指纹（`sk-abc…f3d2`）；**不含密钥** |
| 诊断日志（★ 含真人提问） | `data/diagnostics.jsonl`（权限 **0600**）：清小搭与网页**共用**这一个文件，靠 `cid` 前缀区分（`chatcmpl-*` / `web-*`）；每行含**提问原文**与模型原始输出，只有凭证是指纹。超 8MB 轮转为 `.1.jsonl`（磁盘最多约 16MB）。Streamlit 用**独立**的 `data/streamlit_diagnostics.jsonl`。要清空：`sudo truncate -s 0 data/diagnostics.jsonl` |

### 常见问题

**Q：网页报"请求过于频繁"？**
同 IP 每分钟超过 20 次。这是防脚本的默认值，可在 `core/web_api.py` 调整
`RATE_LIMIT_PER_MINUTE`（0 = 关闭）。Nginx 未转发 `X-Forwarded-For` 时
所有用户会共享同一配额，先修反代配置。

**Q：用户看到"服务器当前请求较多，请稍后重试"？**
这是**在途上限**而不是用户填错了东西：服务器同时在处理的出站调用
达到上限（默认全局 16、单 IP 4）就会快速失败，避免把内存/线程耗尽。默认值够用；
如果是正常高峰想放宽，设环境变量 `CHEM_AGENT_MAX_INFLIGHT=32`（以及
`CHEM_AGENT_MAX_INFLIGHT_PER_IP`）后重启服务。**如果日志里这个提示很频繁、
而且来源 IP 很分散**，那多半是有人在打你，考虑上 WAF/CDN 限速。

**Q：用户说图片传不上去 / 提示图片过大？**
三道限制，按顺序看：① 单张 8MB、单次合计 16MB（解码后）——让用户
压缩或减少张数；② 反代的 `client_max_body_size`（§2.3 应设 48m，nginx 默认只有 1MB，
小图也会被 413 拒掉，且用户看不到原因）；③ 前端也会先拦一次并给提示。
如果报的是"图片格式不支持"，那是魔数校验：附件必须是**真图片**（png/jpg/gif/bmp/webp）。

**Q：图示一直显示"（图示未能渲染）"？**
服务器缺 LaTeX 工具链。装：
`apt install texlive-xetex texlive-latex-extra texlive-lang-chinese poppler-utils`
（文字回答不受影响，只是不出图）。

**Q：用户填了自定义端点被拒（"端点地址不可用"）？**
服务端默认拒绝内网/本机/云元数据地址（SSRF 防护）。用户的公网 OpenAI 兼容
端点（DeepSeek / 智谱 / SiliconFlow / OpenRouter…）不受影响。

**Q：用户看到"接口地址返回了重定向 / 本服务不跟随跳转"？**
这是**有意的安全策略**：出站请求一律不跟随 302/301。否则一个
公网域名可以先通过 SSRF 校验、再 302 跳到内网或云元数据地址，校验就白做了。
让用户把**最终的完整地址**直接填进设置里的「接口地址」即可（例如填
`https://…` 而不是会被 301 跳转的 `http://…`）。

**Q：想让所有访问者白嫖服务器自己的 Key？**
那是另一种形态（共享密钥），本项目按需求做成了**纯 BYOK**：服务端不提供共享
密钥。若将来要放开，需要在 `core/web_api.py::_require_credentials` 里显式回退
到环境配置，并自行加上配额与鉴权——**不要**默默放开。

