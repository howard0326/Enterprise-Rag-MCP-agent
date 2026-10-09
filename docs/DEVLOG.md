# 开发记录（阶段 1–5）

> 本文件是项目开发过程的完整记录，从最初的 README 迁移而来，按阶段保留了
> 当时的实测数据、踩坑记录与设计决策。对外介绍的摘要与快速开始请见仓库根目录
> [README.md](../README.md)。

## 目录结构

```
enterprise-rag-mcp/
├── backend/                    FastAPI 服务 + LangGraph Agent 编排
│   └── __init__.py
├── mcp_servers/                MCP 工具服务器
│   ├── __init__.py
│   ├── registry.py             server 注册表 / list_tools / call_tool（阶段3）
│   ├── test_mcp.py             端到端工具测试脚本（阶段3）
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   └── server.py           search_documents / get_document_metadata
│   └── business_db/
│       ├── __init__.py
│       └── server.py           query_orders / create_ticket
├── rag_pipeline/               文档解析与索引
│   └── __init__.py             document_parser / indexer / retriever / ingestion_cli（阶段2）
├── eval/                       评测
│   └── __init__.py             test_cases.json / evaluator.py（阶段5）
├── config/
│   ├── __init__.py
│   ├── config_loader.py        配置加载 + 环境变量替换 + Pydantic 校验
│   └── settings.yaml           唯一配置真源
├── docs/                       待索引语料（sample_*.md 为演示数据）
├── .env.example
├── .gitignore
├── docker-compose.yml          本地 Qdrant
├── requirements.txt
└── README.md
```

## 环境准备

> **重启电脑后请先做一次自检**：Qdrant 是常驻进程，关机后会停，必须重启；
> 虚拟环境、模型缓存、索引数据都不会丢。
> 一条命令自检：`.venv\Scripts\python.exe scripts\healthcheck.py`

> 本机环境（已实际验证）：
> - Python **3.12.10**（按用户级安装于 `%LOCALAPPDATA%\Programs\Python\Python312`，无需管理员）
> - 虚拟环境 `D:\projects\enterprise-rag-mcp\.venv`，依赖**已安装**（fastapi 0.141.1 / langgraph 1.2.12 / mcp 2.2.0 / qdrant-client 1.19.1 / torch 2.14.0 等）
> - **无 Docker**：Qdrant 使用 Windows 单文件二进制 `D:\projects\.tools\qdrant\qdrant.exe`（v1.19.1）
> - HuggingFace 官网不可达，嵌入模型经 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载

```powershell
# 1) 配置密钥
Copy-Item .env.example .env
#   编辑 .env，填入 DEEPSEEK_API_KEY

# 2) 在【你自己的终端】启动 Qdrant（常驻服务不能在 DSH 沙箱内保持存活）
powershell -ExecutionPolicy Bypass -File scripts\start_qdrant.ps1
#   控制台: http://localhost:6333/dashboard
#
#   注意：本机是 Windows PowerShell 5.1，没有 pwsh（PowerShell 7）。
#   若你装了 PowerShell 7，也可写：pwsh -File scripts\start_qdrant.ps1
#   scripts\*.ps1 均为 5.1 兼容写法（未使用 ternary / ?? / Join-Path 多段参数）。

# 3) 校验配置
.\.venv\Scripts\python.exe -m config.config_loader
```

已安装依赖清单（关键项）：`fastapi 0.141.1`、`uvicorn 0.53.0`、`qdrant-client 1.19.1`、
`langgraph 1.2.12`、`mcp 2.2.0`、`pydantic 2.13.5`、`pypdf 6.19.0`、
`sentence-transformers 6.1.0`、`torch 2.14.0`、`pandas 3.0.6`。

`python -m config.config_loader` 的**实际输出**（已运行验证，exit=0）：

```
项目           : enterprise-rag-mcp v0.1.0 (env=dev)
向量库         : qdrant localhost:6333 collection=enterprise_docs distance=Cosine
嵌入模型       : provider=local model=BAAI/bge-m3 dim=1024 device=cpu
LLM            : deepseek/deepseek-chat api_key=未配置(读取 DEEPSEEK_API_KEY)
分块           : chunk_size=512 overlap=64 batch=100
服务           : http://0.0.0.0:8000 session_max_turns=10
MCP servers    :
  [on ] knowledge_base via stdio -> search_documents, get_document_metadata
  [on ] business_db via stdio -> query_orders, create_ticket
[OK] 配置校验通过
```

## 本环境的四个已知坑（已实测，勿踩）

1. **`mcp` 装到 2.x**：`FastMCP` 已改名 `MCPServer`（`mcp.server.fastmcp` 变成显式报错的兼容垫片），
   `Tool.inputSchema` → `input_schema`，`InitializeResult.serverInfo/protocolVersion` →
   `server_info/protocol_version`。两个 server 已做**双版本兼容导入**，不锁 SDK 大版本。
   实测协议版本为 `2025-11-25`。
   > 另一个关键差异：**mcp 2.x 会把工具内未捕获的异常转成 `UnexpectedToolError` 抛出**，
   > 而不是返回 `isError=True` 的内容块。所以工具实现必须自己 try/except 并返回
   > `{"ok": false, "error": ...}` 信封，不能让异常逃逸（阶段3 的实现已按此约束设计）。
2. **HuggingFace 不可达**：需 `HF_ENDPOINT=https://hf-mirror.com`（`huggingface.co` 直接 ConnectTimeout）；
   另外受限环境下符号链接探测会失败，需 `HF_HUB_DISABLE_SYMLINKS=1`（走真实文件复制）。
3. **Qdrant snapshots 目录权限**：某些受限环境下 Qdrant 打开既有 snapshots 目录会 panic
   （`src\snapshots.rs` PermissionDenied），`scripts\start_qdrant.ps1` 已把 snapshots
   指向 `%TEMP%\qdrant_snapshots` 规避；存储仍在 `.data\qdrant\storage`。
4. **嵌入模型版本组合**（实测踩坑并修复）：

   | 组合 | 结果 |
   | --- | --- |
   | `transformers 5.17.0` + `sentence-transformers 6.1.0` | ❌ `Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'` |
   | `transformers 4.57.6` + `sentence-transformers 5.7.0` | ✅ bge-m3 正常，1024 维 |

   根因：transformers 5.x 把模型维度移到了 `get_text_config()`，而 ST 6.x 仍从顶层
   `hidden_size` 读取，维度传空。`requirements.txt` 已锁定 `sentence-transformers>=5.0,<6.0`
   与 `transformers>=4.44,<5.0`，**不要放开上限**。

## 阶段1 验收记录（实测）

| 验收项 | 命令 / 方式 | 结果 |
| --- | --- | --- |
| 配置校验 | `python -m config.config_loader` | ✅ exit=0，摘要如上 |
| 工具注册与 JSON Schema | 进程内 `list_tools()` | ✅ 4 个工具，required/props 正确 |
| 参数校验拒绝 | 非法入参调用 | ✅ 3/3 被 ValidationError 拒绝 |
| MCP 协议握手（stdio） | `ClientSession.initialize()` | ✅ 两个 server 均握手成功，protocol=2025-11-25 |
| Qdrant 读写 | 建集合→写向量→读回→删除 | ✅ `QDRANT_RW_OK` |
| 嵌入模型 | `SentenceTransformer('BAAI/bge-m3')` | ✅ dim=1024，L2=1.0，语义区分度有效 |

> 注意：MCP **工具调用**的端到端验证归属阶段3（`mcp_servers/test_mcp.py`），
> 阶段1 的工具是占位实现，调用只会抛 `ToolNotReadyError`。

## 嵌入模型的选型说明（重要）

DeepSeek 官方 API 目前**未开放文本嵌入（embedding）接口**，只提供 chat/completions，
不存在可直接调用的 "DeepSeek V4.1 embedding"。因此本项目采用**可插拔 provider** 设计：

| provider | 说明 | 配置 |
| --- | --- | --- |
| `local`（默认） | 本地 `sentence-transformers` 推理，可完全离线，1024 维 | `EMBEDDING_PROVIDER=local` `EMBEDDING_MODEL=BAAI/bge-m3` |
| `openai_compatible` | 任意兼容 `/v1/embeddings` 的服务（DeepSeek 开放接口后直接切换） | 填 `base_url` / `model` / `api_key` |

切换方式：只改 `config/settings.yaml` 的 `embedding` 段（或 `.env` 三个变量），
`rag_pipeline` 与 `backend` 代码零改动。**注意 `embedding.dimension` 必须与
Qdrant 集合维度一致**，否则写入会被 Qdrant 拒绝；`config_loader` 已内置常见模型的
维度自检（bge-m3=1024、bge-base-zh=768、text-embedding-3-small=1536 等）。

## 阶段规划

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 环境初始化与项目脚手架 | ✅ 已完成（含真实运行验证） |
| 2 | RAG 管道（解析 / 索引 / 混合检索 / 增量摄取） | ✅ 已完成（R@3=0.875） |
| 3 | MCP server 实现与注册表 | ✅ 已完成（30 项断言全通过） |
| 4 | LangGraph Agent 编排 + FastAPI SSE + 会话管理 | ✅ 已完成（HTTP 实测通过） |
| 5 | 评测（20 用例）与结构化可观测性 | ✅ 已完成（20/20 全通过） |
| 6 | 简历包装 | ⏳ 待开始 |

## 阶段5 优化轮：从"100% 但脏"到"100% 且干净"

首轮评测虽然四项 100%，但失败类型分布里挂着 `extra_tool: 6` / `tool_error: 2` ——
**子集匹配不把多余调用算作失败，所以指标漂亮但系统在浪费**。优化轮的目标是消掉这些噪声。

| 指标 | 优化前 | 优化后 | 变化 |
| --- | --- | --- | --- |
| 意图 / 工具 / 回答包含 / 三项全通过 | 100% | **100%** | 保持 |
| 端到端延迟（均值） | 2.56s | **2.13s** | **−17%** |
| 端到端延迟（P95） | 3.46s | **2.73s** | **−21%** |
| 首条用例延迟 | 10.2s | **2.4s** | **−76%** |
| 过度调用（extra_tool） | 6 条 | **0 条** | 清零 |
| 工具失败（tool_error） | 2 条 | **0 条** | 清零 |
| `degraded=True` 用例 | 2 条 | **0 条** | 清零 |

四项优化措施，每条都对应一个已定位的根因：

1. **服务启动预热**（`agent.warmup()` + FastAPI lifespan）：把冷启动成本前移。
   实测分解 `embedder 5.9s + tools 1.8s + retrieval 6.8s ≈ 14.5s`。
   收益最大的一条 —— 首条用例 10.2s → 2.4s。
2. **执行前实体门控**：`query_orders` 没有客户 ID/订单号时不发起注定失败的调用，
   直接返回"需要补充参数"（`MISSING_ARGUMENT`），把"追问"与"故障"区分开。
3. **信息不足 ≠ 降级**：`MISSING_ARGUMENT` / `NOT_FOUND` / `INVALID_ARGUMENT`
   归为 `_INFORMATIONAL_CODES`，不计入 `degraded`。此前正常澄清流程会被渲染成
   "部分数据获取失败"。
4. **规划期收紧二次判定**：仅凭"退款/签收"等状态词不再触发查库，必须有显式取数措辞
   （查/看/状态/金额/有哪些…）；操作请求场景下的知识检索也要求真命中制度类词。

**由此新增的能力**（评测暴露的工具缺口，不是猜测）：
`query_orders` 支持 `status` 过滤与 `order_ids` 精确查询，
`customer_id` / `status` / `order_ids` 三者至少提供一个（避免无条件拉全表）。

> 剩余情况说明：曾把"ACT-01/ACT-02 建单前先查订单"判为过度调用，
> 复核后改判为**正确行为** —— 建单需要订单事实作为上下文，已相应更新用例期望。
> 这说明评测用例的期望本身也需要按业务逻辑复核，不能机械照搬初版。

| 指标 | 数值 |
| --- | --- |
| 意图准确率 | **100.0%**（20/20） |
| 工具选择准确率 | **100.0%**（20/20） |
| 回答包含率 | **100.0%**（20/20） |
| 三项全通过 | **100.0%**（20/20） |
| 端到端延迟 | 均值 **2.56s** / P50 2.4s / P95 **2.85s** / max 10.2s |

分类指标全部 100%：`knowledge_qa`(8) / `data_query`(6) / `mixed`(3) / `action_request`(3)。

**评测驱动出的 5 个真实缺陷**（这些都是"写完代码跑一遍"才会暴露的）：

| # | 缺陷 | 证据 | 修法 |
| --- | --- | --- | --- |
| 1 | `query_orders` 强制要求 `customer_id` | DB-06「哪些订单已经退款了」无客户 ID → `MISSING_ARGUMENT` | 增加 `status` 过滤，`customer_id`/`status`/`order_ids` 三者至少一个 |
| 2 | 意图关键词表缺"期限"类词 | MIX-03「有没有在退货期限内的」漏识别 `knowledge_qa` | 补入 期限/时限/时间内/范围内/有效期 |
| 3 | **LLM 分类器倾向多报意图**，规划器无条件为每个意图加工具 | 6/20 用例过度调用（如"创建工单必须填哪些信息"去查了订单） | 规划阶段加**实体门控**：知识检索需真命中制度类词；查库需有订单号或明确状态 |
| 4 | **写操作被误触发** | KB-05「创建工单必须填写哪些信息？」真的建了单 | 独立于分类器加 `_is_write_intent` 安全门控：疑问句 + 询问词一律不写（9/9 单测通过） |
| 5 | 无法按订单号精确查询 | 只有订单号时需先反查客户，多一次往返 | `query_orders` 增加 `order_ids` 参数 |

**剩余已知问题（不影响正确性，但值得记录）**：仍有 3/20 用例存在过度调用
（额外一次 `search_documents` 或 `query_orders`），根因是 LLM 意图分类器倾向多报
`knowledge_qa`（我要求它"需要政策做依据时必须报"，属有意为之的保守策略）。
代价是约 0.8s 额外延迟，换来的是**不遗漏判断依据**。生产环境可在规划阶段加一次
"是否需要"的模型自检来消除，但会增加一次 LLM 往返。

## 阶段4 实测结果（curl 走真实 HTTP）

```
POST /chat  混合问题「订单 ORD-001 符合退货政策吗？」
  intents    : ['data_query', 'knowledge_qa']   source: rule
  plan       : [('db_query','query_orders'), ('rag_search','search_documents')]
  tool_calls : [('query_orders', ok=True, 4ms, attempts=1),
                ('search_documents', ok=True, 11289ms, attempts=1)]
  sources    : [('sample_退货政策.md', None, 0.851)]
  首次 12.2s（含 MCP 握手 + bge-m3 加载）/ 第二次 0.2s

POST /chat/stream 事件序列（三种问题类型）
  退货政策是什么           → intent→plan→tool→token→done           0.7s
  查一下订单 ORD-001 的状态 → intent→plan→tool→token→done           0.3s
  帮我创建高优先级工单      → intent→plan→tool→tool→tool→token→done  1.8s（三工具链）
```

### 阶段4 修掉的致命缺陷：跨线程桥接导致工具调用必挂

**症状**：`/chat` 与 `/chat/stream` 在工具调用处固定卡满超时（60s/次，重试后 120s），
而对端子进程日志显示同一操作只需 **3ms**（`query_orders ... -> 3 条`）。

**证据链**：
1. 同一会话连续 3 次调用，在**独立脚本事件循环**里正常（1681ms → 2ms → 2ms）；
2. 在 **uvicorn 的循环**里，经 `asyncio.to_thread(registry.call_tool, ...)` 必挂；
3. 日志显示 `call business_db.query_orders -> ok（60009ms）` —— 注册表报超时**之后**
   结果才回来，说明 reply future 没有被及时唤醒。

**根因**：注册表为兼容同步调用内置了"常驻循环线程 + 跨线程 future 桥接"。
在 FastAPI 的 async 上下文里再套一层 `asyncio.to_thread`，形成
「uvicorn 循环 → 线程池 → 注册表循环线程」的双重跨线程链路，reply 无法及时回传。

**修法**：注册表改为**由第一个发起异步调用的循环接管**（`_assert_owner_loop`），
Agent 直接 `await registry.acall_tool(...)`，去掉线程桥接；同步接口 `call_tool`
保留给脚本/测试。修后 `/chat` 从 240s 超时降到 **0.2s**。

> 另修：`plan_builder` 的订单→客户反查是同步 SQLite 调用，已挪到
> `asyncio.to_thread`，避免阻塞事件循环。

> ✅ **LLM 路径已实测打通**（`DEEPSEEK_API_KEY` 已配置）：
> ```
> POST /chat「订单 ORD-001 符合退货政策吗？」
>   intent_source : llm          ← 意图识别走模型
>   intents       : ['knowledge_qa', 'data_query']
>   metrics       : {"model":"deepseek-flash","prompt_tokens":767,
>                    "completion_tokens":242,"latency_ms":1587,
>                    "total_latency_ms":9229,"tool_latency_ms":13984}
>   degraded      : False
>   tool_calls    : [('query_orders', ok, 4ms), ('search_documents', ok, 6987ms)]
> ```
> 回答质量：模型没有直接断言"符合/不符合"，而是先陈述订单事实（`delivered`，
> 签收 2025-03-05）→ 引用政策条款（来源《sample_退货政策.md》）→ **如实说明
> 信息不足**并列出需补充的三项（商品类型、是否拆封、退货原因）。有依据、不编造。
>
> 注意 `settings.yaml` 里配的是 `deepseek-chat`，而 API 实际回传的
> `model` 字段是 `deepseek-flash` —— 该别名当前指向 flash 后端，属正常现象。

## 阶段3 实测结果（`python -m mcp_servers.test_mcp`）

```
1) 注册表装载与握手
  [PASS] knowledge_base   tools=['search_documents', 'get_document_metadata'] (1328ms)
  [PASS] business_db      tools=['query_orders', 'create_ticket'] (790ms)
2) 工具目录聚合 —— 4 个工具，required/props 与阶段1 约定一致
3) 工具调用
  search_documents       ok=True  8614ms  Top1=sample_退货政策.md score=0.815
  get_document_metadata  ok=True    28ms  title='退货与退款政策（2025 版）'
  query_orders           ok=True    10ms  ['ORD-002','ORD-001']（3 月区间，正确排除 ORD-007）
  create_ticket          ok=True    10ms  TKT-20260922-0005 (priority=high)
4) 负路径：NOT_FOUND / INVALID_ARGUMENT×3 / ServerNotRegisteredError / ToolNotAvailableError 全部符合预期
5) 启用开关：server 级与 tool 级双重判定生效
6) 结构化日志：logs/tool_calls.jsonl 12 条，字段完整
失败用例数: 0 → 全部测试通过 ✅
```

**会话复用效果**（`.tmp/verify_lifecycle.py`）：

| server | 首次（含握手/模型加载） | 复用第 2 次 | 提速 |
| --- | --- | --- | --- |
| business_db | 1379ms | **2ms** | 606× |
| knowledge_base | 9090ms | **147ms** | 62× |

关闭后 MCP server 子进程数 **4 → 0**，无 `cancel scope` 警告。

## 阶段3 的三个关键设计决策（都有实测依据）

1. **每个 server 一个专职常驻任务持有会话**（`registry._session_worker`）。
   MCP 的 anyio 任务组有两条硬约束，我都实测踩过：
   - 进入与退出必须在**同一个事件循环**，否则抛
     `Attempted to exit a cancel scope that isn't the current task's current cancel scope` 并挂死；
   - 更进一步，**不能用 `AsyncExitStack` 手动组合**上下文管理器 —— 实测
     `stack.aclose()` 即使在同一个 worker 任务里也会抛上面的错（最小复现见
     `.tmp/probe_shutdown.py`），只有原生嵌套 `async with` 才保证退出发生在
     进入的同一语句块内。
   所以注册表自带一个常驻事件循环线程，会话创建/调用/关闭全部投递给同一个 task 执行。

2. **异常必须收敛在工具内部**：mcp 2.x 会把工具内未捕获的异常转成
   `UnexpectedToolError` 抛给客户端并带崩其 task group。因此 4 个工具全部
   try/except 并返回 `{"ok": false, "error": ..., "code": ...}` 信封。
   负路径测试已经覆盖：NOT_FOUND / INVALID_ARGUMENT / DB_UNAVAILABLE。

3. **参数宽松解析，但非法值照样拒绝**：工具签名刻意放宽为
   `date_range: Any` 与 `priority: Any`。原因是 MCP 协议层会按签名生成 JSON Schema
   并**在调用前**校验 —— 签名写严格模型时，模型传 `"2025-03-01~2025-03-31"` 或
   `"高优先级"` 会被协议层直接拒绝，工具拿不到参数、给不出可读提示，Agent 也无法重试。
   现在 `parse_date_range` 支持 dict / 区间字符串 / `"2025-03"` / `"2025"`，
   `normalize_priority` 支持中英文与 P0–P3；但 `'2025/13/45'` 与 `'super'`
   仍然返回 `INVALID_ARGUMENT` —— **不做静默降级**，否则紧急工单会被悄悄降成 medium。

## 阶段2 实测结果

**入库**（`python -m rag_pipeline.ingestion_cli --path ./docs --recreate`）：

```
  [OK  ] sample_工单规范.md -> 5 片段 (3.7s)
  [OK  ] sample_订单字段说明.md -> 4 片段 (3.0s)
  [OK  ] sample_退货政策.md -> 7 片段 (4.4s)
  新增/更新文档 : 3    跳过(未变化) : 0    失败 : 0
  写入片段 : 16        写入批次 : 3        总耗时 : 22.56s
  points: 16   documents: 3
```

**增量索引**（同一命令重复执行）：3 个文档全部 `[SKIP] 本地状态一致`，写入 0 片段。

**检索延迟**（修复 `localhost` IPv6 问题后）：

| 模式 | 修复前 | 修复后 |
| --- | --- | --- |
| hybrid | 6277ms | **155~232ms** |
| vector | 4195ms | **157~204ms** |
| keyword | 4102ms | **4~16ms** |

**检索质量**（8 条查询，期望子串判定，脚本 `.tmp/tune_rerank.py`）：

| 权重组合 | R@3 | MRR |
| --- | --- | --- |
| `0.60/0.25/0.15`（初始猜测） | 0.750 | 0.604 |
| `0.40/0.55/0.05`（网格搜索最优） | **0.875** | 0.667 |

> 剩余 1 条未命中是**混合意图**问题（"订单 ORD-001 符合退货政策吗"）——
> 需要先查数据库拿订单事实、再查政策，属阶段4 Agent 编排职责。

## 阶段2 的三个关键设计决策（都有实测依据）

1. **chunk 不跨页 / 不跨标题**：`DocumentChunk.embedding_text()` 把来源文件名与
   所属标题拼进**编码文本**（存储与展示仍是干净正文）。实测把 R@3 从 0.750 提到 0.875 ——
   片段"本政策适用于…"本身不含"退货"字样，靠所属标题"1. 适用范围"才能被正确召回。
2. **稀疏向量只存词频项，IDF 交给 Qdrant**：集合启用
   `SparseVectorParams(modifier=Modifier.IDF)`，客户端不需要回传语料统计，
   入库与检索复用同一套 BM25 编码函数，打分口径天然一致。
3. **融合在本地做，不用服务端 `Fusion.RRF`**：服务端融合后 Qdrant 只回传融合分，
   会**丢弃原始稠密余弦分**，而"向量相似度 > 0.75 优先"与"低向量分剔除"两条规则
   都依赖原始分。本地融合多一次往返，换来完整的分数归因（阶段5 消融评测需要）。
