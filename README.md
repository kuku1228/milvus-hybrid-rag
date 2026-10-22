# 智识阁 · 大模型技术知识库问答（RAG）

> 基于 **Milvus 混合检索（Dense + BM25 稀疏）+ GTE-Rerank + DeepSeek Agent** 的中文 RAG 问答系统，带 **Gradio 流式对话**界面。

把技术文档（Markdown / PDF）解析、切分后灌入 Milvus，用「语义向量 + 关键词」双路召回，再经 Rerank 精排，最后由 DeepSeek 基于检索结果生成回答；多轮会话通过 MySQL 持久化。

---

## 功能特性

- **双索引混合检索**：dense（BGE 中文向量，512 维 / HNSW / IP）+ sparse（Milvus 内置 BM25 Function，jieba 分词），用 RRF 融合两路召回结果。
- **两段式排序**：先按 `Top_k_recall=20` 召回，再经 DashScope `gte-rerank-v2` 精排到 `Top_k_rerank=10`。
- **结构化解析**：基于 Unstructured 的 `elements` 模式，按 `parent_id` 把标题与正文合并成「标题 → 内容」，把章节路径写进正文，提升切片可读性与检索命中率。
- **语义切分**：单块超过 5000 字符时才交给 `SemanticChunker` 按语义边界二次切分，避免无脑定长切分割裂上下文。
- **多进程灌库**：生产者（解析）/ 消费者（写入）双进程 + 有界队列，解析慢、入库快互不阻塞，内存可控。
- **Agent 化问答**：检索被封装成 LangChain `@tool`，由 `create_agent` 自主决定是否检索，知识库无答案时明确告知而不编造。
- **流式输出**：Gradio 逐 token 推送，只渲染主模型节点的输出，跳过工具/子 Agent 的中间 LLM 调用。
- **会话持久化**：`PyMySQLSaver`（对话状态）+ `PyMySQLStore`（长期记忆），刷新页面/重启服务不丢上下文。

---

## 系统架构

```mermaid
flowchart LR
    subgraph 离线灌库
        A["data/*.md, *.pdf"] --> B["Unstructured 元素解析"]
        B --> C["标题-正文合并"]
        C --> D["超长块语义切分"]
        D --> E[("Milvus Collection<br/>dense: BGE 512维<br/>sparse: BM25")]
    end

    subgraph 在线问答
        Q["用户提问"] --> F["Agent: search_knowledge_base"]
        F --> G["混合召回<br/>dense + sparse + RRF"]
        E --> G
        G --> H["gte-rerank-v2 精排 Top10"]
        H --> I["DeepSeek 生成回答"]
        I --> J["Gradio 流式展示"]
        I <-.-> K[("MySQL<br/>会话状态 / 记忆")]
    end
```

**一次问答的完整链路**

1. 用户提问 → DeepSeek 判断是否需要查知识库；
2. 需要时调用 `search_knowledge_base`，BGE 把问题编码为向量，与原始文本分别走 dense / sparse 检索；
3. `RRFRanker(60)` 融合两路结果得到 20 条候选；
4. `gte-rerank-v2` 对候选重排，取前 10 条拼成「【文档N】标题 + 内容」返回；
5. 模型基于这些片段作答，回答以流式方式逐字输出到前端。

---

## 技术栈

| 层次 | 选型 |
|---|---|
| LLM | DeepSeek `deepseek-flash`（`langchain-deepseek`） |
| Agent 框架 | LangChain 1.x `create_agent` + LangGraph |
| 向量库 | Milvus（`pymilvus`；本地开发可用 Milvus Lite） |
| 稠密向量 | `BAAI/bge-small-zh-v1.5`（`sentence-transformers`，CPU 推理，L2 归一化） |
| 稀疏向量 | Milvus 内置 BM25 Function（jieba tokenizer + cnalphanumonly 过滤器） |
| 重排模型 | 阿里云 DashScope `gte-rerank-v2` |
| 文档解析 | `langchain-unstructured` + `unstructured`（elements 模式） |
| 语义切分 | `langchain-experimental` `SemanticChunker`（DashScope Embedding 作为切分度量） |
| 会话持久化 | `langgraph-checkpoint-mysql`（PyMySQLSaver / PyMySQLStore） |
| 前端 | Gradio 6 |
| 日志 | Loguru |

---

## 项目结构

```
RAG/
├── config.py                     # 全局检索参数：召回数 / 重排后保留数
├── gradio_web.py                 # Gradio Web 入口（流式问答）
├── agent/
│   ├── rag_agent.py              # 命令行版 Agent（MySQL 持久化 + 工具调用）
│   └── rag_agent_demo.py         # 轻量示例：tool-calling agent + 内存历史
├── tools/
│   └── retriever_tools.py        # search_knowledge_base 工具封装
├── RAGRetriever/
│   └── RAGRetriever.py           # 混合检索 + Rerank 核心实现
├── documents/
│   ├── milvus_db.py              # Collection Schema / 索引 / 写入封装
│   ├── write_milivus.py          # 多进程灌库入口（解析进程 + 写入进程）
│   ├── markdown_parser.py        # Markdown 解析 + 合并 + 语义切分
│   ├── pdf_parser.py             # PDF 解析 + 合并 + 语义切分
│   └── test_datas.py             # 集合数据抽查脚本
├── LLM_Models/
│   ├── env_utils.py              # 环境变量集中读取（含 Collection 名）
│   ├── my_llm.py                 # DeepSeek 模型实例
│   ├── embedding_model.py        # BGE 向量模型 / DashScope Embedding
│   └── log_utils.py              # Loguru 日志配置
├── search_tools/                 # 三种检索方式对照实验（dense / BM25 / hybrid）
├── test_load/                    # 文档加载与解析实验脚本
├── test_vector/                  # 向量与 Milvus 建表实验脚本
├── test_milvus/                  # Milvus 基础增删改查实验脚本
├── data/                         # 知识库原始文件（Markdown / PDF）
└── .env                          # 密钥配置（不要提交到 Git）
```

---

## 快速开始

先克隆仓库并进入目录：

```bash
git clone https://github.com/kuku1228/milvus-hybrid-rag.git
cd milvus-hybrid-rag
```

### 1. 环境要求

- Python **3.12**（开发环境 3.12.9）
- Milvus：**2.5+** 服务端，或用 Milvus Lite（无需部署，指定本地 `.db` 文件即可）
- MySQL **8.0+**（仅 Web / CLI 的会话持久化需要）
- 阿里云 DashScope 账号（Embedding 语义切分 + Rerank）
- DeepSeek API Key
- 首次运行需下载 `BAAI/bge-small-zh-v1.5`

### 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt   # 或参考下方「参考依赖版本」直接安装
```

参考依赖版本（开发环境实测可用）：

```text
langchain==1.4.7
langchain-core==1.6.13
langchain-community==0.4.2
langchain-experimental==0.4.2
langchain-text-splitters==1.1.3
langchain-classic==1.0.8
langchain-deepseek==1.1.1
langchain-milvus==0.4.0
langchain-huggingface==1.2.3
langchain-unstructured==1.0.1
langgraph==1.2.16
langgraph-checkpoint-mysql==3.0.0
pymilvus==3.0.2
milvus-lite==3.2.2
dashscope==1.27.8
gradio==6.30.0
sentence-transformers==6.1.0
unstructured==0.27.30
python-dotenv==1.2.4
loguru==0.7.3
PyMySQL==1.2.3
torch==2.14.1        # 建议按平台/是否使用 GPU 单独安装
```

### 3. 配置环境变量

在项目根目录创建 `.env`（已在 `.gitignore` 中）：

```dotenv
# DeepSeek（Agent 主模型）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 阿里云 DashScope（语义切分的 Embedding + GTE-Rerank）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# HuggingFace（下载 BGE 模型）
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxx

# Milvus：服务端地址，或 Milvus Lite 本地文件路径，如 ./milvus_demo.db
MILVUS_URI=http://127.0.0.1:19530

# Unstructured 云端高精度解析（仅 test_load/demo1.py 用到）
UNSTRUCTURED_API_KEY=xxxxxxxxxxxxxxxx

# MySQL（会话持久化）
MYSQL_URL=mysql://user:password@127.0.0.1:3306/rag
```

> Collection 名不在 `.env` 中，定义在 [`LLM_Models/env_utils.py`](LLM_Models/env_utils.py) 的 `COLLECTION_NAME`，默认 `t_collection02`，换知识库时改这里。

### 4. 准备知识库

把 Markdown / PDF 文件放进 `data/` 目录。默认灌库入口会扫描该目录下所有 `.md` 与 `.pdf`：

```python
# documents/write_milivus.py
data_dir = r'D:\Python\PycharmProjects\RAG\data'   # ← 改成你的目录
```

### 5. 建库并写入数据

```bash
python -m documents.write_milivus
```

该入口会先重建 Collection（删除旧集合、索引后重新创建），再启动两个进程：

- **解析进程**：逐文件解析 → 合并标题正文 → 语义切分，攒够 20 条文档放入队列；
- **写入进程**：从队列取批次写入 Milvus，日志中会打印累计写入条数。

> 注意：重建逻辑会清空同名 Collection，切换知识库时请确认 `COLLECTION_NAME`。

### 6. 启动服务

**Web 界面（推荐）**

```bash
python gradio_web.py
# 浏览器访问 http://127.0.0.1:7860
```

首次启动会自动执行 `checkpointer.setup()` / `store.setup()` 建表。界面支持流式输出、示例问题、清空对话（清空即更换 `thread_id`，后端上下文一并重置）。

**命令行对话**

```bash
python -m agent.rag_agent
```

固定 `thread_id="user_001"` 保持多轮上下文，输入 `exit` / `quit` / `退出` 结束。

---

## 检索链路详解

### 1. 解析：从「元素」还原「章节」

Unstructured 以 `elements` 模式输出，每个元素带 `category`（Title / NarrativeText / Table …）、`element_id`、`parent_id`。`merge_title_content()` 做了三件事：

- `parent_id is None` 的 NarrativeText 直接保留（独立成段）；
- 遇到 Title 时把标题文本写入 `metadata["title"]`；若其父元素已存在，则拼接成 `父标题 -> 当前标题`，形成章节路径；
- 带 `parent_id` 的非标题元素，其正文累积到父节点上，父节点 `category` 标记为 `content`。

### 2. 切分：只对超长块做语义切分

`len(page_content) > 5000` 的块才用 `SemanticChunker(breakpoint_threshold_type="percentile")` 切分，度量模型为 DashScope Embedding。短块原样保留，减少 Embedding 调用与切分噪声。

### 3. 存储：一个 Collection 承载两种索引

- `dense`：`FLOAT_VECTOR`，dim = **512**（`bge-small-zh-v1.5` 输出维度），HNSW 索引，metric = `IP`；
- `sparse`：`SPARSE_FLOAT_VECTOR`，由 Milvus 内置 BM25 Function 从 `text` 字段自动生成，无需在客户端手写稀疏向量；
- `text` 字段开启 `enable_analyzer`，分词器 `jieba` + `cnalphanumonly` 过滤器（适配中文）；
- 另存 `title` / `category` / `category_depth` / `source` / `filename` / `filetype` 便于过滤与溯源。

### 4. 召回与重排

| 步骤 | 参数 | 说明 |
|---|---|---|
| dense 召回 | `metric_type=IP`, `ef=64`, `limit=20` | 语义相似 |
| sparse 召回 | `metric_type=BM25`, `limit=20` | 关键词精确命中，弥补向量对专有名词的迟钝 |
| 融合 | `RRFRanker(60)` | 只依赖排名，天然规避两路分数量纲不一致 |
| 重排 | `gte-rerank-v2`, `top_n=10` | 交叉编码器精排，显著提升 top 段质量 |

---

## 关键参数与调优

| 位置 | 参数 | 默认 | 调优建议 |
|---|---|---|---|
| `config.py` | `Top_k_recall` | 20 | 召回不足导致漏答时先加大到 30~50，代价是重排耗时上升 |
| `config.py` | `Top_k_rerank` | 10 | 直接决定塞进 Prompt 的上下文长度，注意 token 预算 |
| `documents/milvus_db.py` | `ef` | 64 | 检索精度/延迟折中，可试 64 / 128 |
| `documents/milvus_db.py` | `M` / `efConstruction` | 16 / 64 | 建索引成本 vs 召回质量 |
| `documents/milvus_db.py` | `bm25_k1` / `bm25_b` | 1.2 / 0.75 | k1 词频饱和度（1.2~2.0），b 长度归一化（0~1） |
| `search_tools/test_search.py` | `drop_ratio_search` | 0.0 | 调大可忽略查询中的低权重词，增强鲁棒性 |
| `documents/*_parser.py` | 语义切分阈值 | 5000 字符 | 太小切碎上下文，太大稀释向量语义 |

对照实验脚本 [`search_tools/test_search.py`](search_tools/test_search.py) 同时实现了纯 dense、纯 BM25、hybrid+RRF 三种检索，可直接对比效果差异。

---

## 常见问题

**Q：Milvus 连不上？**
确认 `MILVUS_URI`：服务端模式形如 `http://127.0.0.1:19530`；本地模式可填文件路径（Milvus Lite），无需额外部署。

**Q：首次运行卡在下载模型？**
`bge-small-zh-v1.5` 需从 HuggingFace 拉取，国内网络可设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`（macOS/Linux 用 `export`）。

**Q：dense 维度为什么是 512？**
`bge-small-zh-v1.5` 输出 512 维；换用其他 Embedding 模型时必须同步修改 `documents/milvus_db.py` 中 `dense` 字段的 `dim`，否则插入报错。

**Q：灌库报 `max_length` 超限？**
`text` 字段上限 8000，语义切分阈值 5000 已留出余量；若自定义解析器调大了阈值，需同步调整 schema。

**Q：中文分词效果差？**
确认 `text` 字段的 analyzer 仍是 `{"tokenizer": "jieba", "filter": ["cnalphanumonly"]}`，且 BM25 Function 的 `input_field_names` 指向 `text`。

**Q：Web 启动报 MySQL 连接失败？**
`MYSQL_URL` 需为 `mysql://user:password@host:port/db` 形式，且库已存在；表由 `checkpointer.setup()` / `store.setup()` 自动创建。

**Q：想换成自己的知识库？**
三步：文件放入 `data/` → 修改 `documents/write_milivus.py` 的 `data_dir` → 按需修改 `COLLECTION_NAME` 后重新执行灌库入口。

---

## 开发调试脚本

以下脚本用于分步验证，可按需单独运行：

| 脚本 | 用途 |
|---|---|
| [`test_load/demo3.py`](test_load/demo3.py) | 直接用 Unstructured 加载 Markdown，观察 elements 结构 |
| [`test_load/demo4.py`](test_load/demo4.py) | Markdown 解析 + 标题正文合并 + 语义切分全流程 |
| [`test_load/demo1.py`](test_load/demo1.py) | Unstructured 云端 `hi_res` 高精度解析（含表格 `text_as_html`），需要 `UNSTRUCTURED_API_KEY` |
| [`test_vector/demo1.py`](test_vector/demo1.py) | 验证 BGE 向量维度与相似度 |
| [`test_vector/demo2.py`](test_vector/demo2.py) | Milvus 稀疏向量 + BM25 索引建表与全文检索 |
| [`test_milvus/demo1.py`](test_milvus/demo1.py) | Milvus 基础操作：建集合 / 插入 / 搜索 / 过滤查询 / 删除 |
| [`documents/test_datas.py`](documents/test_datas.py) | 抽查集合行数、前 5 条数据与 BM25 命中效果 |

---

## 说明

- 本项目用于学习与技术交流，实现上追求链路完整、参数可解释，而非生产级高可用。
- `data/` 中的示例文档请自行确认版权，**不建议**把第三方 PDF 一并提交到公开仓库。
- 仓库暂未附带 `LICENSE` 文件，如需正式开源建议补充（如 MIT）。

---

如果这个项目对你有帮助，欢迎 Star ⭐
