# HAL1000 — B1 Agent 运行时 & B4 LLM 接入层

> **李芊芊 · 个人模块说明**
> 仓库地址：https://github.com/0vos/HAL1000

---

## 1. 模块概述

### 1.1 模块名称

`B1：Agent 运行与消息管理模块` + `B4：Agent LLM 决策模块`

### 1.2 模块说明

我负责 B1 和 B4 两个模块，二者共同构成整个 Agent 系统的核心调度层与推理引擎。

**B1（`hal_chat.py` + `task_planner.py` + `task_executor.py`）**是整个系统的"大脑"。它接收用户自然语言输入，调用 B4 将其解析为 JSON 任务图（DAG），并发调度工具调用，通过 LLM 语义裁判验证执行结果，维护完整的多轮消息历史，并将结果归档至 B5 记忆系统。没有 B1，B2/B3/B5 各自都是孤立的工具函数，无法串联成一次完整的 Agent 任务。

**B4（`b4_local_agent_llm.py` + `b4_model_switch.py`）**是系统的"推理引擎"。它将 Qwen3.5-4B 本地模型封装为标准接口，提供 `generate_ai_message`（工具调用决策）、`generate_text_only`（Planner JSON 生成）、`generate_vision_answer`（视觉问答）三个功能，并维护全局模型缓存避免重复加载。没有 B4，系统无法在完全本地、无网络的条件下运行。

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| B1 基础要求 | ✅ 多轮对话、tool_calls 循环、max_turns 防死循环、消息序列维护、B5 记忆注入 |
| B4 基础要求 | ✅ 读取 model.yaml、工具绑定、AIMessage 生成与解析、raw_model_output 记录 |
| B1 PPT 进阶 | ✅ 多轮 tool_calls 循环、断点续跑（--resume）、批量任务、消息压缩、多阶段 system prompt |
| B4 PPT 进阶 | ✅ 单轮多 tool_calls、Plan-and-Execute（DAG）、多阶段模型切换、视觉 VQA |
| B1 自研进阶 | ✅ LLM 语义裁判、DAG 并发调度、人在环确认、fallback 链、产物版本管理、/stop 中断、/branch /undo、代码入口点验证、阻塞型代码识别 |
| B4 自研进阶 | ✅ json_mode 参数（Planner/代码生成场景隔离）、全局模型缓存、三阶段代码验证 |
| 可独立运行的演示 | ✅ `python hal_chat.py --mode prompt_json` 交互式终端；各子模块可独立命令行调用 |
| 与团队系统集成情况 | ✅ 通过 `execute_tool_calls()` 调用 B3；通过 `b5_memory.py` 接口读写 B5；所有格式以 `common/schemas.py` 为标准 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.10 |
| 必要依赖 | torch、transformers、PyYAML、sentencepiece、pillow |
| 是否需要模型 | 需要（Qwen3.5-4B，约 8GB） |
| 是否需要 GPU | 推荐（≥10GB 显存）；CPU 可运行但推理约慢 30-90s |
| 是否需要外部数据集 | 不需要；测试样例已内置在 `data/` 目录 |

### 2.2 模型依赖

| 模型 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| Qwen3.5-4B | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.5-4B) | `../Qwen3.5-4B`（由 `configs/model.yaml` 配置） | 主推理模型（工具调用 + 生成 + Planner） |
| all-MiniLM-L6-V2（可选） | HuggingFace | `code/models/all-MiniLM-L6-V2/` | Episodic Memory 向量检索（fallback 到 BM25） |

```bash
# 设置模型路径环境变量（或直接修改 configs/model.yaml）
export HAL_MODEL_PATH=/path/to/Qwen3.5-4B
```

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| `b1_fixture_input.json` | 项目自带 | `data/b1_fixtures/` | B1 基础演示固定输入 |
| `runtime_input.json` | 项目自带 | `data/` | 全系统集成演示输入 |
| `messages_no_tool.json` | 项目自带 | `data/messages/` | B4 单独推理演示输入 |
| `tools_schema_basic.json` | 项目自带 | `data/messages/` | B4 工具绑定演示用 schema |
| `IMG_0532.png` | 项目自带 | `data/` | 视觉问答（VQA）演示图片 |
| `GDPRPrivacyNotice.pdf` | 项目自带 | `data/` | 多步文档分析演示 |

### 2.4 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/0vos/HAL1000
cd HAL1000/agent

# 2. 创建并激活 conda 环境
conda create -n hal python=3.10 -y
conda activate hal
export PYTHONNOUSERSITE=1

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置模型路径
export HAL_MODEL_PATH=/path/to/Qwen3.5-4B

# 5. 验证安装
cd code
python -c "from b4_local_agent_llm import generate_ai_message; print('B4 OK')"
python -c "from task_planner import plan; print('B1 OK')"
```

**常见问题：**
- `local model path does not exist`：未设置 `HAL_MODEL_PATH`，见 `docs/PORTABILITY.md`
- `flash-linear-attention not installed`：正常 warning，不影响运行
- GPU 显存不足：在 `configs/model.yaml` 中改 `torch_dtype: float32`

---

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
agent/
├── code/
│   ├── hal_chat.py              # B1 主控制器：多轮 REPL、FSM 状态机、LLM 裁判、/stop /branch /undo
│   ├── task_planner.py          # B1 LLM 规划器：用户输入 → JSON DAG（TaskDAG 数据结构）
│   ├── task_executor.py         # B1 DAG 执行器：并发调度、节点 Retry、参数解析、代码验证
│   ├── b4_local_agent_llm.py    # B4 推理引擎：generate_ai_message / generate_text_only / generate_vision_answer
│   ├── b4_model_switch.py       # B4 多阶段模型切换（plan / execute / summarize）
│   ├── b1_compress.py           # B1 上下文压缩（compress_after=10, keep_recent=6）
│   ├── b1_checkpoint.py         # B1 会话保存
│   ├── b1_resume.py             # B1 会话恢复（--resume）
│   ├── b1_batch_runner.py       # B1 批量任务运行
│   ├── artifact_registry.py     # B1 产物版本管理（rollback + diff）
│   ├── episodic_memory.py       # B1/B5 分层记忆（SQLite + BM25 / 向量）
│   └── common/
│       └── schemas.py           # 统一格式：AIMessage / ToolMessage / SkillResult / normalize_tool_call
├── configs/
│   ├── model.yaml               # 模型路径、生成参数
│   ├── tools.yaml               # 工具集定义
│   └── model_roster.yaml        # 多阶段模型切换配置
├── prompts/
│   ├── local_tool_agent.txt     # FSM 执行阶段 system prompt
│   └── task_planner.txt         # Planner 阶段 system prompt（要求输出 JSON DAG）
├── data/
│   ├── b1_fixtures/             # B1 基础演示固定输入
│   ├── messages/                # B4 演示用 messages / schema
│   └── ...
└── outputs/
    ├── sessions/                # 会话持久化文件（session_*.json）
    └── artifacts/               # 产物版本记录
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| 输入（B1） | 用户终端 / `runtime_input.json` | 自然语言字符串 / JSON | 用户问题、toolset、max_turns 等配置 |
| 输入（B4） | B1 调用 | `messages: list[dict]`、`tools_schema: list` | 多轮消息历史 + 工具 schema |
| 输出（B4） | 返回给 B1 | `AIMessage`（含 `content` 或 `tool_calls`） | LLM 生成的工具调用决策或最终回答 |
| 调用 B3（张立杭） | B1 → B3 | `execute_tool_calls(tool_calls, tools_config, toolset)` | 工具分发执行，返回 `ToolMessage[]` |
| 调用 B5（楚可欣） | B1 → B5 | `recall(query)` / `save(messages, trace, answer)` | 记忆召回与归档 |
| 输出（B1） | 终端打印 / 文件 | 自然语言 + `outputs/sessions/*.json` | 最终回答 + 会话持久化 |

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

**B1 基础功能：**
- 接收用户问题，调用 B5 获取记忆上下文，构造初始 `messages`（SystemMessage + HumanMessage）
- 调用 B3 获取 `tools_schema`，传给 B4 完成工具绑定
- 调用 B4，获得 `AIMessage`，判断是否含 `tool_calls`
- 若含 `tool_calls`，调用 B3 执行工具，获取 `ToolMessage`，追加进 `messages`，循环直至最终回答
- 支持 `max_turns` 防止死循环，输出 `final_answer.md` + `messages.json` + `trace.json`

**B4 基础功能：**
- 读取 `model.yaml`，加载本地 Qwen3.5-4B
- 接收 `messages` + `tools_schema`，完成 prompt 注入，调用模型推理
- 将原始输出解析为标准 `AIMessage`（含 `content` 或 `tool_calls`）
- 记录 `raw_model_output.json` 和 `ai_message.json`

### 4.2 基础功能实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `hal_chat.py::HALChat.chat()` | B1 主循环入口，维护 messages、调用 B3/B4/B5 |
| `b4_local_agent_llm.py::generate_ai_message()` | B4 核心推理接口，返回 AIMessage |
| `b3_tool_layer.py::execute_tool_calls()` | B3 工具执行（张立杭负责），B1 调用 |
| `common/schemas.py::make_ai_message()` | 将模型原始输出解析为标准格式 |

```text
用户输入
  → B5 召回记忆 → 构造 messages
  → B3 生成 tools_schema → 传给 B4
  → B4 推理 → AIMessage
  → 含 tool_calls？
      是 → B3 执行工具 → ToolMessage → 追加 messages → 回到 B4
      否 → 输出 final_answer → B5 保存记忆
```

### 4.3 基础功能输入格式与样例

| 字段 | 类型 | 是否必需 | 说明 |
|---|---|---|---|
| `user_input` | 字符串 | 是 | 用户自然语言问题 |
| `system_prompt_path` | 路径字符串 | 否 | 默认使用 `prompts/local_tool_agent.txt` |
| `toolset` | 字符串 | 否 | 默认 `basic_tools`，可选 `all_tools` |
| `max_turns` | 整数 | 否 | 默认 10，防止工具调用死循环 |
| `save_memory` | 字符串 | 否 | `conversation` 或 `global`，控制是否归档记忆 |

| 样例文件 | 用途 |
|---|---|
| `data/b1_fixtures/b1_fixture_input.json` | B1 基础演示：使用预设消息，不调用真实模型 |
| `data/runtime_input.json` | 全系统集成演示：真实模型 + 真实工具调用 |
| `data/messages/messages_no_tool.json` | B4 独立演示：无工具调用，直接生成回答 |

### 4.4 基础功能演示命令

```bash
cd agent/code

# B1 基础演示（fixture 模式，无需 GPU）
python b1_agent_runtime.py \
    --input ../data/b1_fixtures/b1_fixture_input.json \
    --outdir ../outputs/B1_fixture

# B1 全系统演示（真实模型）
python b1_agent_runtime.py \
    --input ../data/runtime_input.json \
    --tools_config ../configs/tools.yaml \
    --memory_config ../configs/memory.yaml \
    --model_config ../configs/model.yaml \
    --llm_mode integrated \
    --outdir ../outputs/B1_runtime

# B4 独立演示（无工具，直接回答）
python b4_local_agent_llm.py \
    --model_config ../configs/model.yaml \
    --messages ../data/messages/messages_no_tool.json \
    --tools_schema ../data/messages/tools_schema_basic.json \
    --mode prompt_json \
    --outdir ../outputs/B4_llm/no_tool

# 交互式终端对话（推荐验收使用）
python hal_chat.py --mode prompt_json \
    --model_path /path/to/Qwen3.5-4B
```

运行后应观察：
- B1 fixture：`outputs/B1_fixture/` 下生成 `messages.json`、`trace.json`、`final_answer.md`
- B1 全系统：终端打印工具调用轮次、LLM 最终回答，`outputs/B1_runtime/` 下生成完整记录
- B4 独立：`outputs/B4_llm/no_tool/` 下生成 `raw_model_output.json`、`ai_message.json`
- 交互式：终端显示 `User >` 提示符，可多轮输入问题

### 4.5 基础功能输出格式

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/*/messages.json` | JSON | 完整多轮消息序列（System / Human / AI / Tool） |
| `outputs/*/final_answer.md` | Markdown | Agent 最终回答文本 |
| `outputs/*/trace.json` | JSON | 执行 trace（turns、tool_rounds、llm_call_count） |
| `outputs/*/raw_model_output.json` | JSON | B4 模型原始输出（未解析） |
| `outputs/*/ai_message.json` | JSON | B4 解析后的标准 AIMessage |
| `outputs/sessions/session_*.json` | JSON | 会话持久化文件，含完整 messages + 元信息 |

### 4.6 基础功能结果截图

<figure>
<img src="运行截图/截屏2026-07-12%2017.52.08.png" alt="B1 DAG调度完成终端日志" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">B1 DAG 调度完成——终端日志展示节点并发启动与完成</figcaption>
</figure>

<figure>
<img src="运行截图/截屏2026-07-12%2017.53.14.png" alt="B4 LLM裁判输出" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">B4 LLM 推理 + 裁判输出 done=true/false 判断结果</figcaption>
</figure>

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求 | 是否完成 | 对应文件 / 函数 | 简要说明 |
|---|---|---|---|
| 多轮 tool_calls 循环 | ✅ | `hal_chat.py::chat()` FSM 循环 | RUNNING→JUDGING 状态机，最多 max_turns 轮 |
| 断点续跑（--resume） | ✅ | `b1_checkpoint.py` + `b1_resume.py` | session_id 恢复完整 messages 上下文 |
| 批量任务运行 | ✅ | `b1_batch_runner.py` | 读取批量输入 JSON，串行执行多个 Agent 任务 |
| 历史消息压缩 | ✅ | `b1_compress.py` | compress_after=10，keep_recent=6，自动触发 |
| 多 system prompt 切换 | ✅ | `b4_model_switch.py` | plan / execute / summarize 三阶段切换 |
| 单轮多 tool_calls | ✅ | `b4_local_agent_llm.py` | AIMessage 支持 tool_calls 列表，B3 并发执行 |
| Plan-and-Execute | ✅ | `task_planner.py` + `task_executor.py` | LLM 生成 DAG，并发调度执行 |
| 多阶段本地模型切换 | ✅ | `b4_model_switch.py` | 不同阶段可切换不同模型参数或配置 |
| 视觉 VQA | ✅ | `b4_local_agent_llm.py::generate_vision_answer()` | 主进程直调，支持图片路径输入 |
| **LLM 语义裁判**（自研） | ✅ | `hal_chat.py::_judge_by_llm()` | 双层裁判：LLM 主路径 + 关键词 fallback |
| **DAG 并发调度**（自研） | ✅ | `task_executor.py::execute_dag()` | fork + 50ms poll，无依赖节点真正并发 |
| **人在环计划确认**（自研） | ✅ | `hal_chat.py::_confirm_plan()` | 执行前展示计划，支持用户修改或取消 |
| **产物版本管理**（自研） | ✅ | `artifact_registry.py` | 工具产物版本化，支持 rollback + diff |
| **/stop 中断**（自研） | ✅ | `task_executor.py` 后台线程 | 后台线程监听 stdin，安全退出 DAG |
| **json_mode 参数**（自研） | ✅ | `b4_local_agent_llm.py::generate_text_only()` | Planner 用 True，代码生成用 False，场景隔离 |
| **全局模型缓存**（自研） | ✅ | `b4_local_agent_llm.py::_MODEL_CACHE` | 避免重复加载，推理复用同一 (tokenizer, model) |

### 5.2 进阶功能一：DAG 并发调度 + 人在环确认

#### 功能说明

基础版 B1 是单步串行的 ReAct 循环，每次只执行一个工具。而对于"读三个文件后综合分析"这类任务，三次读取本可以并发——因此我设计了 DAG 并发调度器，让无依赖关系的节点真正同时执行，并在执行前加入人在环确认，让用户可以修改计划或取消。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `task_planner.py::plan()` | 调用 B4 生成 JSON DAG，解析为 TaskDAG 数据结构 |
| `hal_chat.py::_confirm_plan()` | 展示任务计划，等待用户确认/修改/取消 |
| `task_executor.py::execute_dag()` | 并发调度，50ms poll 轮询子进程，依赖解锁后继节点 |
| `task_executor.py::_resolve_args()` | 解析占位符（`__GENERATE__` / `__FROM_FILE__`） |

```text
用户输入
  → Planner (B4 json_mode=True) → TaskDAG
  → _confirm_plan() 人在环确认
  → execute_dag()：
      ready_nodes() → 并发 fork 子进程
      50ms poll → 节点完成 → 解锁后继节点
      全部完成 → B4 Summarize → _judge_by_llm() 裁判
      裁判驳回 → fallback FSM
```

#### 输入格式与样例

| 字段 | 类型 | 是否必需 | 说明 |
|---|---|---|---|
| `user_input` | 字符串 | 是 | 任意自然语言，Planner 自动拆解 |
| `--mode prompt_json` | 参数 | 是（真实模型） | 使用本地 Qwen3.5-4B 推理 |
| `--toolset all_tools` | 参数 | 否 | 使用全部工具（含 shell_exec、image_qa） |

#### 演示命令

```bash
cd agent/code

# 启动交互式 Agent，输入多步任务触发 DAG
python hal_chat.py --mode prompt_json \
    --model_path /path/to/Qwen3.5-4B

# 示例输入（在 User > 提示符后）：
# 读取 /path/to/文件.docx，总结主要内容，并判断代码是否满足要求
```

#### 输出格式

| 输出 | 格式 | 说明 |
|---|---|---|
| 终端日志 `[▶ tN tool]` | 文本 | 节点启动日志 |
| 终端日志 `[✓ tN tool]` | 文本 | 节点完成日志（含耗时） |
| `outputs/artifacts/session_*.json` | JSON | 各节点产物版本记录 |

#### 示例图片

<figure>
<img src="运行截图/截屏2026-07-12%2017.24.16.png" alt="Planner生成DAG并展示计划确认" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">Planner 生成 DAG 后展示任务计划，等待用户确认</figcaption>
</figure>

<figure>
<img src="运行截图/截屏2026-07-13%2019.06.26.png" alt="多步文件分析任务执行完整过程" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">多步文件分析任务——Agent 依次调用工具读取文件并生成综合报告</figcaption>
</figure>

### 5.3 进阶功能二：LLM 语义裁判 + fallback 链

#### 功能说明

基础版 B1 只要模型不再输出 tool_calls 就算任务完成，但 Qwen3.5-4B 小模型经常在任务未真正完成时就给出空洞的"已完成"回答。因此我设计了 LLM 语义裁判：由同一个 Qwen 模型扮演裁判角色，判断回答是否真正满足用户需求，不满足则注入 nudge 提示让模型重新执行，最多重试 3 次。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `hal_chat.py::_judge_by_llm()` | LLM 裁判主路径，输出 `{"done": bool, "reason": str}` |
| `hal_chat.py::_judge()` | 裁判入口，LLM 失败时自动 fallback 到关键词规则裁判 |
| `prompts/task_planner.txt` | Planner 专用 system prompt，要求输出 JSON |

```text
模型给出最终回答
  → _judge_by_llm(user_input, model_answer)
      → done=true → DONE，输出给用户
      → done=false → 注入 nudge → RUNNING 继续
      → nudge ≥ 3 次 → FAILED
  → LLM 裁判本身崩溃 → fallback 关键词规则裁判
```

#### 演示命令

```bash
# 裁判在交互式运行中自动触发，无需单独命令
# 终端会打印：[✓ 裁判] done=True 或 [驳回] reason: ...
python hal_chat.py --mode prompt_json \
    --model_path /path/to/Qwen3.5-4B
```

#### 输出格式

| 输出 | 格式 | 说明 |
|---|---|---|
| 终端 `[✓ 裁判] done=True` | 文本 | 裁判通过，任务完成 |
| 终端 `[驳回] reason: ...` | 文本 | 裁判驳回，说明原因，触发 nudge |

#### 示例图片

<figure>
<img src="运行截图/截屏2026-07-12%2017.53.14.png" alt="LLM裁判输出done判断" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">LLM 裁判输出 done=true/false，决定任务是否真正完成</figcaption>
</figure>

### 5.4 进阶功能三：断点续跑（--resume）+ 会话持久化

#### 功能说明

Agent 在执行长任务时可能被意外中断（Ctrl+C、服务器断连等）。我实现了完整的会话持久化机制：每轮对话结束后自动将完整 messages、session_id、执行状态写入 `outputs/sessions/session_*.json`，支持通过 `--resume session_id` 从中断点恢复。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `b1_checkpoint.py::save_session()` | 每轮后保存 messages + 元信息到 session JSON |
| `b1_resume.py::load_session()` | 读取 session JSON，恢复 messages 和 artifacts |
| `hal_chat.py` `--resume` 参数 | 启动时检测并加载历史 session |

#### 演示命令

```bash
# 正常启动，自动保存 session
python hal_chat.py --mode prompt_json --model_path /path/to/Qwen3.5-4B
# 终端会显示：[session] 已保存 session_xxxxxxxx

# 恢复历史 session
python hal_chat.py --mode prompt_json \
    --model_path /path/to/Qwen3.5-4B \
    --resume session_xxxxxxxx
```

#### 输出格式

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/sessions/session_*.json` | JSON | 完整 messages 序列 + conversation_id + 时间戳 |

#### 示例图片

<figure>
<img src="运行截图/截屏2026-07-12%2019.16.57.png" alt="sessions目录下多个持久化session文件" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">sessions/ 目录下积累的多个会话持久化文件，每个对应一次完整对话</figcaption>
</figure>

<figure>
<img src="运行截图/截屏2026-07-13%2016.50.09.png" alt="checkpoint.json内容展示" style="max-width:100%">
<figcaption style="text-align:center;color:#666;font-size:0.9em">checkpoint.json 内容：含 conversation_id、resume_from_turn 和完整 messages</figcaption>
</figure>

---

## 6. 与团队系统的集成说明

B1 是整个团队系统的调度中心，而 B4 是 B1 内部的推理能力来源，二者对外的集成关系如下：

**调用 B3（张立杭）：**
B1 通过 `execute_tool_calls(tool_calls, tools_config, toolset)` 接口调用 B3，B3 返回 `ToolMessage[]`，B1 将其追加进 `messages`。联调初期遇到模型把参数放在 tool_call 顶层（不嵌套在 `args` 里）的格式问题，我在 `common/schemas.py::normalize_tool_call()` 中增加了顶层字段兜底，统一修复后联调正常。

**调用 B5（楚可欣）：**
B1 在每轮开始前调用 `recall(query)` 获取历史记忆，注入 system prompt；每轮结束后调用 `save()` 归档本轮对话。B5 的记忆格式为字符串列表，B1 将其拼接后注入 messages。

**被 B3 依赖的 Skills（杨贺淳）：**
B2 的各 Skill 中，`docx_reader`、`pdf_reader`、`image_qa` 因加载 transformers 库在子进程中超时（>120s），我将这三个 Skill 改为主进程直接调用，绕过子进程隔离限制，同时共享 B4 已缓存的模型权重，解决了联调中最难定位的超时问题。

**统一格式约定：**
所有模块间数据传递以 `common/schemas.py` 中定义的格式为准，由我统一维护。联调时若出现字段不一致（如 `output` vs `result`），以 `schemas.py` 为准修改调用方，不修改格式定义本身。

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| 沙箱非完全隔离 | 使用 `signal.SIGALRM` + 受限 builtins，非容器级隔离 | 改用 RestrictedPython / Docker 沙箱 |
| LLM 裁判额外耗时 1-2s | 每次裁判需完整推理一次 Qwen3.5-4B | 引入轻量规则裁判优先，LLM 裁判兜底 |
| tool_cache 无 TTL | LRU 只按数量淘汰，可能命中过期结果 | 加 `max_age_seconds` TTL 字段 |
| Planner JSON 输出偶发不稳定 | Qwen3.5-4B 小模型格式遵从能力有限 | 增强后处理正则容错 + 换更大模型 |
| DAG 为静态图 | 规划阶段一次性生成，执行时不可动态调整 | 支持动态节点插入（执行中重新规划） |
