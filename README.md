# HAL1000 Agent 框架 · 综合实训 B 方向

本项目使用 Python 3.10 实现一个本地文件驱动的 Agent 框架。B1–B5 均保留独立命令行入口，使用服务器本地 Qwen3.5-4B。在此基础上，B2 / B3 完成了所有进阶要求，并新增了若干工具函数与配置。

---

## 模块一览

| 模块 | 入口文件 | 职责 |
|---|---|---|
| B1 | `code/b1_agent_runtime.py` | Agent 总控、消息管理、循环控制和产物汇总。 |
| B2 | `code/b2_run_skill.py` + `code/b2_advanced.py` | 基础 5 Skill + 进阶 2 Skill（复合 / 沙箱）的独立运行。 |
| B3 | `code/b3_tool_layer.py` + `code/b3_advanced.py` | 基础 schema 生成 + tool_call 执行；进阶 auto_schema / retry / cache / stats。 |
| B4 | `code/b4_local_agent_llm.py` | 使用 mock 或本地 LLM 生成标准 AIMessage，不执行工具。 |
| B5 | `code/b5_memory.py` | 查找、截断、保存 memory 文档并维护索引。 |
| 完整演示 | `code/run_full_demo.py` | 调用 B1 跑通完整 Agent，并生成汇总报告。 |
| **进阶实现** | `code/auto_schema.py` `code/retry.py` `code/tool_cache.py` `code/tool_stats.py` `code/schema_ablation.py` `code/composite_skill.py` `code/safe_python_exec.py` `code/skills_error_codes.py` | B2/B3 进阶模块 |
| **演示脚本** | `scripts/run_b2_baseline.sh` `scripts/run_b2_advanced.sh` `scripts/run_b3_baseline.sh` `scripts/run_b3_advanced.sh` `scripts/run_b3_ablation.sh` `scripts/run_full_system_demo.sh` `scripts/run_full_demo_skills.sh` `scripts/run_full_demo_pj.sh` `scripts/run_all_demos.sh` | 一键演示脚本 |

B4 的 mock 模式不真实加载、运行模型，作为无 GPU、无模型或模块联调时的调试模式。`prompt_json` 模式则加载本地模型真实运行。

## 1. 环境准备

所有模块统一使用项目根目录下的 `requirements.txt`。推荐每位同学新建自己的 conda 环境，安装步骤如下：

```bash
conda create -n your_env python=3.10 -y
conda activate your_env
export PYTHONNOUSERSITE=1
pip install -r requirements.txt
```

其中 `export PYTHONNOUSERSITE=1` 的作用是：让 Python 启动时禁止加载用户级 site-packages 目录，保证只用当前环境自己的包。

模型使用 Qwen3.5-4B。模型路径在 `configs/model.yaml` 中配置，默认：

```yaml
model:
  model_name_or_path: ${HAL_MODEL_PATH:-../Qwen3.5-4B}
  tokenizer_name_or_path: ${HAL_MODEL_PATH:-../Qwen3.5-4B}
```

三种解析方式（按顺序尝试）：

1. `${HAL_MODEL_PATH}` 环境变量（推荐本地 / Windows 用户）
2. 绝对路径
3. 相对路径（相对 `configs/model.yaml`，即 `<agent root>/Qwen3.5-4B`）

详见 [`docs/PORTABILITY.md`](docs/PORTABILITY.md)。

## 2. SkillResult

B2 与 B3 都基于以下 JSON 结构（来自 `common/schemas.py`）：

```json
{
  "skill_name": "<skill 名称>",
  "status": "success | error",
  "input":  { ... 传入参数回显 ... },
  "output": { ... 成功时输出 ... } | null,
  "error":  null | { "type": "<异常类>", "message": "<异常信息>", "code": "<错误码>" },
  "latency_ms": <耗时，毫秒>
}
```

CLI 退出码：流程错误返回 1；业务错误通过 `status=error` 表达，CLI 仍返回 0。

## 3. AIMessage / ToolMessage

```json
// AIMessage（role=assistant）
{
  "role": "assistant",
  "content": "<final answer or empty>",
  "tool_calls": [
    {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}}
  ]
}

// ToolMessage（role=tool）
{
  "role": "tool",
  "tool_call_id": "call_001",
  "name": "file_reader",
  "content": "<SkillResult 的 JSON 字符串>",
  "status": "success | error"
}
```

---

## 4. B2 — Skill 工具函数模块

### 4.1 基础要求

5 个基础 Skill（`skills/*.py`）：

| Skill | 关键参数 | 返回结构 |
|---|---|---|
| `calculator` | `expression: str` | `{"result": <number>}` |
| `file_reader` | `path: str`, `max_chars: int=2000` | `{"content", "num_chars", "source", "truncated"}` |
| `local_file_search` | `query`, `root_dir`, `file_types`, `top_k` | `{"results": [{"path", "score", "snippet"}]}` |
| `table_analyzer` | `path`, `max_rows_preview`, `describe` | `{"num_rows", "num_columns", "columns", "preview", "describe": {列统计}}` |
| `format_converter` | `text`, `target_format: "markdown"\|"json"`, `output_filename`, `output_dir` | `{"formatted_text", "generated_file_path"}` |

CLI 测试（详见 `scripts/run_b2_baseline.sh`）：

```bash
cd agent/code
python b2_run_skill.py --skill calculator \
    --input ../data/tool_inputs/tool_input_calculator.json \
    --outdir ../outputs/B2_skills/calculator_ok
```

产物：`outputs/B2_skills/<skill>_<ok|err>/<skill>_result.json` + `skill_run_log.jsonl`。

### 4.2 进阶要求

| 进阶项 | 实现位置 |
|---|---|
| 强化 Skill（calculator / file_reader 等加超时 + ErrorCode） | `skills/*.py` + `skills_error_codes.py` |
| 复合 Skill `read_and_convert` | `code/composite_skill.py` + `code/b2_advanced.py` |
| 沙箱代码执行 Skill `safe_python_exec` | `code/safe_python_exec.py` + `code/b2_advanced.py` |
| 完善的错误分类 | `code/skills_error_codes.py`（`ErrorCode` 枚举 9 种） |
| 高耗时 / 高风险 Skill 限制 | `signal.SIGALRM` 3-5s 超时 + 静态黑名单 + 受限 builtins |

CLI：

```bash
python b2_advanced.py --skill read_and_convert \
    --input ../data/tool_inputs/advanced/composite_ok.json \
    --outdir ../outputs/B2_advanced/composite/ok

python b2_advanced.py --skill safe_python_exec \
    --input ../data/tool_inputs/advanced/sandbox_ok.json \
    --outdir ../outputs/B2_advanced/sandbox/ok
```

**B2 报告**：[`reports/B2_report.md`](reports/B2_report.md)。

---

## 5. B3 — 工具说明生成与工具调用模块

### 5.1 基础要求

```bash
# 导出 schema
python b3_tool_layer.py --tools_config ../configs/tools.yaml \
    --toolset basic_tools --export_schema --outdir ../outputs/B3_tools/schema

# 执行 tool_calls（4 种场景）
python b3_tool_layer.py --tools_config ../configs/tools.yaml \
    --toolset basic_tools \
    --tool_calls ../data/messages/ai_message_with_tool_calls.json \
    --execute --outdir ../outputs/B3_tools/with_tool_calls
```

4 种测试场景（对应 4 个 `data/messages/b3_*.json` 与 `data/messages/ai_message_with_tool_calls.json`）：

1. **合法 tool_calls**：正常 Skill 执行。
2. **format_converter_valid**：合法 + output_dir，写出文件。
3. **unknown_tool**：未知工具名 → `status=error` 的 ToolMessage。
4. **missing_required**：缺必填参数 → `status=error` 的 ToolMessage。

### 5.2 进阶要求

| 进阶项 | 实现位置 |
|---|---|
| 自动从 Python 函数生成 schema | `code/auto_schema.py` + `code/b3_advanced.py auto_schema` |
| 可恢复错误重试 | `code/retry.py` + `b3_advanced.py execute --retry N` |
| tool_call 结果缓存 | `code/tool_cache.py` + `b3_advanced.py execute --cache` |
| 工具调用统计 | `code/tool_stats.py` + `b3_advanced.py execute --stats` |
| **schema 描述对工具调用准确率的影响** | `code/schema_ablation.py`（30 mock + 5 prompt_json） |

**B3 报告**：[`reports/B3_report.md`](reports/B3_report.md)。

---

## 6. 一键演示脚本

服务器端：

```bash
cd /root/siton-tmp/HAL1000/agent

# 基础演示（mock 模式，秒级）
bash scripts/run_b2_baseline.sh        # B2 基础 10 用例
bash scripts/run_b2_advanced.sh        # B2 进阶 12 用例
bash scripts/run_b3_baseline.sh        # B3 基础 5 用例
bash scripts/run_b3_advanced.sh        # B3 进阶
bash scripts/run_full_system_demo.sh   # 全系统（mock）

# 进阶实验（含真实模型，分钟级）
bash scripts/run_b3_ablation.sh        # schema A/B 对比（mock 30 + prompt_json 5）
bash scripts/run_full_demo_pj.sh       # 全系统（prompt_json 真模型）

# 一键全跑
bash scripts/run_all_demos.sh          # 全 mock
bash scripts/run_all_demos.sh pj       # 含 prompt_json
```

---

## 7. 全系统演示

全链路：B1 → B5(加载记忆) → B3(生成 schema) → B4(LLM) → B3(execute tool_calls) → B2(Skill 执行) → ToolMessage → B4 → final_answer → B5(保存记忆)。

```bash
bash scripts/run_full_system_demo.sh         # mock
bash scripts/run_full_demo_pj.sh             # prompt_json（真模型）
bash scripts/run_full_demo_skills.sh         # 5 个不同 Skill 任务的 mock 演示
```

输出（每个 `full_demo*` 目录都包含）：

- `messages.json`：完整 5 条消息序列
- `trace.json`：含 turns / tool_rounds / llm_call_count
- `final_answer.md`：最终回答
- `tool_messages.json`：B3 执行的 ToolMessage 数组
- `selected_memory.json` + `saved_memory.json`：B5 记忆操作

---

## 8. 输出目录速查

| 路径 | 含义 |
|---|---|
| `outputs/B2_skills/` | B2 个人演示基础产物（5 Skill × 正常+异常 = 10 个 SkillResult） |
| `outputs/B2_advanced/` | B2 个人演示进阶产物（12 个用例 + ErrorCode） |
| `outputs/B3_tools/` | B3 个人演示基础产物（5 个场景） |
| `outputs/B3_advanced/` | B3 个人演示进阶产物（auto_schema + cache + retry + stats） |
| `outputs/B3_ablation/` | B3 schema A/B 对比产物（mock 30 + prompt_json 5） |
| `outputs/full_demo/` | 全系统演示（mock） |
| `outputs/full_demo_prompt_json/` | 全系统演示（prompt_json 真模型） |
| `outputs/full_demo_{calc,search,table,format}/` | 各 Skill 的全系统 mock 演示 |
| `memory/conversations/`, `memory/global/` | B5 记忆文档 |
| `reports/B2_report.md`, `reports/B3_report.md` | 报告 |
| `docs/PORTABILITY.md` | 可移植性指南 |

---

## 9. 关于 SSH 连接

服务器要求 `ssh -p 20021`，Windows OpenSSH 不支持交互式密码登录。本项目作者使用 paramiko 封装 `ssh_run.py`：

```bash
# 任意命令
python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py "echo hello"

# 传 stdin 命令（避免 PowerShell 引号转义）
type cmd.txt | python E:\...\ssh_run.py pipe

# 文件传输
python E:\...\ssh_run.py put <local> <remote>
python E:\...\ssh_run.py get <remote> <local>
python E:\...\ssh_run.py get-tree <remote_dir> <local_dir>
```

详见 `docs/PORTABILITY.md` § 4。

---

## 10. 测试样例

每个模块都给出至少一个正常 + 异常输入样例：

| 模块 | 输入样例目录 | 异常样例 |
|---|---|---|
| B2 | `data/tool_inputs/tool_input_<skill>.json` | `*_error.json` |
| B2 advanced | `data/tool_inputs/advanced/` | composite_err / sandbox_blocked / sandbox_timeout / calc_overflow |
| B3 | `data/messages/ai_message_with_tool_calls.json` + 3 个 `b3_*.json` | unknown_tool / missing_required |
| B3 ablation | `code/schema_ablation.py` 内置 30 条 | — |

---

## 11. 已知限制

- **沙箱不是真正的隔离**：`safe_python_exec` 通过静态黑名单 + 受限 builtins + 超时实现，适合短表达式，不能替代 `RestrictedPython` / WASM 沙箱。
- **retry 仅对白名单错误重试**：参数错误（INVALID_INPUT / UNSUPPORTED_TYPE）不重试，避免死循环。
- **tool_cache 仅 LRU 256 项，无 TTL**：demo 足够；生产环境应加 `max_age_seconds`。
- **schema_ablation 只跑了 5 条 prompt_json**：受 GPU 时间限制；如需严格结论需扩到 50+ 条 + bootstrap 置信区间。
- **mock 模式 `_mock_generate` 只对"messages 含 tool role"返回 file_reader 调用**：因此 mock 的 ablation 实验对所有 schema 完全相同（详细差异只能用真模型观察）。

---

## 12. 参考

- 实训 PPT 与说明文档（来自教务）
- Qwen3.5-4B 模型卡：`https://www.modelscope.cn/models/Qwen/Qwen3.5-4B`
- ToolLLM：`https://arxiv.org/abs/2307.16789`
- StateFlow：`https://arxiv.org/abs/2403.11322`
- HuggingGPT：`https://arxiv.org/abs/2303.17580`
- ReAct：`https://arxiv.org/abs/2210.03629`
- MemGPT：`https://arxiv.org/abs/2310.08560`