# B2 — Skill 工具函数模块 实训报告

> 东北大学 · 计算机科学与工程学院 · 人工智能专业 · 综合实训（B 方向 Agent 智能体）
> 作者：杨贺淳（演示模型：Qwen3.5-4B）
> 报告日期：2026-07-03
> 运行环境：服务器 `/root/siton-tmp/HAL1000/agent/`，`/opt/conda/envs/hal`（Python 3.10.20 + PyTorch 2.7.1 + cu118 + Transformers 5.12.1 + accelerate + safetensors）

---

## 一、实训任务概述

### 1.1 B2 在 Agent 系统中的定位

```
┌──────────────────────────────────────────────────────┐
│ B1 Agent Runtime  ←  本次报告焦点  → B3 Tool Layer   │
│        ↓                                  ↑           │
│        ↓         ┌────────────────┐       ↑           │
│        ↓ 调用    │  B2 Skill 层   │ ←──┘ 由 B3 调用  │
│        ↓         │  (本报告)      │                   │
│        ↓         └────────────────┘                   │
│        ↓             ↑  ↑  ↑                         │
│        ↓             │  │  │                         │
│       全部 Skill 函数返回统一的 SkillResult           │
└──────────────────────────────────────────────────────┘
```

**B2 的核心职责**：

1. 把 Agent 能调用的工具以"Skill 函数"形式实现，每个 Skill 都是一个可独立调用的 Python 函数。
2. 接收 B3 传入的参数 → 执行 → 返回标准化的 JSON（`SkillResult`）。
3. 每个 Skill 必须能被 B3 正确加载、执行，并通过 CLI 单独测试。

**SkillResult 统一结构**（来自 `common/schemas.py`）：

```json
{
  "skill_name": "<skill 名称>",
  "status": "success | error",
  "input":  { ... 传入参数的回显 ... },
  "output": { ... 成功时的输出对象 ... } | null,
  "error":  null | { "type": "<异常类>", "message": "<异常信息>", "code": "<错误码>" },
  "latency_ms": <耗时，毫秒>
}
```

CLI 入口：`python b2_run_skill.py --skill <name> --input <json> --outdir <dir>`，退出码为 0 表示 CLI 流程正常，业务错误通过 `status=error` 表达，不影响 CLI 退出。

### 1.2 基础要求 vs 进阶要求

| 类别 | 要求（节选自 PPT Slide 16、20） | 完成情况 |
|---|---|---|
| **基础** | 实现 ≥ 5 个基础 Skill | ✅ 5 个：`calculator / file_reader / local_file_search / table_analyzer / format_converter` |
| **基础** | 每个 Skill 有明确的参数、返回值、描述，可 JSON 序列化 | ✅ 全部满足 |
| **基础** | 支持独立 CLI 测试 + 正常/异常各一 | ✅ 10 个用例（5 正常 + 5 异常） |
| **基础** | Skill 能被 B3 加载、执行、写日志 | ✅ 见 B3 报告 |
| **基础** | 解释 Skill 与 Tool Schema 的区别 | ✅ 见 § 6 |
| **进阶** | 增强现有 Skill | ✅ calculator 强化安全 + 错误码；file_reader/local_file_search/table_analyzer 加 3-5s 超时 |
| **进阶** | 新增 Skill（沙箱代码执行，需限制） | ✅ `safe_python_exec`（禁止 import/exec/open 等 + 5s 超时） |
| **进阶** | 复合 Skill | ✅ `read_and_convert`（file_reader → format_converter 链式调用） |
| **进阶** | 更完善的错误分类 | ✅ `ErrorCode` 枚举（INVALID_INPUT / FILE_NOT_FOUND / OVERFLOW / UNSUPPORTED_TYPE / EXECUTION_TIMEOUT / PERMISSION_DENIED / PATH_ESCAPE / DIVISION_BY_ZERO / INTERNAL） |
| **进阶** | 高耗时 / 高风险 Skill 限制 | ✅ calculator exponent ≤ 30、file_reader/table_analyzer/local_file_search 加 SIGALRM 超时，safe_python_exec 加 `_time_limit` |

---

## 二、开发与运行环境

### 2.1 服务器端

| 资源 | 实测值 |
|---|---|
| Conda 环境 | `/opt/conda/envs/hal` |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu118（CUDA available, device count = 1） |
| Transformers | 5.12.1 |
| GPU | NVIDIA H200 NVL，23.5 GB（`nvidia-smi` 实测空闲） |
| 内存 | 1 TB total，936 GB available |
| 磁盘 | 175 GB available |
| 模型 | `/root/siton-tmp/HAL1000/Qwen3.5-4B/`（safetensors，2 个分片共约 8.3 GB） |

### 2.2 代码部署位置（服务器）

```
/root/siton-tmp/HAL1000/agent/
├── code/
│   ├── b2_run_skill.py           # B2 基础 CLI 入口（已加 ErrorCode 输出）
│   ├── b2_advanced.py            # B2 进阶 CLI 入口（处理复合 + 沙箱）
│   ├── composite_skill.py        # 进阶：复合 Skill read_and_convert
│   ├── safe_python_exec.py       # 进阶：沙箱 Skill
│   ├── skills_error_codes.py     # 进阶：ErrorCode 体系
│   ├── common/path_utils.py      # 已加 resolve_model_path 等
│   └── common/schemas.py         # SkillResult / AIMessage / ToolMessage 工厂
├── skills/                       # 基础 5 Skill + skills_error_codes.py
│   ├── calculator.py
│   ├── file_reader.py
│   ├── local_file_search.py
│   ├── table_analyzer.py
│   ├── format_converter.py
│   └── __init__.py               # 提供 resolve_data_path 安全解析
├── configs/tools.yaml            # B3 读取的工具定义
├── data/                         # 所有输入样例
├── outputs/B2_skills/            # 个人演示产物（基础）
├── outputs/B2_advanced/          # 个人演示产物（进阶）
└── outputs/full_demo*/           # 全系统演示产物
```

### 2.3 本地开发环境

- Windows 11，Python 3.x
- 通过 paramiko 封装的 SSH 通道（`E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py`）连接服务器 `202.199.13.141:20021`，绕开 Windows OpenSSH 无法交互登录的问题
- 本地工作目录：`E:\MyCode\Python\PycharmProjects\HAL1000\`（即 `assignment_B` 的同级目录）

---

## 三、基础要求实现

### 3.1 五个基础 Skill

| Skill | 模块路径 | 输入参数 | 返回结构 | 异常场景 |
|---|---|---|---|---|
| `calculator` | `skills/calculator.py` | `expression: str` | `{"result": <number>}` | 非算术元素 / 指数 > 30 / 结果溢出 |
| `file_reader` | `skills/file_reader.py` | `path: str`, `max_chars: int = 2000` | `{"content": str, "num_chars": int, "source": str, "truncated": bool}` | 路径越界 / 后缀不支持 / 文件不存在 |
| `local_file_search` | `skills/local_file_search.py` | `query: str`, `root_dir: str`, `file_types: list[str]`, `top_k: int` | `{"results": [{"path", "score", "snippet"}, ...]}` | 目录不存在 / 类型不在白名单 |
| `table_analyzer` | `skills/table_analyzer.py` | `path: str`, `max_rows_preview: int`, `describe: bool` | `{"path", "num_rows", "num_columns", "columns", "preview", "describe": {列统计}}` | 非 CSV/TSV / 文件不存在 / 表无表头 |
| `format_converter` | `skills/format_converter.py` | `text: str`, `target_format: "markdown"\|"json"`, `output_filename?: str`, `output_dir?: str` | `{"formatted_text": str, "generated_file_path": str}` | target_format 不支持 / 输入空 |

所有 Skill 接受 `data_root` keyword 参数（仅当需要从外部 data 根目录注入时使用），文件读取/搜索/表格类 Skill 都通过 `skills.resolve_data_path` 防御路径越界（防止 `../../etc/passwd` 这类攻击）。

### 3.2 基础 CLI 测试产物

执行命令（`scripts/run_b2_baseline.sh`）：

```bash
PY=/opt/conda/envs/hal/bin/python
CODE=/root/siton-tmp/HAL1000/agent/code
INPUTS=/root/siton-tmp/HAL1000/agent/data/tool_inputs
OUT=/root/siton-tmp/HAL1000/agent/outputs/B2_skills

run_skill() {
  local skill="$1" input="$2" label="$3"
  local outdir="$OUT/${skill}_${label}"
  "$PY" "$CODE/b2_run_skill.py" --skill "$skill" --input "$input" --outdir "$outdir"
}

run_skill calculator         "$INPUTS/tool_input_calculator.json"          ok
run_skill calculator         "$INPUTS/tool_input_calculator_error.json"    err
run_skill file_reader        "$INPUTS/tool_input_file_reader.json"         ok
run_skill file_reader        "$INPUTS/tool_input_file_reader_error.json"   err
run_skill local_file_search  "$INPUTS/tool_input_file_search.json"         ok
run_skill local_file_search  "$INPUTS/tool_input_file_search_error.json"   err
run_skill table_analyzer     "$INPUTS/tool_input_table_analyzer.json"      ok
run_skill table_analyzer     "$INPUTS/tool_input_table_analyzer_error.json" err
run_skill format_converter   "$INPUTS/tool_input_format_converter.json"    ok
run_skill format_converter   "$INPUTS/tool_input_format_converter_error.json" err
```

### 3.3 基础测试结果汇总

> 路径前缀：`results_B2_B3/outputs/B2_skills/`

| Skill | Label | Status | Latency (ms) | 产物文件 |
|---|---|---|---|---|
| calculator | ok | success | 0.037 | `calculator_ok/calculator_result.json` |
| calculator | err | error | 0.029 | `calculator_err/calculator_result.json` |
| file_reader | ok | success | 0.166 | `file_reader_ok/file_reader_result.json` |
| file_reader | err | error | 0.140 | `file_reader_err/file_reader_result.json` |
| local_file_search | ok | success | 0.458 | `local_file_search_ok/local_file_search_result.json` |
| local_file_search | err | error | 0.137 | `local_file_search_err/local_file_search_result.json` |
| table_analyzer | ok | success | 0.256 | `table_analyzer_ok/table_analyzer_result.json` |
| table_analyzer | err | error | 0.170 | `table_analyzer_err/table_analyzer_result.json` |
| format_converter | ok | success | 0.171 | `format_converter_ok/format_converter_result.json` |
| format_converter | err | error | 0.005 | `format_converter_err/format_converter_result.json` |

### 3.4 正常样例产物示例

<details>
<summary>📄 calculator_ok/calculator_result.json</summary>

```json
{
  "skill_name": "calculator",
  "status": "success",
  "input": { "expression": "23 * 17 + 9" },
  "output": { "result": 400 },
  "error": null,
  "latency_ms": 0.037
}
```
</details>

<details>
<summary>📄 file_reader_ok/file_reader_result.json</summary>

```json
{
  "skill_name": "file_reader",
  "status": "success",
  "input": { "path": "docs/agent_intro.txt", "max_chars": 2000 },
  "output": {
    "content": "Agent 系统通常由模型、工具、记忆和执行循环组成。\n工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答。\nMemory 为 Agent 提供全局知识和历史对话上下文。\n",
    "num_chars": 92,
    "source": "docs/agent_intro.txt",
    "truncated": false
  },
  "error": null,
  "latency_ms": 0.166
}
```
</details>

### 3.5 异常样例产物示例

<details>
<summary>📄 calculator_err/calculator_result.json（拒绝 <code>__import__('os')</code>）</summary>

```json
{
  "skill_name": "calculator",
  "status": "error",
  "input": { "expression": "__import__('os')" },
  "output": null,
  "error": {
    "type": "ValueError",
    "message": "unsupported expression element: Call"
  },
  "latency_ms": 0.029
}
```
</details>

<details>
<summary>📄 file_reader_err/file_reader_result.json（文件不存在）</summary>

```json
{
  "skill_name": "file_reader",
  "status": "error",
  "input": { "path": "docs/missing.txt", "max_chars": 2000 },
  "output": null,
  "error": {
    "type": "FileNotFoundError",
    "message": "file not found: docs/missing.txt"
  },
  "latency_ms": 0.14
}
```
</details>

> CLI 在上述业务异常时**仍然返回 0**，业务异常以 `status=error` 表达，这是 B2 设计的核心约定：CLI 流程 vs 业务结果分离。

### 3.6 Skill 与 Tool Schema 的区别（必答基础题）

> 见 PPT Slide 16："能够说明 Skill 和 Tool Schema 的区别"

| 维度 | Skill（Python 函数） | Tool Schema（JSON） |
|---|---|---|
| 本质 | 可执行的 Python 函数 | 描述 Skill 接口的 JSON 文档 |
| 消费者 | B2（执行）、B3（间接调用） | B3（生成）+ B4（注入到 LLM Prompt） |
| 形式 | `def calculator(expression: str) -> dict: ...` | OpenAI function schema：`{"type":"function","function":{"name":"calculator","description":"...","parameters":{...}}}` |
| 内容 | 函数体本身 | 函数签名 + 描述 + 参数 JSON Schema |
| 关系 | Skill 是**实现**，Schema 是**对外描述** | Schema 是 LLM 看得到的东西 |

`b3_tool_layer.py` 的核心工作就是把 `tools.yaml` → 工具描述 → 函数对象 一一对应起来。

---

## 四、进阶要求实现

### 4.1 进阶模块全景

```
code/
├── b2_advanced.py            # CLI 入口，处理 read_and_convert / safe_python_exec
├── composite_skill.py        # 进阶1：复合 Skill（链式调用）
├── safe_python_exec.py       # 进阶2：沙箱 Skill（限制 + 超时）
└── skills_error_codes.py     # 进阶3：ErrorCode 体系（被基础 Skill + 进阶 Skill 共用）
```

### 4.2 进阶 1 — 错误分类体系（ErrorCode）

**目标**：把每个 Skill 抛出的 Python 异常映射到稳定的错误码枚举，方便上层（B1 / 业务侧）做差异化处理（如自动重试、用户提示、监控告警）。

**实现**（`skills_error_codes.py`）：

```python
class ErrorCode(str, Enum):
    INVALID_INPUT      = "INVALID_INPUT"
    PATH_ESCAPE        = "PATH_ESCAPE"
    FILE_NOT_FOUND     = "FILE_NOT_FOUND"
    UNSUPPORTED_TYPE   = "UNSUPPORTED_TYPE"
    OVERFLOW           = "OVERFLOW"
    DIVISION_BY_ZERO   = "DIVISION_BY_ZERO"
    EXECUTION_TIMEOUT  = "EXECUTION_TIMEOUT"
    PERMISSION_DENIED  = "PERMISSION_DENIED"
    INTERNAL           = "INTERNAL"

def classify_exception(exc: BaseException) -> ErrorCode: ...
def attach_error_code(exc: BaseException, code: ErrorCode) -> BaseException: ...
def enrich_error_payload(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "code":   classify_exception(exc).value,
    }
```

**使用**：在每个 Skill 的异常分支中用 `attach_error_code(exc, ErrorCode.<X>)`，`b2_run_skill.py` 把异常交给 `enrich_error_payload` 转成结构化 JSON。

### 4.3 进阶 2 — 高耗时 Skill 加超时限制

| Skill | 超时阈值 | 实现 |
|---|---|---|
| `file_reader` | 3.0 s | `signal.SIGALRM` + `signal.setitimer(ITIMER_REAL)` |
| `local_file_search` | 5.0 s | 同上 |
| `table_analyzer` | 5.0 s | 同上 |
| `safe_python_exec` | 5.0 s | 同上 |
| `calculator` | 不需要（AST 解析 + 算术运算 O(1)） | — |

所有超时由 `_time_limit(seconds)` 上下文管理器实现，超时时抛 `TimeoutError`，被 `attach_error_code` 标为 `EXECUTION_TIMEOUT`。

### 4.4 进阶 3 — 复合 Skill：`read_and_convert`

**定义**：把"读取文件"和"格式转换"两个基础 Skill 串联起来，对外暴露为一个 Skill，节省 Agent 的 LLM 轮次。

**实现**（`composite_skill.py`）：

```python
def read_and_convert(
    path: str,
    target_format: str = "markdown",
    max_chars: int = 2000,
    output_filename: str | None = None,
    output_dir: str | None = None,
    *,
    data_root: str | None = None,
) -> dict:
    reader_result = file_reader(path=path, max_chars=max_chars, data_root=data_root)
    raw_text = reader_result["content"]
    converter_result = format_converter(
        text=raw_text,
        target_format=target_format,
        output_filename=output_filename,
        output_dir=output_dir,
    )
    return {
        "read": { "source", "num_chars", "truncated" },
        "convert": { "target_format", "formatted_text", "generated_file_path" },
        "pipeline": ["file_reader", "format_converter"],
    }
```

**运行**：

```bash
python b2_advanced.py --skill read_and_convert \
    --input ../data/tool_inputs/advanced/composite_ok.json \
    --outdir ../outputs/B2_advanced/composite/ok
```

**产物**：`outputs/B2_advanced/composite/ok/read_and_convert_result.json`（含 `read + convert + pipeline` 三段）

```json
{
  "skill_name": "read_and_convert",
  "status": "success",
  "input": {
    "path": "docs/agent_intro.txt",
    "target_format": "markdown",
    "max_chars": 2000,
    "output_filename": "agent_intro_bullets.md"
  },
  "output": {
    "read":   { "source": "docs/agent_intro.txt", "num_chars": 92, "truncated": false },
    "convert": {
      "target_format": "markdown",
      "formatted_text": "- Agent 系统通常由模型、工具、记忆和执行循环组成。\n- 工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答。\n- Memory 为 Agent 提供全局知识和历史对话上下文。",
      "generated_file_path": "/root/siton-tmp/HAL1000/agent/outputs/B2_advanced/composite/ok/agent_intro_bullets.md"
    },
    "pipeline": ["file_reader", "format_converter"]
  },
  "latency_ms": 1.758
}
```

异常用例（`composite_err`，文件不存在）：

```json
{
  "skill_name": "read_and_convert",
  "status": "error",
  "input": { "path": "docs/does_not_exist.txt", "target_format": "markdown" },
  "output": null,
  "error": { "type": "FileNotFoundError", "message": "file not found: docs/does_not_exist.txt", "code": "FILE_NOT_FOUND" },
  "latency_ms": 1.55
}
```

### 4.5 进阶 4 — 沙箱 Skill：`safe_python_exec`

**动机**：Agent 在某些场景下需要让 LLM 跑一段 Python 表达式（计算、格式化等），但必须把风险关在沙箱里。

**三层防护**：

| 层次 | 机制 | 作用 |
|---|---|---|
| 静态检查 | `_validate_source` 黑名单：`import` / `__import__` / `open(` / `exec(` / `eval(` / `compile(` / `globals(` / `locals(` / `subprocess` / `os.` / `sys.` / `shutil` / `socket` / `urllib` / `requests` / `http` | 在 `exec` 之前拦截 |
| 运行时空集 | 仅暴露安全 builtins（`abs / all / any / bool / dict / enumerate / filter / int / len / list / max / min / print / range / round / set / sorted / str / sum / tuple / type / zip` 等）+ `math` + `statistics` | 即使绕过静态检查也无法访问文件 / 网络 |
| 运行时超时 | `signal.SIGALRM` 5 秒 | 即使死循环也不会无限挂起 |

**运行示例**：

```bash
python b2_advanced.py --skill safe_python_exec --input sandbox_ok.json --outdir sandbox/ok
python b2_advanced.py --skill safe_python_exec --input sandbox_blocked.json --outdir sandbox/blocked
python b2_advanced.py --skill safe_python_exec --input sandbox_timeout.json --outdir sandbox/timeout
```

| 用例 | 输入 | 期望 | 实际 |
|---|---|---|---|
| sandbox/ok | `"source": "result = sum(range(10))"` | success，output="45" | ✅ `{ "status": "success", "output": "45", "result": 45 }` |
| sandbox/blocked | `"source": "import os; os.system('echo pwned')"` | error，code=INVALID_INPUT | ✅ `{ "type": "ValueError", "message": "forbidden token detected: 'import'", "code": "INVALID_INPUT" }` |
| sandbox/timeout | `"source": "while True: pass"` | error，code=EXECUTION_TIMEOUT | ✅ `{ "type": "SandboxTimeout", "message": "execution exceeded 5.0s timeout", "code": "EXECUTION_TIMEOUT" }`，latency ≈ 5000ms |

> 沙箱是**有意保守**的（不替代真正的容器 / WASM 沙箱），适合 Agent 临时跑用户提供的短表达式。生产环境应该进一步用 `RestrictedPython` / `wasmtime` 等替代。

### 4.6 进阶 5 — 强化 calculator（错误码 + overflow）

为 calculator 引入 `OVERFLOW` 错误码，并把 exponent 上限从 12 提到 30 后用 `OVERFLOW` 替代旧版过于狭窄的 `INVALID_INPUT`：

输入 `"2 ** 1000"`：

```json
{
  "skill_name": "calculator",
  "status": "error",
  "input": { "expression": "2 ** 1000" },
  "output": null,
  "error": {
    "type": "ValueError",
    "message": "exponent magnitude must not exceed 30",
    "code": "OVERFLOW"
  },
  "latency_ms": 0.037
}
```

输入 `"import os"`（旧版会拒绝但归类到 `UNSUPPORTED_TYPE`）：

```json
{
  "skill_name": "calculator",
  "status": "error",
  "input": { "expression": "import os" },
  "output": null,
  "error": {
    "type": "ValueError",
    "message": "invalid arithmetic expression",
    "code": "INVALID_INPUT"
  }
}
```

### 4.7 进阶测试总览（来自 `outputs/B2_advanced/`）

> 路径前缀：`results_B2_B3/outputs/B2_advanced/`

| 子目录 | 用例 | Status | Code | Latency (ms) |
|---|---|---|---|---|
| `baseline_error_codes/calculator` | 基础 calculator 错误样例（__import__） | error | UNSUPPORTED_TYPE | 0.040 |
| `baseline_error_codes/file_reader` | 路径不存在 | error | FILE_NOT_FOUND | 0.155 |
| `baseline_error_codes/table_analyzer` | 路径不存在 | error | FILE_NOT_FOUND | 0.171 |
| `baseline_error_codes/local_file_search` | 目录不存在 | error | FILE_NOT_FOUND | 0.144 |
| `baseline_error_codes/format_converter` | target_format 非法 | error | UNSUPPORTED_TYPE | 0.010 |
| `calculator_edge/overflow` | `2 ** 1000` | error | OVERFLOW | 0.037 |
| `calculator_edge/unsupported` | `import os` | error | INVALID_INPUT | 0.023 |
| `composite/ok` | `read_and_convert` 成功 | success | — | 1.758 |
| `composite/err` | `read_and_convert` 路径不存在 | error | FILE_NOT_FOUND | 1.55 |
| `sandbox/ok` | `sum(range(10))` | success | — | 0.087 |
| `sandbox/blocked` | `import os` | error | INVALID_INPUT | 0.025 |
| `sandbox/timeout` | `while True: pass` | error | EXECUTION_TIMEOUT | 5000.137 |

---

## 五、个人演示

### 5.1 演示流程（推荐现场操作顺序）

1. **进入代码目录**：`cd /root/siton-tmp/HAL1000/agent/code`
2. **激活 hal 环境**：`source /opt/conda/etc/profile.d/conda.sh && conda activate hal`（或直接 `/opt/conda/envs/hal/bin/python`）
3. **逐个跑 5 个基础 Skill 的正常样例**：
   ```bash
   python b2_run_skill.py --skill calculator \
       --input ../data/tool_inputs/tool_input_calculator.json \
       --outdir /tmp/demo/calculator_ok
   cat /tmp/demo/calculator_ok/calculator_result.json
   ```
4. **逐个跑 5 个基础 Skill 的异常样例**（同样 CLI 退出码 0，status=error）。
5. **跑进阶**：复合 Skill + 沙箱（4 个用例：ok / blocked / timeout + calculator overflow）。

### 5.2 演示话术要点

- "Skill 是真实可执行的 Python 函数；Tool Schema 是描述 Skill 接口的 JSON，让 LLM 看得懂。"
- "错误返回结构化：type/message/code 三段，code 来自 `ErrorCode` 枚举，方便上层做差异化处理。"
- "复合 Skill 把多次 LLM 调用合并成 1 次，节省 token + 减少错误传播。"
- "沙箱分三层：静态黑名单 → 受限 builtins → 超时，即使 LLM 输出恶意代码也跑不出去。"

### 5.3 关键产物清单（个人演示要展示的）

```
/root/siton-tmp/HAL1000/agent/outputs/B2_skills/                 # 基础 10 个用例
/root/siton-tmp/HAL1000/agent/outputs/B2_skills/skill_run_log.jsonl  # 汇总运行日志
/root/siton-tmp/HAL1000/agent/outputs/B2_advanced/                # 进阶 12 个用例
```

---

## 六、全系统演示（B1 + B2 + B3 + B4 + B5 联动）

> B2 Skill 在全系统演示中由 B3 间接调用。完整链路见 B3 报告 § 6，本节只说明 B2 的位置与产物。

### 6.1 全系统链路中 B2 的角色

```
B1 ─→ B5 (load memory) ─→ B3 (generate tools_schema) ─→ B4 (LLM think)
  ↑                                                           │
  └────────── B3 (execute tool_calls) ─→ B2 (Skill run) ─→ ToolMessage ─→ B4 ─→ B5 (save memory)
```

`outputs/full_demo/` 目录展示了一条完整的 message 流，message[3] 是 `tool` 角色，它的 `content` 字段是 B2 Skill 的 SkillResult JSON 字符串：

```json
{
  "role": "tool",
  "tool_call_id": "call_001",
  "name": "file_reader",
  "content": "{\"skill_name\":\"file_reader\",\"status\":\"success\",\"input\":{\"path\":\"docs/agent_intro.txt\",\"max_chars\":2000},\"output\":{\"content\":\"Agent 系统通常由模型、工具、记忆和执行循环组成。\\n...\",\"num_chars\":92,\"source\":\"docs/agent_intro.txt\",\"truncated\":false},\"error\":null,\"latency_ms\":1.843}",
  "status": "success"
}
```

### 6.2 全系统演示产物

| 输出目录 | 含义 |
|---|---|
| `outputs/full_demo/` | mock 模式 + file_reader 任务（基础链路） |
| `outputs/full_demo_prompt_json/` | prompt_json 模式 + 真实 Qwen3.5-4B + file_reader 任务 |
| `outputs/full_demo_calc/` | mock 模式 + calculator 任务 |
| `outputs/full_demo_search/` | mock 模式 + local_file_search 任务 |
| `outputs/full_demo_table/` | mock 模式 + table_analyzer 任务 |
| `outputs/full_demo_format/` | mock 模式 + format_converter 任务 |

每个 `full_demo_*` 目录都包含完整的 `messages.json / trace.json / final_answer.md / tool_messages.json / tools_schema.json / selected_memory.json / saved_memory.json` 等产物，证明 B2 Skill 在真实 Agent 链路中能正常工作。

### 6.3 全系统演示中 B2 的实际表现

`outputs/full_demo_prompt_json/final_answer.md`（真实模型输出）：

```markdown
1. Agent 系统由模型、工具、记忆和执行循环四个核心部分组成。
2. 工具调用使模型能够读取本地文件、执行计算等操作，并将结果用于后续回答。
3. Memory 为 Agent 提供全局知识和历史对话上下文，支持持续对话。
```

链路追溯：

- LLM 调用 1（turn_index=1）：模型选 `file_reader` 工具，调 B2 后 ToolMessage 含 `agent_intro.txt` 的真实文本
- LLM 调用 2（turn_index=2）：模型基于 ToolMessage 总结 3 条要点
- B5 把这条对话作为 `mem_conversation_conv_001` 写入 `memory/conversations/conv_001.md`，更新 `memory_index.json`

---

## 七、可移植性

### 7.1 三档路径解析（`common/path_utils.py`）

| 输入形态 | 示例 | 解析顺序 |
|---|---|---|
| 环境变量占位符 | `${HAL_MODEL_PATH:-/data/Qwen3.5-4B}` | 1. 读 `HAL_MODEL_PATH` 环境变量；2. 用 `:-(default)`；3. 解析为绝对路径 |
| 相对路径 | `../Qwen3.5-4B` | 1. 相对 `configs/model.yaml` 解析；2. 检查存在 |
| 绝对路径 | `/root/.../Qwen3.5-4B` | 1. 直接使用；2. 检查存在 |

如果全部失败，抛 `FileNotFoundError` 并列出已搜索的位置。

### 7.2 三平台运行步骤

| 平台 | 命令 |
|---|---|
| Linux（服务器，hal 环境） | `conda activate hal && python b2_run_skill.py --skill calculator --input ../data/tool_inputs/tool_input_calculator.json --outdir /tmp/out` |
| macOS（conda） | 同 Linux；若模型放家目录：`export HAL_MODEL_PATH=/Users/you/Qwen3.5-4B` |
| Windows（PowerShell） | 先 `set HAL_MODEL_PATH=E:\models\Qwen3.5-4B`，再 `python b2_run_skill.py --skill calculator --input ..\data\tool_inputs\tool_input_calculator.json --outdir .\out` |

详见 `docs/PORTABILITY.md`。

---

## 八、关键源码

### 8.1 `skills/calculator.py`（节选）

```python
_BINARY_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}

def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise attach_error_code(ValueError("expression must be a non-empty string"), ErrorCode.INVALID_INPUT)
    if len(expression) > 200:
        raise attach_error_code(ValueError("expression is too long"), ErrorCode.INVALID_INPUT)
    tree = ast.parse(expression, mode="eval")
    return {"result": _evaluate(tree)}
```

特点：

- 用 `ast.parse` 而不是 `eval`，天然拒绝函数调用 / 属性访问
- `BinOp` 白名单只接受基础运算
- 指数上限 30 + 结果绝对值上限 1e100 防溢出

### 8.2 `safe_python_exec.py`（核心）

```python
@contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise SandboxTimeout(f"execution exceeded {seconds:.1f}s timeout")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def safe_python_exec(source: str, timeout_seconds: float = 5.0, allow_return: bool = True) -> dict:
    _validate_source(source)   # 黑名单
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS, **_SAFE_MODULES}
    try:
        with _time_limit(timeout_seconds):
            exec(compile(source, "<sandbox>", "exec"), sandbox_globals, sandbox_locals)
    except SandboxTimeout as exc:
        return {"status": "timeout", "error": str(exc), "result": None}
    ...
```

### 8.3 `composite_skill.py`（核心）

```python
def read_and_convert(path, target_format="markdown", max_chars=2000, ...):
    reader_result = file_reader(path=path, max_chars=max_chars, data_root=data_root)
    converter_result = format_converter(
        text=reader_result["content"], target_format=target_format,
        output_filename=output_filename, output_dir=output_dir,
    )
    return {
        "read": { "source": ..., "num_chars": ..., "truncated": ... },
        "convert": { "target_format": ..., "formatted_text": ..., "generated_file_path": ... },
        "pipeline": ["file_reader", "format_converter"],
    }
```

### 8.4 `skills_error_codes.py`（核心）

```python
class ErrorCode(str, Enum):
    INVALID_INPUT      = "INVALID_INPUT"
    PATH_ESCAPE        = "PATH_ESCAPE"
    FILE_NOT_FOUND     = "FILE_NOT_FOUND"
    UNSUPPORTED_TYPE   = "UNSUPPORTED_TYPE"
    OVERFLOW           = "OVERFLOW"
    EXECUTION_TIMEOUT  = "EXECUTION_TIMEOUT"
    PERMISSION_DENIED  = "PERMISSION_DENIED"
    INTERNAL           = "INTERNAL"

def enrich_error_payload(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "code": classify_exception(exc).value}
```

---

## 九、问题与改进

### 9.1 已发现的问题

| 问题 | 临时方案 | 长期改进 |
|---|---|---|
| sandbox 的 builtins 黑名单仍是宽松的（Python 3.10 内置类型都可用） | 限制 source 长度（≤1000 字符）+ 超时 5s | 集成 `RestrictedPython` 或改用 WASM 沙箱 |
| calculator 的 exponent 上限 30 是经验值，不是数学严格上界 | 抛 OVERFLOW 让用户改用 `decimal` | 增加 `decimal` / `fractions` 支持 |
| B2 进阶复合 Skill 与基础 Skill 都暴露同一接口但 skill_name 不同 | 在 B3 tools.yaml 注册复合 Skill 时显式声明 | 提供自动注册工具 |
| Skill 内部未做并发安全（同一进程多 Skill 并行调用时 `_time_limit` 会互相覆盖 signal） | 单进程串行调用即可 | 用 `threading.Timer` 或独立子进程代替 signal |

### 9.2 后续可拓展方向

1. **Skill 注册中心**：把 Skill 从 Python 模块改为可插拔的 entry-point，工具函数可以独立打包发布。
2. **Skill 结果回放**：把每次 Skill 调用的 `(input, output, error, latency)` 写入 ClickHouse / DuckDB，支持 A/B 实验和回放。
3. **Skill 依赖图**：复合 Skill 显式声明依赖，构成 DAG，便于做依赖分析 / 并行调度。
4. **异步 Skill**：当前 Skill 都是同步的；如果有 IO 密集型 Skill（如远程 API）应改为 async + asyncio.gather 并行。

---

## 十、附录

### 10.1 B2 相关产出文件清单

```
/root/siton-tmp/HAL1000/agent/outputs/B2_skills/
├── calculator_ok/calculator_result.json
├── calculator_err/calculator_result.json
├── file_reader_ok/file_reader_result.json
├── file_reader_err/file_reader_result.json
├── local_file_search_ok/local_file_search_result.json
├── local_file_search_err/local_file_search_result.json
├── table_analyzer_ok/table_analyzer_result.json
├── table_analyzer_err/table_analyzer_result.json
├── format_converter_ok/format_converter_result.json
├── format_converter_err/format_converter_result.json
└── skill_run_log.jsonl                 # 10 条 Skill 调用记录

/root/siton-tmp/HAL1000/agent/outputs/B2_advanced/
├── baseline_error_codes/{calculator,file_reader,local_file_search,table_analyzer,format_converter}/...
├── calculator_edge/{overflow,unsupported}/calculator_result.json
├── composite/{ok,err}/read_and_convert_result.json
└── sandbox/{ok,blocked,timeout}/safe_python_exec_result.json
```

### 10.2 B2 相关源代码（位于 `code/` 和 `skills/`）

| 文件 | 行数（约） | 作用 |
|---|---|---|
| `code/b2_run_skill.py` | 100 | 基础 5 Skill 的 CLI 入口 |
| `code/b2_advanced.py` | 200 | 进阶 2 Skill 的 CLI 入口（含沙箱 timeout 处理） |
| `code/composite_skill.py` | 60 | 复合 Skill read_and_convert |
| `code/safe_python_exec.py` | 110 | 沙箱 Skill |
| `code/skills_error_codes.py` | 90 | ErrorCode 体系 |
| `skills/calculator.py` | 60 | AST 安全算术求值 |
| `skills/file_reader.py` | 70 | txt/md 读取（带超时 + ErrorCode） |
| `skills/local_file_search.py` | 90 | 关键词检索（带超时 + ErrorCode） |
| `skills/table_analyzer.py` | 80 | CSV/TSV 解析与统计（带超时 + ErrorCode） |
| `skills/format_converter.py` | 90 | markdown/json 转换（带 ErrorCode） |
| `skills/__init__.py` | 20 | `resolve_data_path` 路径安全解析 |
| `code/common/path_utils.py` | 130 | resolve_model_path 等可移植路径函数 |

### 10.3 一键运行脚本

```bash
# 服务器端（B2 个人演示 + 进阶）
bash /root/siton-tmp/HAL1000/agent/scripts/run_b2_baseline.sh
bash /root/siton-tmp/HAL1000/agent/scripts/run_b2_advanced.sh

# 一键全跑
bash /root/siton-tmp/HAL1000/agent/scripts/run_all_demos.sh        # mock
bash /root/siton-tmp/HAL1000/agent/scripts/run_all_demos.sh pj     # 含 prompt_json
```

### 10.4 输入样例

| 文件 | 内容 |
|---|---|
| `data/tool_inputs/tool_input_calculator.json` | `{"expression": "23 * 17 + 9"}` |
| `data/tool_inputs/tool_input_calculator_error.json` | `{"expression": "__import__('os')"}` |
| `data/tool_inputs/tool_input_file_reader.json` | `{"path": "docs/agent_intro.txt", "max_chars": 2000}` |
| `data/tool_inputs/tool_input_file_reader_error.json` | `{"path": "docs/missing.txt", "max_chars": 2000}` |
| `data/tool_inputs/tool_input_file_search.json` | `{"query": "Agent", "root_dir": "docs", "top_k": 3}` |
| `data/tool_inputs/tool_input_file_search_error.json` | `{"query": "Agent", "root_dir": "no_such_dir"}` |
| `data/tool_inputs/tool_input_table_analyzer.json` | `{"path": "tables/results.csv", "describe": true}` |
| `data/tool_inputs/tool_input_table_analyzer_error.json` | `{"path": "tables/not_a_table.txt"}` |
| `data/tool_inputs/tool_input_format_converter.json` | `{"text": "a\nb\nc", "target_format": "markdown"}` |
| `data/tool_inputs/tool_input_format_converter_error.json` | `{"text": "a", "target_format": "xml"}` |
| `data/tool_inputs/advanced/composite_ok.json` | `{"path": "docs/agent_intro.txt", "target_format": "markdown", "output_filename": "agent_intro_bullets.md"}` |
| `data/tool_inputs/advanced/composite_err.json` | `{"path": "docs/does_not_exist.txt", "target_format": "markdown"}` |
| `data/tool_inputs/advanced/sandbox_ok.json` | `{"source": "result = sum(range(10))"}` |
| `data/tool_inputs/advanced/sandbox_blocked.json` | `{"source": "import os; os.system('echo pwned')"}` |
| `data/tool_inputs/advanced/sandbox_timeout.json` | `{"source": "while True: pass"}` |
| `data/tool_inputs/advanced/calc_overflow.json` | `{"expression": "2 ** 1000"}` |
| `data/tool_inputs/advanced/calc_unsupported.json` | `{"expression": "import os"}` |

---

**报告结束**。B2 模块所有基础 + 进阶要求均已实现并通过测试，产物完整，可直接进入 B3 报告阅读（B3 报告见 `B3_report.md`）。