"""
hal_chat.py — HAL1000 多轮终端对话 REPL

用法：
    python hal_chat.py                          # 默认 mock 模式
    python hal_chat.py --mode prompt_json       # 真实模型
    python hal_chat.py --model_path /root/siton-tmp/HAL1000/Qwen3.5-4B
    python hal_chat.py --resume SESSION_ID      # 恢复历史会话
    python hal_chat.py --toolset basic_tools    # 指定工具集
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import multiprocessing
import os
import signal
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

# 用户输入 /stop 时设置，让正在执行的 agent turn 在下一个安全点退出
_STOP_REQUESTED = False


def _interruptible_call(fn, *args, prompt_hint: str = "", **kwargs):
    """
    把阻塞函数 fn 放到子线程运行，主线程等待完成。
    如果子线程抛异常，异常透传到主线程。
    返回 fn 的返回值。
    注意：不在这里监听 stdin（会破坏 readline），/stop 由 task_executor 的后台线程处理。
    """
    import threading

    result_box: list = [None]
    done_event = threading.Event()

    def _worker():
        try:
            result_box[0] = ("ok", fn(*args, **kwargs))
        except Exception as e:
            result_box[0] = ("err", e)
        finally:
            done_event.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    done_event.wait()

    status, val = result_box[0]
    if status == "err":
        raise val
    return val
# ── 启用终端行编辑（光标移动/退格/Ctrl+A⾜⾝）+ 历史命令（上下方向键）──
# 在 Linux 上，Python 的 input() 默认不支持光标移动/历史，必须 import readline 才会自动接管 stdin。
try:
    import readline  # noqa: F401  — side-effect import: enables GNU readline for input()
    import atexit

    _HISTORY_FILE = str(Path.home() / ".hal1000_history")
    try:
        readline.read_history_file(_HISTORY_FILE)
    except (FileNotFoundError, PermissionError):
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, _HISTORY_FILE)
except ImportError:
    # Windows 无 GNU readline，静默降级，不影响基本功能
    pass

# ---------------------------------------------------------------------------
# Project root bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# ANSI Colors (no third-party deps)
# ---------------------------------------------------------------------------
COLORS = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "cyan":    "\033[96m",    # 用户输入提示
    "green":   "\033[92m",    # 助手回答
    "yellow":  "\033[93m",    # 工具调用
    "red":     "\033[91m",    # 错误
    "gray":    "\033[90m",    # 元信息/时间戳
    "blue":    "\033[94m",    # 系统信息
    "magenta": "\033[95m",    # 工具结果
}

def C(name: str, text: str) -> str:
    """Wrap text in ANSI color code."""
    return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"

# ---------------------------------------------------------------------------
# Lazy imports with friendly fallback
# ---------------------------------------------------------------------------
try:
    from common.io_utils import read_yaml, write_json, read_json, ensure_dir
    from common.schemas import make_ai_message, make_tool_message, make_skill_result, normalize_tool_call
    from common.path_utils import resolve_from_file
    from common.logging_utils import now_iso
    _COMMON_OK = True
except ImportError as _e:
    print(C("red", f"[错误] 无法导入 common 模块: {_e}"))
    print(C("yellow", "[提示] 请确保在 HAL1000/code/ 目录下运行，或已设置正确的 PYTHONPATH"))
    sys.exit(1)

try:
    from b3_tool_layer import (
        _load_tools_config,
        _resolve_toolset,
        _validate_args,
        get_tools_schema,
    )
    _B3_OK = True
except ImportError as _e:
    print(C("red", f"[错误] 无法导入 b3_tool_layer: {_e}"))
    _B3_OK = False

try:
    from b4_local_agent_llm import generate_ai_message, PARSE_ERROR_CONTENT
    _B4_OK = True
except ImportError as _e:
    print(C("yellow", f"[警告] 无法导入 b4_local_agent_llm: {_e}"))
    PARSE_ERROR_CONTENT = "模型输出解析失败，无法生成有效工具调用或最终回答。"
    _B4_OK = False

try:
    from b1_compress import maybe_compress_messages
    _B1_COMPRESS_OK = True
except ImportError:
    _B1_COMPRESS_OK = False

# ---------------------------------------------------------------------------
# Tool Validators — semantic validation beyond status=="success"
# ---------------------------------------------------------------------------
TOOL_VALIDATORS: dict[str, Any] = {
    "file_reader": lambda r: (
        isinstance(r.get("output", {}).get("content"), str)
        and len(r.get("output", {}).get("content", "")) > 0
    ),
    "calculator": lambda r: (
        r.get("output", {}).get("result") is not None
    ),
    "local_file_search": lambda r: (
        isinstance(r.get("output", {}).get("results"), list)
        and len(r.get("output", {}).get("results", [])) > 0
    ),
    "table_analyzer": lambda r: (
        r.get("output", {}).get("shape") is not None
    ),
    "format_converter": lambda r: (
        isinstance(r.get("output", {}).get("content"), str)
        and len(r.get("output", {}).get("content", "")) > 0
    ),
    "code_executor": lambda r: (
        r.get("output", {}).get("returncode") == 0
        and isinstance(r.get("output", {}).get("stdout"), str)
    ),
    "file_writer": lambda r: (
        isinstance(r.get("output", {}).get("written_path"), str)
        and r.get("output", {}).get("num_bytes", 0) > 0
    ),
}


def validate_tool_result(tool_name: str, skill_result: dict) -> tuple[bool, str]:
    """
    语义验证工具结果。
    返回 (is_valid: bool, reason: str)
    status==success 但语义验证失败 → is_valid=False, reason="empty output"
    """
    if skill_result.get("status") != "success":
        err = skill_result.get("error") or {}
        if isinstance(err, dict):
            msg = err.get("message", "unknown error")
        else:
            msg = str(err)
        return False, f"tool error: {msg}"

    validator = TOOL_VALIDATORS.get(tool_name)
    if validator is None:
        return True, "no validator"

    try:
        ok = validator(skill_result)
    except Exception as exc:
        return False, f"validator exception: {exc}"

    if not ok:
        return False, "empty output"
    return True, "ok"

# ---------------------------------------------------------------------------
# Token budget management
# ---------------------------------------------------------------------------
MAX_INPUT_TOKENS = 3500


def _estimate_tokens(messages: list[dict]) -> int:
    """简单估算：中文字符 * 1.5 + 英文词 * 1.3"""
    total = 0
    for m in messages:
        text = m.get("content", "") or ""
        for tc in m.get("tool_calls", []):
            text += str(tc)
        total += int(len(text) * 1.5)
    return total


def _maybe_compress(messages: list[dict]) -> list[dict]:
    if _estimate_tokens(messages) > MAX_INPUT_TOKENS:
        if _B1_COMPRESS_OK:
            compressed, did = maybe_compress_messages(messages, compress_after=10, keep_recent=6)
            if did:
                print(C("gray", "[系统] 对话历史过长，已自动压缩"))
            return compressed
        else:
            print(C("gray", "[系统] 对话历史较长（b1_compress 不可用，跳过压缩）"))
    return messages

# ---------------------------------------------------------------------------
# Isolated tool execution via multiprocessing
# ---------------------------------------------------------------------------
def _worker(tool_name: str, args: dict, config_path: str, toolset: str, queue: multiprocessing.Queue) -> None:
    """Worker函数：在子进程中执行单个工具并将结果放入队列。"""
    try:
        import importlib as _il
        import inspect as _ins
        from pathlib import Path as _P
        import sys as _sys
        _here = _P(__file__).resolve().parent
        if str(_here) not in _sys.path:
            _sys.path.insert(0, str(_here))

        from common.io_utils import read_yaml
        from common.schemas import make_skill_result
        from common.path_utils import resolve_from_file
        from b3_tool_layer import _load_tools_config, _resolve_toolset, _validate_args

        from time import perf_counter
        start = perf_counter()
        cfg_path, config = _load_tools_config(config_path)
        _, allowed = _resolve_toolset(config, toolset)
        data_root_setting = config.get("settings", {}).get("data_root", "../data")
        resolved_data_root = resolve_from_file(data_root_setting, cfg_path)
        definition = config["tools"].get(tool_name)
        if definition is None or tool_name not in allowed:
            raise ValueError(f"tool {tool_name!r} not in toolset")
        _validate_args(args, definition)
        module = _il.import_module(definition["module"])
        fn = getattr(module, definition["function"])
        kwargs = dict(args)
        sig = _ins.signature(fn)
        if "data_root" in sig.parameters:
            kwargs["data_root"] = str(resolved_data_root)
        output = fn(**kwargs)
        latency_ms = round((perf_counter() - start) * 1000, 3)
        result = make_skill_result(tool_name, "success", args, output, None, latency_ms)
        queue.put(result)
    except Exception as exc:
        from common.schemas import make_skill_result
        result = make_skill_result(
            tool_name, "error", args, None,
            {"type": type(exc).__name__, "message": str(exc)}, None
        )
        queue.put(result)


def _run_tool_isolated(
    tool_name: str,
    args: dict,
    config_path: str,
    toolset: str,
    timeout: float = 10.0,
) -> dict:
    """
    在子进程中执行工具，隔离崩溃和超时。
    返回 SkillResult dict。
    """
    # Use fork context on Linux to avoid CUDA context copy issues
    ctx = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(tool_name, args, config_path, toolset, queue))
    p.start()
    try:
        result = queue.get(timeout=timeout)
        p.join(timeout=2)
        return result
    except Exception:
        p.terminate()
        p.join(timeout=2)
        from common.schemas import make_skill_result
        return make_skill_result(
            tool_name, "error", args, None,
            {"type": "Timeout", "message": f"工具执行超时 ({timeout}s)"}, None
        )

# ---------------------------------------------------------------------------
# Mock LLM generate (fallback when b4 not available)
# ---------------------------------------------------------------------------
def _mock_generate_local(messages: list[dict], tools_schema: list[dict]) -> dict:
    """
    多步 ReAct mock：模拟真实模型的 Reason → Act → Observe → Reason 循环。

    策略：
    1. 分析用户意图，提取所有需要完成的子任务
    2. 查看已完成的工具调用，判断还剩什么
    3. 如果还有未完成的步骤，继续调用工具；否则综合所有结果给出最终答案
    """
    import re

    user_messages = [m for m in messages if m.get("role") == "user"]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    last_user = user_messages[-1].get("content", "") if user_messages else ""
    lower = last_user.lower()

    # --- 解析用户意图：找出所有子任务 ---
    tasks: list[tuple[str, dict]] = []  # (tool_name, args)

    # 文件读取：仅提取真实路径格式（包含/或.ext）
    file_matches = re.findall(r'(?:^|\s|["\'])([\w/.-]+\.(?:txt|md|csv|json))(?:\s|["\']|$)', last_user)
    for fpath in file_matches:
        tasks.append(("file_reader", {"path": fpath.strip(), "max_chars": 2000}))
    if not file_matches and any(kw in lower for kw in ["读", "read", "文件", "file", "阅读"]):
        tasks.append(("file_reader", {"path": "docs/agent_intro.txt", "max_chars": 2000}))

    # 计算表达式
    calc_matches = re.findall(r"[\d]+(?:[\s]*[+\-*/][\s]*[\d]+)+", last_user)
    for expr in calc_matches:
        tasks.append(("calculator", {"expression": expr.strip()}))
    if not calc_matches and any(kw in lower for kw in ["计算", "calc", "算一算"]):
        tasks.append(("calculator", {"expression": "1+1"}))

    # 保存文件：用户要求保存/写入
    if any(kw in lower for kw in ["保存", "写入", "存到", "存入", "save", "write to file", "写文件"]):
        # 找已经生成的代码（从 code_executor 工具消息里取）
        code_content = None
        for tm in tool_messages:
            if tm.get("name") == "code_executor":
                try:
                    r = json.loads(tm.get("content", "{}"))
                    if r.get("status") == "success":
                        # 从对应的 ai_message 里取原始代码
                        for m in messages:
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                for tc in m["tool_calls"]:
                                    if tc.get("name") == "code_executor" or (isinstance(tc, dict) and tc.get("function", {}).get("name") == "code_executor"):
                                        args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                                        if isinstance(args, str):
                                            import json as _j
                                            args = _j.loads(args)
                                        code_content = args.get("code", "")
                except Exception:
                    pass
        if code_content is None:
            # 没有已执行的代码，默认写一个示例
            code_content = "# 示例代码\nresult = [i*i for i in range(1, 6)]\nprint(result)\n"
        # 推断文件名
        import re as _re
        import time as _time
        import uuid as _uuid
        fn_match = _re.search(r"[\w]+\.py", last_user)
        if fn_match:
            save_path = fn_match.group(0)
        else:
            # 推不出文件名时，用唯一名字避免多个任务都落到 output/result.py 上相互覆盖
            _stamp = _time.strftime("%Y%m%d_%H%M%S")
            _suffix = _uuid.uuid4().hex[:6]
            save_path = f"task_{_stamp}_{_suffix}.py"
        if "/" not in save_path:
            save_path = f"algorithms/{save_path}"
        tasks.append(("file_writer", {"path": save_path, "content": code_content}))

    # 搜索
    if any(kw in lower for kw in ["搜索", "查找", "search", "find"]):
        tasks.append(("local_file_search", {"query": last_user[:50]}))

    # 代码执行：用户让写代码/编程/实现某功能
    if any(kw in lower for kw in ["写代码", "编写", "实现", "写一个", "写个", "程序", "脚本",
                                   "write code", "implement", "generate code", "编程"]):
        if any(kw in lower for kw in ["斐波那契", "fibonacci", "fib"]):
            code = """def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for i in range(10):
    print(f'fib({i}) = {fib(i)}')
"""
        elif any(kw in lower for kw in ["排序", "sort"]):
            code = """import random
data = [random.randint(1, 100) for _ in range(10)]
print('原始数据:', data)
data.sort()
print('排序后:', data)
print('最大值:', data[-1], '最小值:', data[0])
"""
        elif any(kw in lower for kw in ["质数", "素数", "prime"]):
            code = """def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

primes = [i for i in range(2, 50) if is_prime(i)]
print('50以内的质数:', primes)
print('共', len(primes), '个')
"""
        elif any(kw in lower for kw in ["统计", "词频", "word count", "frequency"]):
            code = """text = 'the quick brown fox jumps over the lazy dog the fox'
words = text.split()
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
for word, count in sorted(freq.items(), key=lambda x: -x[1]):
    print(f'{word}: {count}')
"""
        else:
            code = """result = []
for i in range(1, 11):
    result.append(i * i)
print('1到10的平方数:', result)
print('总和:', sum(result))
"""
        tasks.append(("code_executor", {"code": code, "language": "python"}))

    # 如果没有识别出任何工具任务，直接回答
    if not tasks:
        snippet = last_user[:80]
        return make_ai_message(
            f"你好！我是 HAL1000，基于 Qwen3.5-4B 的本地 Agent。\n"
            f"我可以帮你读取文件、计算表达式、搜索本地文档、写代码并在子进程中执行。\n"
            f'你说："{snippet}"\n'
            f"请告诉我你需要什么帮助。",
            [],
        )

    # --- 查看已完成的工具调用 ---
    tool_results: dict[str, dict] = {}
    for tm in tool_messages:
        name = tm.get("name", "")
        try:
            res = json.loads(tm.get("content", "{}"))
        except Exception:
            res = {}
        if res.get("status") == "success":
            tool_results[name] = res.get("output", {})

    # --- 找下一个还没完成的任务 ---
    for i, (tool_name, args) in enumerate(tasks):
        already_done = any(
            tm.get("name") == tool_name and
            json.loads(tm.get("content", "{}")).get("status") == "success"
            for tm in tool_messages
        )
        already_failed = any(
            tm.get("name") == tool_name and
            json.loads(tm.get("content", "{}")).get("status") == "error"
            for tm in tool_messages
        )
        if already_done or already_failed:
            continue
        call_id = f"call_{i+1:03d}"
        return make_ai_message("", [{"id": call_id, "name": tool_name, "args": args}])

    # --- 所有子任务完成，综合输出最终答案 ---
    parts = []
    for tool_name, args in tasks:
        output = tool_results.get(tool_name, {})
        if tool_name == "file_reader":
            content = output.get("content", "（无内容）")
            num_chars = output.get("num_chars", len(content))
            parts.append(f"📄 **文件读取** `{args.get('path', '?')}`（{num_chars} 字符）：\n{content[:300]}")
        elif tool_name == "calculator":
            res = output.get("result")
            parts.append(f"🔢 **计算** `{args.get('expression', '?')}` = **{res}**")
        elif tool_name == "local_file_search":
            results = output.get("results", [])
            if results:
                items = "\n".join(f"  - {r.get('path','?')}: {r.get('snippet','')[:60]}" for r in results[:3])
                parts.append(f"🔍 **搜索结果**（共 {len(results)} 条）：\n{items}")
            else:
                parts.append("🔍 **搜索**：未找到相关文件。")
        elif tool_name == "code_executor":
            stdout = output.get("stdout", "")
            stderr = output.get("stderr", "")
            returncode = output.get("returncode", -1)
            elapsed = output.get("elapsed_ms", 0)
            if returncode == 0:
                parts.append(
                    f"💻 **代码执行成功**（{elapsed:.0f}ms）\n"
                    f"```\n{stdout.strip()}\n```"
                )
            else:
                parts.append(
                    f"💻 **代码执行出错**（exit={returncode}）\n"
                    f"stderr: {stderr[:200]}"
                )
        elif tool_name == "file_writer":
            written = output.get("written_path", "?")
            nb = output.get("num_bytes", 0)
            nl = output.get("num_lines", 0)
            parts.append(f"💾 **文件已保存** ({nl} 行, {nb} 字节)\n路径: `{written}`")

    if not parts:
        return make_ai_message("所有工具均返回失败，请检查输入或文件路径。", [])
    summary = "\n\n".join(parts)
    return make_ai_message(
        f"所有任务已完成，以下是汇总结果：\n\n{summary}",
        [],
    )

# ---------------------------------------------------------------------------
# Streaming mock output
# ---------------------------------------------------------------------------
def _stream_print_mock(text: str, color: str = "green") -> None:
    """Simulate streaming output: print word by word with slight delay."""
    words = text.split(" ")
    for i, word in enumerate(words):
        end = " " if i < len(words) - 1 else ""
        print(f"{COLORS[color]}{word}{end}", end="", flush=True)
        time.sleep(0.02)
    print(COLORS["reset"])

# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
_SESSION_DIR = _PROJECT_ROOT / "sessions"


def _make_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    return _SESSION_DIR / f"{session_id}.json"


def _save_session_to_file(
    session_id: str,
    messages: list[dict],
    tool_rounds: int,
    mode: str,
    created_at: str,
    parent_id: str | None = None,
) -> Path:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    data = {
        "session_id": session_id,
        "parent_id": parent_id,
        "created_at": created_at,
        "saved_at": now_iso(),
        "mode": mode,
        "tool_rounds": tool_rounds,
        "messages": messages,
    }
    write_json(data, path)
    return path


def _load_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# REPL State
# ---------------------------------------------------------------------------
class HALChat:
    def __init__(
        self,
        mode: str = "mock",
        tools_config: str | None = None,
        model_config: str | None = None,
        toolset: str = "basic_tools",
        max_turns: int = 10,
        verbose: bool = False,
        system_prompt: str | None = None,
        session_id: str | None = None,
        resume_messages: list[dict] | None = None,
        resume_tool_rounds: int = 0,
        auto_confirm: bool = False,
    ):
        self.mode = mode
        self.toolset = toolset
        self.max_turns = max_turns
        self.verbose = verbose
        # 人在环确认：DAG 规划后执行前是否需要人工确认。
        # 非交互环境（stdin 不是 tty，如管道/脚本）强制自动确认，避免卡死等待输入。
        self.auto_confirm = auto_confirm or (not sys.stdin.isatty())

        # Paths
        self.tools_config = tools_config or str(_PROJECT_ROOT / "configs" / "tools.yaml")
        self.model_config = model_config or str(_PROJECT_ROOT / "configs" / "model.yaml")

        # Load tools
        self.tools_schema: list[dict] = []
        self._config_path: str = self.tools_config
        self._tool_names: list[str] = []
        self._load_tools()

        # System prompt
        default_system_path = _PROJECT_ROOT / "prompts" / "local_tool_agent.txt"
        if system_prompt:
            try:
                self.system_prompt = Path(system_prompt).read_text(encoding="utf-8")
            except Exception:
                self.system_prompt = system_prompt
        elif default_system_path.exists():
            self.system_prompt = default_system_path.read_text(encoding="utf-8")
        else:
            self.system_prompt = (
                "You are HAL1000, a local tool-using AI Agent built on Qwen3.5-4B. "
                "Use available tools to answer user questions accurately. "
                "Do not invent file contents. If a tool call fails, say so honestly."
            )

        # Session state
        self.session_id = session_id or _make_session_id()
        self.created_at = now_iso()
        self.tool_rounds = resume_tool_rounds
        self.parent_id: str | None = None  # 主线会话为 None，分支会话为主线 session_id

        # Messages
        self.messages: list[dict] = resume_messages or []
        if not any(m.get("role") == "system" for m in self.messages):
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Signal handler
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Tool result cache (LRU, 内存+磁盘持久化)
        # 不缓存 code_executor / file_writer（副作用工具，每次都应真实执行）
        _CACHE_BLACKLIST = {"code_executor", "file_writer"}
        cache_path = _PROJECT_ROOT / "outputs" / "tool_cache.json"
        try:
            from tool_cache import ToolCache
            self._cache: ToolCache | None = ToolCache(
                max_entries=512, persist_path=cache_path
            )
            self._cache_blacklist: set[str] = _CACHE_BLACKLIST
            self._cache_hits = 0
            self._cache_misses = 0
        except ImportError:
            self._cache = None
            self._cache_blacklist = set()
            self._cache_hits = 0
            self._cache_misses = 0

        # Episodic Memory（分层上下文管理）
        try:
            from episodic_memory import EpisodicMemory
            self._episodic: EpisodicMemory | None = EpisodicMemory(
                session_id=self.session_id,
                persist_dir=str(_PROJECT_ROOT / "outputs" / "episodic"),
                max_working_tokens=2000,
                archive_after_turns=8,
                keep_recent=6,
            )
        except Exception:
            self._episodic = None

    def _load_tools(self) -> None:
        """Load tools schema from config."""
        if not _B3_OK:
            print(C("yellow", "[警告] b3_tool_layer 不可用，工具调用将被跳过"))
            return
        try:
            self.tools_schema = get_tools_schema(self.tools_config, self.toolset)
            _, config = _load_tools_config(self.tools_config)
            _, names = _resolve_toolset(config, self.toolset)
            self._tool_names = names
            if self.verbose:
                print(C("gray", f"[调试] 加载工具集 {self.toolset}: {', '.join(names)}"))
        except Exception as exc:
            print(C("red", f"[错误] 加载工具配置失败: {exc}"))

    def _handle_sigint(self, sig: int, frame: Any) -> None:
        print()
        print(C("blue", "[系统] 收到中断信号，正在保存会话..."))
        self._save_session()
        print(C("blue", "[系统] 已保存，再见！"))
        sys.exit(0)

    def _save_session(self) -> Path:
        path = _save_session_to_file(
            self.session_id,
            self.messages,
            self.tool_rounds,
            self.mode,
            self.created_at,
            parent_id=getattr(self, 'parent_id', None),
        )
        return path

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _execute_one_tool(
        self, call: dict, max_retries: int = 2
    ) -> tuple[dict, dict, bool, str]:
        """
        Execute one tool call (isolated), with exponential backoff retry.
        Returns (tool_message, skill_result, is_valid, reason).

        Retry 策略：
          - 工具返回 status=="error" 且错误不是 "PathEscape" 或 "ValidationError"（那些重试没意义）时重试
          - 间隔：0.5s, 1.0s（指数退退）
          - 每次重试打印 [retry N/M]
        """
        tool_name = call["name"]
        args = call["args"]
        _NO_RETRY_ERRORS = {"PathEscape", "ValidationError", "PermissionError"}
        use_cache = (
            self._cache is not None
            and tool_name not in self._cache_blacklist
        )

        # ── 缓存查找 ─────────────────────────────────────────────
        if use_cache:
            cached = self._cache.get(tool_name, args)
            if cached is not None:
                self._cache_hits += 1
                skill_result = cached["result"]
                is_valid, reason = validate_tool_result(tool_name, skill_result)
                output = skill_result.get("output", {}) or {}
                summary = _output_summary(tool_name, output)
                print(
                    C("yellow", f"  [✓ {tool_name}]") +
                    C("magenta", f" → {summary}") +
                    C("gray", " (0.0ms) ") +
                    C("cyan", "[cache hit]") +
                    (C("green", " [valid]") if is_valid else C("red", f" [invalid: {reason}]"))
                )
                content = json.dumps(skill_result, ensure_ascii=False, separators=(",", ":"))
                tool_message = make_tool_message(
                    call["id"], tool_name, content, skill_result.get("status", "error")
                )
                return tool_message, skill_result, is_valid, reason
            else:
                self._cache_misses += 1

        args_str = " ".join(f'{k}="{str(v)[:60]}"' for k, v in args.items())
        print(C("yellow", f"  [▶ {tool_name}] {args_str} ..."), flush=True)

        # ── 真实执行（含 retry） ──────────────────────────────────
        skill_result = {}
        t0 = perf_counter()
        for attempt in range(max_retries + 1):
            t0 = perf_counter()
            if _B3_OK:
                if tool_name == "image_qa":
                    # 视觉模型缓存在进程内全局变量里，子进程 fork 后缓存为空每次重载。
                    # image_qa 必须在主进程里直接调用，才能复用已加载的视觉模型。
                    try:
                        from b3_tool_layer import execute_tool_calls as _etc
                        results = _etc(
                            [{"id": "call_direct", "name": tool_name, "args": args}],
                            self.tools_config,
                            self.toolset,
                        )
                        skill_result = results[0] if results else {"status": "error", "error": {"type": "Empty", "message": "no result"}}
                    except Exception as e:
                        skill_result = {"status": "error", "error": {"type": type(e).__name__, "message": str(e)}}
                else:
                    skill_result = _run_tool_isolated(
                        tool_name, args, self.tools_config, self.toolset, timeout=30.0
                    )
            else:
                skill_result = _mock_skill_result(tool_name, args)

            if skill_result.get("status") == "success":
                break

            err = skill_result.get("error") or {}
            err_type = err.get("type", "") if isinstance(err, dict) else ""
            if err_type in _NO_RETRY_ERRORS or attempt >= max_retries:
                break
            wait = 0.5 * (2 ** attempt)
            print(C("yellow", f"  [retry {attempt+1}/{max_retries}] {tool_name} 失败（{err_type}），{wait:.1f}s 后重试..."))
            time.sleep(wait)

        elapsed_ms = round((perf_counter() - t0) * 1000, 1)
        is_valid, reason = validate_tool_result(tool_name, skill_result)

        # ── 成功则写入缓存 ────────────────────────────────────────
        if use_cache and skill_result.get("status") == "success":
            self._cache.put(tool_name, args, skill_result)

        # ── 打印结果 ──────────────────────────────────────────────
        status_icon = "✓" if is_valid else "✗"
        valid_tag = C("green", "[valid]") if is_valid else C("red", f"[invalid: {reason}]")

        if skill_result.get("status") == "success" and is_valid:
            output = skill_result.get("output", {}) or {}
            summary = _output_summary(tool_name, output)
            print(
                C("yellow", f"  [{status_icon} {tool_name}]") +
                C("magenta", f" → {summary}") +
                C("gray", f" ({elapsed_ms}ms) ") + valid_tag
            )
        else:
            err = skill_result.get("error") or {}
            err_msg = err.get("message", reason) if isinstance(err, dict) else str(err)
            err_code = err.get("type", "ERROR") if isinstance(err, dict) else "ERROR"
            print(C("red", f"  [✗ {tool_name}] ERROR: {err_msg} ({err_code})"))
            if tool_name == "file_reader" and "not found" in err_msg.lower():
                # 尝试从最近一次 local_file_search 结果里找到匹配的 full_path，并直接修正 args
                _filename = args.get("path", "")
                _fixed = False
                for _msg in reversed(self.messages):
                    if _msg.get("name") == "local_file_search" and _msg.get("status") == "success":
                        try:
                            _sr = json.loads(_msg.get("content", "{}"))
                            _entries = (_sr.get("output") or {}).get("entries", [])
                            for _e in _entries:
                                if _e.get("type") == "file" and Path(_e.get("full_path", "")).name == Path(_filename).name:
                                    args["path"] = _e["full_path"]
                                    print(C("gray", f"  [路径修正] file_reader path: {_filename!r} → {_e['full_path']!r}"))
                                    _fixed = True
                                    break
                        except Exception:
                            pass
                    if _fixed:
                        break
                if not _fixed:
                    print(C("gray", f"  [提示] 如果文件在 data/ 外，请用绝对路径（如 /root/siton-tmp/.../xxx.log）"))

        content = json.dumps(skill_result, ensure_ascii=False, separators=(",", ":"))
        tool_message = make_tool_message(
            call["id"], tool_name, content, skill_result.get("status", "error")
        )
        return tool_message, skill_result, is_valid, reason

    # ------------------------------------------------------------------
    # Task Validator ("裁判")
    # ------------------------------------------------------------------
    def _judge_by_keywords(self, user_input: str, tool_history: list[dict]) -> tuple[bool, str]:
        """
        关键词规则裁判（mock 模式 fallback）。
        返回 (is_done: bool, reason: str)
        """
        lower = user_input.lower()
        executed: dict[str, list[dict]] = {}
        for record in tool_history:
            name = record["name"]
            if record["status"] == "success":
                executed.setdefault(name, []).append(record["output"])

        requirements: list[tuple[str, str]] = []
        code_kw = ["写代码", "编写", "实现", "写一个", "写个", "程序", "脚本", "排序", "算法",
                   "冒泡", "fibonacci", "prime", "write code", "implement", "coding"]
        save_kw = ["保存", "写入", "存到", "存入", "save", "write to file", "写文件", "文件里"]
        read_kw = ["读", "阅读", "打开", "read", "查看文件"]
        calc_kw = ["计算", "calc", "算"]

        if any(kw in lower for kw in code_kw):
            requirements.append(("code_executor", "执行代码验证"))
        if any(kw in lower for kw in save_kw):
            requirements.append(("file_writer", "保存文件"))
        if any(kw in lower for kw in read_kw) and not any(kw in lower for kw in code_kw):
            requirements.append(("file_reader", "读取文件"))
        if any(kw in lower for kw in calc_kw) and not any(kw in lower for kw in code_kw):
            requirements.append(("calculator", "完成计算"))

        if not requirements:
            return True, "[规则裁判] 无明确工具需求，视为完成"

        unmet = []
        for tool_name, desc in requirements:
            if tool_name not in executed:
                unmet.append(f"{desc}（{tool_name} 未执行）")
            elif tool_name == "file_writer":
                if not any(o.get("num_bytes", 0) > 0 for o in executed[tool_name]):
                    unmet.append(f"{desc}（文件写入字节数为 0）")
        if unmet:
            return False, "[规则裁判] 未完成：" + "；".join(unmet)
        return True, "[规则裁判] 所有需求已满足"

    def _judge_by_llm(self, user_input: str, tool_history: list[dict], model_answer: str = "") -> tuple[bool, str]:
        """
        LLM 语义裁判：让模型判断任务是否真正完成。
        返回 (is_done: bool, reason: str)
        失败时 fallback 到关键词规则裁判。
        """
        if not _B4_OK:
            return self._judge_by_keywords(user_input, tool_history)

        # 构造 tool_summary：把 tool_history 格式化为可读文本
        summary_lines = []
        for rec in tool_history:
            name = rec["name"]
            status = rec["status"]
            out = rec.get("output") or {}
            if name == "code_executor":
                rc = out.get("returncode", "?")
                lines = len(out.get("stdout", "").strip().splitlines())
                summary_lines.append(f"- {name}: {status}, exit={rc}, stdout={lines}行")
            elif name == "file_writer":
                nb = out.get("num_bytes", 0)
                path = out.get("written_path", "?")
                summary_lines.append(f"- {name}: {status}, {nb} bytes → {path}")
            elif name == "file_reader":
                nc = out.get("num_chars", 0)
                summary_lines.append(f"- {name}: {status}, {nc} chars")
            elif name == "shell_exec":
                stdout = out.get("stdout", "")
                stderr = out.get("stderr") or ""
                rc = out.get("returncode", "?")
                # 取 stdout 头 200 字符作为摘要，让裁判能看到内容
                preview = stdout[:200].replace("\n", " ") if stdout else ""
                if stderr and not stdout:
                    preview = f"stderr: {stderr[:100]}"
                summary_lines.append(f"- {name}: {status}, rc={rc}, output={repr(preview)}")
            elif name == "calculator":
                res = out.get("result", "?")
                summary_lines.append(f"- {name}: {status}, result={res}")
            else:
                summary_lines.append(f"- {name}: {status}")
        tool_summary = "\n".join(summary_lines) if summary_lines else "（无工具调用记录）"

        judge_prompt = (
            "你是任务完成度裁判。判断模型的回答是否满足了用户需求。\n"
            "只输出一个 JSON 对象，格式：{\"done\": true/false, \"reason\": \"一句话说明\"}\n"
            "不要输出其他任何文字。\n\n"
            f"用户原始需求\n{user_input}\n\n"
        )
        if model_answer:
            judge_prompt += (
                f"模型的回答\n{model_answer[:600]}\n\n"
                "如果模型的回答内容已经直接回答了用户的问题，done=true。不要根据工具调用来判断。\n"
                "只有回答为空、或明显答非所问、或声称无法完成时才 done=false。"
            )
        else:
            judge_prompt += (
                f"已执行工具记录\n{tool_summary}\n\n"
                "判断标准\n"
                "- 如果需求是写代码：code_executor 必须 exit=0 且有输出\n"
                "- 如果需求是保存文件：file_writer 必须成功且 bytes > 0\n"
                "- 如果需求是读取/分析文件：file_reader 或 table_analyzer 或 shell_exec(内容非空) 必须成功\n"
                "- 如果需求是浏览目录/执行 shell 命令：shell_exec 成功并且 output 非空即为完成\n"
                "- 如果需求是计算：calculator 必须成功\n"
                "- 如果需求是纯问答/闲聊/解释概念等，不需要任何工具，done=true\n"
                "- 其他需求：根据工具执行情况综合判断，不要过于严格\n"
                "- 只有明确需要工具但没有任何工具调用时才设 done=false"
            )

        messages = [
            {"role": "system", "content": "你是任务完成度裁判，只输出 JSON。"},
            {"role": "user",   "content": judge_prompt},
        ]

        try:
            from b4_local_agent_llm import generate_text_only
            raw = generate_text_only(
                model_config=self.model_config,
                messages=messages,
                mode=self.mode,
                max_new_tokens=128,
            )
            # 提取 JSON
            import re as _re
            m = _re.search(r'\{.*?\}', raw, _re.DOTALL)
            if not m:
                raise ValueError(f"LLM 裁判输出无 JSON: {raw[:80]}")
            data = json.loads(m.group(0))
            done = bool(data.get("done", False))
            reason = f"[LLM裁判] {data.get('reason', '无说明')}"
            return done, reason
        except Exception as e:
            print(C("gray", f"  [裁判] LLM 裁判失败 ({e})，fallback 到规则裁判"))
            return self._judge_by_keywords(user_input, tool_history)

    def _judge(self, user_input: str, tool_history: list[dict], model_answer: str = "") -> tuple[bool, str]:
        """
        裁判入口：mock 模式用关键词规则，真实模型模式用 LLM 语义判断。
        """
        if self.mode == "mock":
            return self._judge_by_keywords(user_input, tool_history)
        return self._judge_by_llm(user_input, tool_history, model_answer=model_answer)

    # ------------------------------------------------------------------
    # 人在环确认：展示 DAG 计划，让用户选择 确认/拒绝+反馈
    # ------------------------------------------------------------------
    def _confirm_plan(self, dag) -> tuple[bool, str]:
        """
        展示 Planner 生成的任务计划，等待用户确认。

        返回 (confirmed, feedback)：
          confirmed=True  → 直接执行
          confirmed=False, feedback=""      → 用户取消（n/空）
          confirmed=False, feedback="..."   → 用户给了修改意见，需重新规划
        """
        print(C("bold", "\n  ┌─ 任务计划确认 " + "─" * 30))
        if dag.strategy:
            print(C("cyan", f"  │ 思路: {dag.strategy}"))
        print(C("gray", f"  │ 共 {len(dag.nodes)} 个步骤："))
        for i, node in enumerate(dag.nodes, 1):
            dep_str = f"  (依赖: {', '.join(node.depends_on)})" if node.depends_on else ""
            print(C("gray", f"  │  {i}. [{node.tool_name}] {node.description}{dep_str}"))
        print(C("bold", "  └─" + "─" * 40))
        print(C("yellow", "  这个计划对吗？回车确认执行，或输入修改意见（输入 n 取消，/stop 停止任务）: "), end="")
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False, ""
        if answer == "" or answer.lower() in ("y", "yes", "好", "确认", "ok"):
            return True, ""
        if answer.lower() in ("n", "no", "取消"):
            return False, ""
        if answer.lower() == "/stop":
            global _STOP_REQUESTED
            _STOP_REQUESTED = True
            print(C("yellow", "  [stop] 任务已停止"))
            return False, ""
        # 其他内容视为修改意见，触发重新规划
        return False, answer

    # ------------------------------------------------------------------
    # FSM execution loop
    # ------------------------------------------------------------------
    def _run_agent_turn(self, user_input: str) -> None:
        """
        带状态机的 Agent 执行循环。

        状态：
          RUNNING   → 正在执行工具
          JUDGING   → 模型给出了最终回答，裁判检验是否真正完成
          NUDGING   → 裁判判定未完成，自动向模型注入 nudge 强制继续
          DONE      → 裁判确认完成
          FAILED    → 达到最大轮次或最大 nudge 次数
        """
        global _STOP_REQUESTED
        _STOP_REQUESTED = False  # 每次新任务开始重置
        # ── DAG 模式 ──────────────────────────────────────────────
        try:
            from task_planner import plan, TaskDAG
            from task_executor import execute_dag
            _DAG_OK = True
        except ImportError:
            _DAG_OK = False

        # ── 生成前：从 Episodic 内存召回相关历史，注入 working memory ──
        if self._episodic is not None and self._episodic.should_archive(self.messages):
            recalled = self._episodic.recall(user_input, top_k=3)
            if recalled:
                self.messages = self._episodic.trim_working_memory(
                    self.messages, recalled_turns=recalled
                )
                print(C("gray", f"  [记忆] 将 {len(recalled)} 条历史记忆注入 working memory"))

        if _DAG_OK:
            available = [t["function"]["name"] for t in self.tools_schema]
            # Planner：真实模型用 LLM 规划，mock 模式用极简规则
            is_mock = (self.mode == "mock")
            planner_llm_fn = None if is_mock else self._llm_plan

            # 从 Artifact Registry 取最近已写入的文件，供 Planner 参考
            recent_written_files: list[str] = []
            try:
                from artifact_registry import ArtifactRegistry
                _art = ArtifactRegistry(
                    session_id=self.session_id,
                    persist_dir=str(_PROJECT_ROOT / "outputs" / "artifacts")
                )
                for artifact in _art.list_session():
                    if artifact.tool_name == "file_writer":
                        cur = artifact.current
                        if cur:
                            path = cur.output.get("written_path", "")
                            if path:
                                recent_written_files.append(path)
                recent_written_files = recent_written_files[-10:]
            except Exception:
                pass

            # ── 人在环节点：规划后、执行前先征求人确认 ─────────────────
            current_planning_input = user_input
            dag = plan(current_planning_input, available, llm_fn=planner_llm_fn, mock=is_mock,
                       recent_files=recent_written_files if recent_written_files else None,
                       conversation_history=self.messages)
            replan_attempts = 0
            MAX_REPLAN = 2
            while not dag.is_empty() and not self.auto_confirm:
                confirmed, feedback = self._confirm_plan(dag)
                if confirmed:
                    break
                if not feedback:
                    print(C("blue", "  [系统] 已取消本次任务"))
                    return
                replan_attempts += 1
                if replan_attempts > MAX_REPLAN:
                    print(C("yellow", f"  [规划器] 已重新规划 {MAX_REPLAN} 次仍未确认，改走 FSM 逐步处理"))
                    dag = TaskDAG(nodes=[], user_input=user_input)
                    break
                current_planning_input = f"{user_input}\n\n[用户补充/修改要求]: {feedback}"
                print(C("blue", f"  [规划器] 根据反馈重新规划（第 {replan_attempts} 次）..."))
                dag = plan(current_planning_input, available, llm_fn=planner_llm_fn, mock=is_mock,
                           recent_files=recent_written_files if recent_written_files else None,
                           conversation_history=self.messages)

            if not dag.is_empty():
                print(C("blue", f"  [规划器{'(LLM)' if not is_mock else '(mock)'}] 识别到 {len(dag.nodes)} 个任务节点，进入 DAG 执行模式"))
                llm_fn = self._llm_generate_code if (self.mode != "mock" and _B4_OK) else None
                dag = execute_dag(
                    dag, self.tools_config, self.toolset,
                    llm_generate_fn=llm_fn, verbose=self.verbose,
                    node_timeout=120.0,
                    tool_cache=self._cache,
                    cache_blacklist=self._cache_blacklist,
                    cache_hit_counter=lambda: setattr(self, '_cache_hits', self._cache_hits + 1),
                    cache_miss_counter=lambda: setattr(self, '_cache_misses', self._cache_misses + 1),
                )
                # 把结果转成 tool_messages 喂给模型做最终汇总
                self.messages.append({"role": "user", "content": user_input})
                for node in dag.nodes:
                    if node.result:
                        content = json.dumps(node.result, ensure_ascii=False)
                        tool_msg = make_tool_message(
                            node.task_id, node.tool_name,
                            content, node.result.get("status", "error"),
                        )
                        self.messages.append(tool_msg)
                # ── artifact registry 接入 DAG 执行 ────────────────
                try:
                    from artifact_registry import ArtifactRegistry
                    registry = ArtifactRegistry(
                        session_id=self.session_id,
                        persist_dir=str(_PROJECT_ROOT / "outputs" / "artifacts")
                    )
                    for node in dag.nodes:
                        if node.result and node.result.get("status") == "success":
                            art_id = registry.register(
                                tool_name=node.tool_name,
                                args=node.args_template,
                                output=node.result.get("output") or {},
                            )
                            if self.verbose:
                                print(C("gray", f"  [artifact] {node.tool_name} → {art_id}"))
                except ImportError:
                    pass

                # ── DAG 全部失败 → fallback 到 FSM ReAct ────────
                all_failed = all(n.status == "failed" for n in dag.nodes)
                if all_failed:
                    fail_reasons = "; ".join(
                        f"{n.task_id}({n.tool_name}): "
                        + ((n.result or {}).get("error") or {}).get("message", "unknown")[:40]
                        for n in dag.nodes
                    )
                    print(C("yellow",
                        f"  [DAG→FSM] 所有节点均失败（{fail_reasons}），切换到 FSM ReAct 路径"))
                    # 清理本轮 DAG 产生的 tool 消息，交给 FSM 重新处理
                    self.messages = [m for m in self.messages if m.get("role") != "tool"]
                    # fall through to FSM below（不 return）
                else:
                    # ── 构建最终汇总回答 ──────────────────────────
                    if self.mode == "mock" or not _B4_OK:
                        # mock 模式：直接从 DAG 节点结果合成汇总
                        final_content = _dag_summary(dag)
                        final_ai = make_ai_message(final_content, [])
                        self.messages.append(final_ai)
                    else:
                        # 真实模型：喂入汇总 prompt
                        self.messages.append({
                            "role": "user",
                            "content": "请根据以上所有工具执行结果给出最终汇总回答，包含真实文件路径（如有）。",
                        })
                        final_ai = self._generate(self.messages, stage="summarize")
                        self.messages.append(final_ai)
                        final_content = final_ai.get("content", "").strip()
                        # summarize 解析失败时重试一次，并给更明确的提示
                        if not final_content or final_content == PARSE_ERROR_CONTENT:
                            retry_msg = "工具已执行完毕，请直接用中文回答用户的问题，包含工具返回的关键信息。"
                            self.messages.append({"role": "user", "content": retry_msg})
                            final_ai2 = self._generate(self.messages, stage="summarize")
                            self.messages.append(final_ai2)
                            final_content2 = final_ai2.get("content", "").strip()
                            if final_content2 and final_content2 != PARSE_ERROR_CONTENT:
                                final_content = final_content2
                    # ── B1 裁判接入 DAG 路径（Summarize 后质量验证）───
                    dag_tool_history = [
                        {
                            "name": n.tool_name,
                            "status": n.result.get("status", "error") if n.result else "error",
                            "output": n.result.get("output", {}) if n.result else {},
                        }
                        for n in dag.nodes
                    ]
                    is_done, judge_reason = self._judge(current_planning_input, dag_tool_history, model_answer=final_content)
                    print(C("gray", f"  [裁判/DAG] {judge_reason}"))
                    if not is_done:
                        # 裁判驳回：fallback 到 FSM，让 NUDGING 机制接手
                        visible_dag_reason = judge_reason.replace("[LLM裁判] ", "").replace("[关键词裁判] ", "")
                        print(C("yellow",
                            f"  [DAG→FSM] 裁判驳回汇总结果（{judge_reason}），切换 FSM 重推"))
                        self.messages = [m for m in self.messages if m.get("role") != "tool"]
                        # fall through to FSM（不 return，不打印，不归档）
                    else:
                        if final_content:
                            print(C("bold", "HAL1000: "), end="")
                            if self.mode == "mock":
                                _stream_print_mock(final_content, "green")
                            else:
                                print(C("green", final_content))
                        # 裁判评语显示给用户
                        visible_dag_ok_reason = judge_reason.replace("[LLM裁判] ", "").replace("[关键词裁判] ", "")
                        print(C("gray", f"  ✓ 裁判评语：{visible_dag_ok_reason}"))
                        # ── Episodic 归档（DAG 路径）──────────────────
                        if self._episodic is not None:
                            user_m = {"role": "user", "content": user_input}
                            ai_m = (self.messages[-1] if self.messages and self.messages[-1].get("role") == "assistant"
                                    else {"role": "assistant", "content": final_content})
                            tool_ms = [m for m in self.messages if m.get("role") == "tool"]
                            self._episodic.archive_turn(user_m, ai_m, tool_ms)
                            if self._episodic.should_archive(self.messages):
                                self.messages = self._episodic.trim_working_memory(self.messages)
                                stats = self._episodic.stats()
                                print(C("gray", f"  [记忆] working memory 已修剪，已归档 {stats['archived_turns']} 轮"))
                        self._save_session()
                        return
        # ── 原有 FSM ReAct 循环（DAG 未识别时走这里）──────────────

        MAX_NUDGES = 3  # 裁判最多驳回并重推几次

        # current_planning_input 在 DAG 路径里会被更新（含用户在人在环节的修改意见）
        # FSM 路径如果之前没有经过 DAG 规划，则等同于原始 user_input
        if 'current_planning_input' not in dir():
            current_planning_input = user_input

        self.messages.append({"role": "user", "content": user_input})
        tool_history: list[dict] = []   # 记录所有成功/失败的工具执行结果
        nudge_count = 0
        state = "RUNNING"

        for turn in range(self.max_turns + MAX_NUDGES):
            # --- /stop 检查 ---
            if _STOP_REQUESTED:
                print(C("yellow", "  [stop] 收到停止指令，放弃当前任务"))
                return
            # --- token 预算检查 ---
            msgs = _maybe_compress(self.messages)
            if msgs is not self.messages:
                self.messages = msgs

            # --- 生成 ---
            ai_message = self._generate(self.messages)
            self.messages.append(ai_message)
            tool_calls = ai_message.get("tool_calls", [])
            content = ai_message.get("content", "").strip()

            # --- RUNNING：有工具调用 → 执行 ---
            if tool_calls:
                state = "RUNNING"
                self.tool_rounds += 1
                for raw_call in tool_calls:
                    try:
                        call = normalize_tool_call(raw_call)
                    except Exception as exc:
                        print(C("red", f"[错误] 工具调用格式无效: {exc}"))
                        continue
                    tool_msg, skill_result, is_valid, reason = self._execute_one_tool(call)
                    self.messages.append(tool_msg)
                    # 记入工具历史供裁判 + Artifact Registry 使用
                    tool_history.append({
                        "name": call["name"],
                        "args": call.get("arguments") or call.get("args") or {},
                        "status": skill_result.get("status", "error"),
                        "output": skill_result.get("output") or {},
                        "valid": is_valid,
                    })
                continue  # 执行完工具，回到循环让模型观察结果

            # --- 模型给出了最终回答（无 tool_calls）→ 进入 JUDGING ---
            if content and content != PARSE_ERROR_CONTENT:
                state = "JUDGING"
                is_done, judge_reason = self._judge(current_planning_input, tool_history)
                print(C("gray", f"  [裁判] {judge_reason}"))

                if is_done:
                    # 裁判通过 → 打印最终答案，结束
                    state = "DONE"
                    print(C("bold", "HAL1000: "), end="")
                    if self.mode == "mock":
                        _stream_print_mock(content, "green")
                    else:
                        print(C("green", content))
                    # 打印裁判评语（去掉 [LLM裁判] 前缀，显示给用户）
                    visible_reason = judge_reason.replace("[LLM裁判] ", "").replace("[关键词裁判] ", "")
                    print(C("gray", f"  ✓ 裁判评语：{visible_reason}"))
                    break
                else:
                    # 裁判驳回 → NUDGING
                    nudge_count += 1
                    visible_reason = judge_reason.replace("[LLM裁判] ", "").replace("[关键词裁判] ", "")
                    if nudge_count > MAX_NUDGES:
                        state = "FAILED"
                        print(C("yellow", f"  [裁判] 已驳回 {MAX_NUDGES} 次仍未完成，放弃"))
                        print(C("bold", "HAL1000: "), end="")
                        print(C("green", content))  # 打印最后的回答
                        print(C("yellow", f"  ⚠ 裁判评语：{visible_reason}"))
                        break

                    state = "NUDGING"
                    nudge_msg = (
                        f"[系统校验] 任务尚未完成：{judge_reason}。"
                        f"请继续调用工具完成剩余步骤，不要只描述，必须实际执行。"
                        f"（已重试 {nudge_count}/{MAX_NUDGES} 次）"
                    )
                    print(C("yellow", f"  [状态机] 裁判驳回，自动 nudge ({nudge_count}/{MAX_NUDGES}): {judge_reason}"))
                    # 把 nudge 注入为 system 消息（让模型感受到压力）
                    self.messages.append({"role": "user", "content": nudge_msg})
                    continue

            elif not content or content == PARSE_ERROR_CONTENT:
                # 模型没有输出任何东西（解析失败或空内容）
                if tool_history:
                    # 已经有工具执行过，nudge 推它给最终答案
                    nudge_count += 1
                    if nudge_count > MAX_NUDGES:
                        state = "FAILED"
                        print(C("yellow", "  [状态机] 模型持续无输出，放弃"))
                        break
                    nudge_msg = "请根据以上工具执行结果，给出最终回答。"
                    print(C("yellow", f"  [状态机] 模型无输出，注入 nudge ({nudge_count}/{MAX_NUDGES})"))
                    self.messages.append({"role": "user", "content": nudge_msg})
                    continue
                else:
                    # 什么都没做就停了
                    print(C("yellow", "  [状态机] 模型无输出且无工具调用"))
                    break

        else:
            print(C("yellow", f"[状态机] 已达到最大轮次 ({self.max_turns + MAX_NUDGES})，强制停止。最终状态: {state}"))

        # ── B1 Artifact Registry 注册（FSM 路径）────────────────────
        try:
            from artifact_registry import ArtifactRegistry
            _registry = ArtifactRegistry(
                session_id=self.session_id,
                persist_dir=str(_PROJECT_ROOT / "outputs" / "artifacts")
            )
            for record in tool_history:
                if record.get("status") == "success":
                    art_id = _registry.register(
                        tool_name=record["name"],
                        args=record.get("args", {}),
                        output=record.get("output") or {},
                    )
                    if self.verbose:
                        print(C("gray", f"  [artifact/FSM] {record['name']} → {art_id}"))
        except ImportError:
            pass

        # ── Episodic 归档：把本轮 user+ai+tool 归入长期记忆 ──────────
        if self._episodic is not None:
            user_m = {"role": "user", "content": user_input}
            ai_m = next((m for m in reversed(self.messages)
                         if m.get("role") == "assistant"), {"role": "assistant", "content": ""})
            tool_ms = [m for m in self.messages if m.get("role") == "tool"]
            self._episodic.archive_turn(user_m, ai_m, tool_ms)

            # 如果 working memory 过长，修剪并将放展事实块注入
            if self._episodic.should_archive(self.messages):
                recalled = self._episodic.recall(user_input, top_k=3)
                self.messages = self._episodic.trim_working_memory(
                    self.messages, recalled_turns=recalled
                )
                stats = self._episodic.stats()
                print(C("gray", f"  [记忆] 工作记忆已修剪，已归档 {stats['archived_turns']} 轮，"
                               f"当前 working memory {len(self.messages)} 条"))

    def _llm_plan(self, messages: list[dict]) -> dict:
        """
        Planner LLM 调用：用 generate_text_only 绕过工具调用格式，
        强制模型输出纯 JSON（task_planner.txt system prompt）。
        返回 {"role": "assistant", "content": "..."}（JSON 字符串）。
        """
        try:
            from b4_local_agent_llm import generate_text_only
            raw = generate_text_only(
                model_config=self.model_config,
                messages=messages,
                mode=self.mode,
                max_new_tokens=1024,
                json_mode=True,
            )
            if self.verbose:
                print(f"  [Planner] 原始输出: {raw[:200]!r}")
            return {"role": "assistant", "content": raw}
        except Exception as e:
            if self.verbose:
                print(f"  [debug] _llm_plan 异常: {e}")
            return {"role": "assistant", "content": ""}

    def _llm_generate_code(self, prompt: str) -> str:
        """让 LLM 生成代码（用于 DAG 节点参数填充）。"""
        import re as _re
        from task_executor import _pick_mock_code
        messages = [
            {"role": "system", "content": (
                "You are a Python code generator. "
                "Output ONLY raw Python code. "
                "No explanations, no markdown, no json, no tool_calls. "
                "Just the Python code itself."
            )},
            {"role": "user", "content": prompt},
        ]
        try:
            result = generate_ai_message(
                model_config=self.model_config,
                messages=messages,
                tools_schema=[],  # 不需要工具
                mode=self.mode,
            )
            content = result["ai_message"].get("content", "").strip()
            print(f"  [debug] _llm_generate_code 第1次输出: {repr(content[:120])}")
            # 去掉 ```python ... ``` 包裹
            m = _re.search(r"```(?:python)?\s*(.*?)```", content, _re.DOTALL)
            if m:
                content = m.group(1).strip()
            # 如果内容不像代码（太短、或包含 JSON 特征）就用更强的 prompt 重试一次
            looks_like_json = content.startswith("{") or ("tool_calls" in content)
            too_short = len(content) < 30
            if looks_like_json or too_short:
                if self.verbose:
                    print(f"  [debug] LLM 代码生成内容无效({len(content)} chars)，原始输出: {repr(content[:80])}，尝试重生成")
                # 第二次尝试：更直接的指令，强制不输出解释
                retry_prompt = (
                    f"TASK: {prompt}\n\n"
                    "OUTPUT PURE PYTHON CODE ONLY. No markdown, no explanation, no JSON.\n"
                    "Start your output directly with Python code (e.g. def ... or import ...)."
                )
                retry_msgs = [
                    {"role": "system", "content": "You are a Python code generator. Output ONLY valid Python code, nothing else."},
                    {"role": "user", "content": retry_prompt},
                ]
                try:
                    from b4_local_agent_llm import generate_text_only
                    raw2 = generate_text_only(
                        model_config=self.model_config,
                        messages=retry_msgs,
                        mode=self.mode,
                        max_new_tokens=512,
                    )
                    print(f"  [debug] generate_text_only 原始输出: {repr(raw2[:120])}")
                    m2 = _re.search(r"```(?:python)?\s*(.*?)```", raw2, _re.DOTALL)
                    content = m2.group(1).strip() if m2 else raw2.strip()
                    if len(content) < 30:
                        # 重试仍然失败：抛出异常让节点失败，不要静默写入垃圾代码
                        raise ValueError(f"LLM 尚无法生成有效代码，输出内容：{raw2[:80]!r}")
                    return content
                except Exception as e2:
                    raise ValueError(f"LLM 代码生成失败，不应退化为 mock：{e2}") from e2
            return content
        except Exception as e:
            if self.verbose:
                print(f"  [debug] _llm_generate_code 异常: {e}，退化到 mock")
            return _pick_mock_code(prompt, "")

    def _generate(self, messages: list[dict], stage: str = "execute") -> dict:
        """
        Call LLM backend.

        stage: "plan" | "execute" | "summarize"
          - plan:      如果有 model_roster，用 fast 配置（小步分解不需要太高质量）
          - execute:   用 default 配置（工具调用阶段）
          - summarize: 用 strict 配置（temperature=0，确保答案稳定）
        """
        if self.mode == "mock" or not _B4_OK:
            return _mock_generate_local(messages, self.tools_schema)

        # 按 stage 选择 model_config
        model_cfg = self.model_config
        try:
            from b4_model_switch import load_model_roster, select_model_for_task
            roster_path = str(_PROJECT_ROOT / "configs" / "model_roster.yaml")
            roster = load_model_roster(roster_path)
            selected_cfg = select_model_for_task(stage, roster)
            # select_model_for_task 返回的是相对路径（如 model.yaml），需要拼接
            selected_path = _PROJECT_ROOT / "configs" / selected_cfg
            if selected_path.exists():
                model_cfg = str(selected_path)
                if self.verbose:
                    print(C("gray", f"  [model_switch] stage={stage} → {selected_cfg}"))
        except Exception as e:
            if self.verbose:
                print(C("gray", f"  [model_switch] 跳过（{e}），用默认配置"))

        result = generate_ai_message(
            model_config=model_cfg,
            messages=messages,
            tools_schema=self.tools_schema,
            mode=self.mode,
            artifact_dir=str(_PROJECT_ROOT / "outputs" / "hal_chat_llm_calls"),
            artifact_stem=f"turn_{len(messages):03d}",
        )
        if self.verbose or result["ai_message"].get("content") == PARSE_ERROR_CONTENT:
            raw_path = _PROJECT_ROOT / "outputs" / "hal_chat_llm_calls" / f"turn_{len(messages):03d}_raw_model_output.json"
            if raw_path.exists():
                import json as _j
                raw = _j.loads(raw_path.read_text())
                raw_text = raw.get("raw_text", "")[:500]
                print(C("gray", f"  [debug] raw_text: {raw_text!r}"))
                print(C("gray", f"  [debug] artifact: {raw_path}"))
        return result["ai_message"]

    # ------------------------------------------------------------------
    # Meta-commands
    # ------------------------------------------------------------------
    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Return True if handled."""
        # Normalize: treat /undo 2 etc as starting with /undo
        cmd = cmd.strip()
        if cmd in ("/quit", "/exit"):
            print(C("blue", "[系统] 正在保存会话..."))
            p = self._save_session()
            print(C("blue", f"[系统] 再见！会话已保存到 {p}"))
            sys.exit(0)

        elif cmd == "/help":
            help_text = f"""{C('bold', '可用命令：')}
  {C('cyan', '/help')}              显示此帮助
  {C('cyan', '/tools')}             列出当前可用工具
  {C('cyan', '/clear')}             清空对话历史（保留 system prompt）
  {C('cyan', '/save')}              手动保存当前会话
  {C('cyan', '/history')}           显示对话历史（简洁格式）
  {C('cyan', '/mode mock|prompt_json')}  切换 LLM 模式
  {C('cyan', '/resume <session_id>')}  切换到指定会话（也可用 --resume session_id）
  {C('cyan', '/auto_confirm on|off')}  开关 DAG 任务计划人工确认（当前: {'on' if self.auto_confirm else 'off'}）
  {C('cyan', '/branch')}            从当前会话创建分支（复制消息，保存到 sessions/）
  {C('cyan', '/branch list')}       列出当前会话的所有分支
  {C('cyan', '/undo')}              撤销最后一轮对话
  {C('cyan', '/undo N')}            撤销最近 N 轮对话
  {C('cyan', '/quit')}              退出（自动保存）

  {C('gray', '在「任务计划确认」提示符处输入 /stop 可单步停止当前任务，返回 User >。')}
"""
            print(help_text)

        elif cmd.startswith("/auto_confirm"):
            arg = cmd[len("/auto_confirm"):].strip().lower()
            if arg in ("on", "true", "1", "开"):
                self.auto_confirm = True
                print(C("blue", "[系统] 任务计划人工确认：已关闭（自动执行）"))
            elif arg in ("off", "false", "0", "关"):
                self.auto_confirm = False
                print(C("blue", "[系统] 任务计划人工确认：已开启"))
            else:
                print(C("blue", f"[系统] 当前任务计划人工确认: {'关闭(自动)' if self.auto_confirm else '开启'}（用 /auto_confirm on|off 切换）"))

        elif cmd == "/tools":
            if self._tool_names:
                print(C("blue", f"[系统] 当前可用工具（{self.toolset}）：") + C("cyan", ", ".join(self._tool_names)))
                if self._cache is not None:
                    stats = self._cache.stats()
                    hit_rate = (
                        f"{self._cache_hits/(self._cache_hits+self._cache_misses)*100:.0f}%"
                        if (self._cache_hits + self._cache_misses) > 0 else "n/a"
                    )
                    print(C("gray", f"[缓存] {stats['size']}/{stats['max_entries']} 条目 | "
                                    f"命中 {self._cache_hits} 次 | "
                                    f"未命中 {self._cache_misses} 次 | "
                                    f"命中率 {hit_rate}"))
            else:
                print(C("yellow", "[系统] 没有可用工具（工具配置未加载）"))

        elif cmd == "/clear":
            system_msgs = [m for m in self.messages if m.get("role") == "system"]
            self.messages = system_msgs
            self.tool_rounds = 0
            print(C("blue", "[系统] 对话历史已清空（系统提示词已保留）"))

        elif cmd == "/save":
            p = self._save_session()
            print(C("blue", f"[系统] 会话已保存到 {p}"))

        elif cmd == "/history":
            non_system = [m for m in self.messages if m.get("role") != "system"]
            if not non_system:
                print(C("gray", "[系统] 对话历史为空"))
            else:
                print(C("bold", f"[系统] 对话历史（共 {len(non_system)} 条消息）："))
                for i, m in enumerate(non_system):
                    role = m.get("role", "?")
                    content = (m.get("content") or "")[:80]
                    tc = m.get("tool_calls", [])
                    # 显示友好名称
                    display_role = {"user": "你", "assistant": "HAL1000", "tool": "工具"}.get(role, role)
                    role_color = {"user": "cyan", "assistant": "green", "tool": "magenta"}.get(role, "gray")
                    tc_info = f" [工具: {', '.join(c.get('name', '?') for c in tc)}]" if tc else ""
                    print(f"  {C('gray', str(i+1)+'.')} {C(role_color, display_role)}: {content}{C('yellow', tc_info)}")

        elif cmd.startswith("/mode "):
            new_mode = cmd[6:].strip()
            if new_mode in ("mock", "prompt_json"):
                self.mode = new_mode
                print(C("blue", f"[系统] 模式已切换为: {C('bold', new_mode)}"))
            else:
                print(C("red", f"[错误] 无效模式: {new_mode!r}（可选: mock, prompt_json）"))

        elif cmd == "/branch" or cmd == "/branch list":
            self._handle_branch(cmd)

        elif cmd.startswith("/undo"):
            self._handle_undo(cmd)

        else:
            print(C("red", f"[错误] 未知命令: {cmd}（输入 /help 查看帮助）"))

        return True

    # ------------------------------------------------------------------
    # Branch & Undo commands
    # ------------------------------------------------------------------

    def _handle_branch(self, cmd: str) -> None:
        """处理 /branch 和 /branch list 命令。"""
        if cmd == "/branch list":
            # 列出本会话的所有分支
            branch_files = list(_SESSION_DIR.glob("session_*.json"))
            branches = []
            for f in branch_files:
                try:
                    data = read_json(f)
                    if data.get("parent_id") == self.session_id:
                        branches.append(data)
                except Exception:
                    continue
            if not branches:
                print(C("blue", f"[分支] 当前会话 {self.session_id} 暂无分支"))
            else:
                print(C("blue", f"[分支] 当前会话 {self.session_id} 的分支列表（共 {len(branches)} 个）："))
                for b in branches:
                    sid = b.get("session_id", "?")
                    saved_at = b.get("saved_at", "?")
                    msg_count = len(b.get("messages", []))
                    print(C("cyan", f"  - {sid}") + C("gray", f"  保存时间: {saved_at}  消息数: {msg_count}"))
            return

        # /branch — 创建分支会话
        # 先保存当前主线
        self._save_session()

        # 新建分支 session_id，复制当前消息
        branch_session_id = _make_session_id()
        branch_messages = [m.copy() for m in self.messages]

        # 持久化分支会话到 sessions/
        _save_session_to_file(
            session_id=branch_session_id,
            messages=branch_messages,
            tool_rounds=self.tool_rounds,
            mode=self.mode,
            created_at=now_iso(),
            parent_id=self.session_id,
        )
        print(C("blue", f"[分支] 已创建分支会话 {branch_session_id}，当前继续主线 {self.session_id}"))
        print(C("gray", f"  分支已保存到 {_session_path(branch_session_id)}"))
        print(C("gray", f"  使用 'python hal_chat.py --resume {branch_session_id}' 进入分支"))

    def _handle_undo(self, cmd: str) -> None:
        """
        处理 /undo 和 /undo N 命令。
        从 self.messages 末尾弹，每轮弹到一条 role=='user' 为止（保留 system prompt）。
        """
        parts = cmd.strip().split()
        try:
            n_rounds = int(parts[1]) if len(parts) >= 2 else 1
        except (ValueError, IndexError):
            n_rounds = 1

        if n_rounds < 1:
            print(C("red", "[错误] /undo N 中 N 必须 >= 1"))
            return

        rounds_done = 0
        for _ in range(n_rounds):
            # 找到最后一条 user 消息（排除 system prompt）
            last_user_idx = None
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            # 如果没有 user 消息（或只剩 system prompt），停止
            if last_user_idx is None:
                print(C("yellow", "[撤销] 已无可撤销的对话轮次"))
                break

            # 弹掉 last_user_idx 及其后所有消息
            self.messages = self.messages[:last_user_idx]
            rounds_done += 1

        if rounds_done > 0:
            non_system = [m for m in self.messages if m.get("role") != "system"]
            print(C("blue", f"[撤销] 已撤销最后 {rounds_done} 轮，现在可以重新输入"))
            print(C("gray", f"  剩余消息数（含 system）: {len(self.messages)}，非 system: {len(non_system)}"))
            self._save_session()
        else:
            print(C("yellow", "[撤销] 没有可撤销的轮次"))

    # ------------------------------------------------------------------
    # Main REPL loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        # Banner
        print()
        print(C("bold", "=" * 56))
        print(C("cyan", "  HAL1000 — 本地 Agent 终端对话"))
        print(C("gray", f"  会话 ID: {self.session_id}"))
        print(C("gray", f"  模式: {self.mode}  |  工具集: {self.toolset}  |  最大轮次: {self.max_turns}"))
        print(C("bold", "=" * 56))
        print(C("blue", "  输入 /help 查看命令，/quit 退出，计划确认时输入 /stop 可中断当前任务"))
        print()

        while True:
            try:
                user_input = input(C("cyan", "User > "))
            except EOFError:
                print()
                print(C("blue", "[系统] 输入流结束，正在保存并退出..."))
                p = self._save_session()
                print(C("blue", f"[系统] 再见！会话已保存到 {p}"))
                break
            except KeyboardInterrupt:
                # Ctrl+C 在等待输入时：退出
                print()
                print(C("blue", "[系统] 正在保存并退出..."))
                self._save_session()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # 允许用户在 REPL 里输入 --resume session_id 或 /resume session_id
            if user_input.startswith("--resume ") or user_input.startswith("/resume "):
                sid = user_input.split(None, 1)[1].strip()
                data = _load_session(sid)
                if data:
                    loaded = data.get("messages", [])
                    self.messages = loaded
                    self.session_id = data.get("session_id", sid)
                    print(C("blue", f"[系统] 已切换到会话 {self.session_id}（{len(loaded)} 条消息）"))
                else:
                    print(C("red", f"[错误] 找不到会话: {sid}。指定的是 session_XXXX 格式的 ID"))
                continue

            if user_input.startswith("/"):
                if user_input == "/stop":
                    # 在等待输入时输入 /stop：就是下一条新指令前什么也不要做
                    print(C("yellow", "  [stop] 没有正在运行的任务。直接输入新指令。"))
                    continue
                self._handle_command(user_input)
                continue

            # Normal user message
            try:
                self._run_agent_turn(user_input)
            except InterruptedError:
                print(C("yellow", "  [stop] 任务已停止。输入新指令继续。"))
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
            except Exception as exc:
                print(C("red", f"[错误] Agent 执行异常: {type(exc).__name__}: {exc}"))
                if self.verbose:
                    import traceback
                    traceback.print_exc()

            # Auto-save after each turn
            try:
                self._save_session()
            except Exception:
                pass
            print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dag_summary(dag) -> str:
    """从 DAG 执行结果直接合成汇总文本。

    强制包含三部分：
      1. 代码执行输出（Step 1 结果）
      2. 文件真实路径（Step 2 结果）
      3. 验证结论（Step 3 结果 对比 Step 1）
    """
    parts: list[str] = []

    # 收集各类型节点的结果
    code_results: list[dict] = []   # 按顺序：[step1_out, step3_out]
    file_result: dict | None = None
    has_any_failure = False

    for node in dag.nodes:
        result = node.result or {}
        status = result.get("status", "error")
        out = result.get("output") or {}

        if status != "success":
            has_any_failure = True
            err = result.get("error") or {}
            errmsg = err.get("message", "unknown") if isinstance(err, dict) else str(err)
            parts.append(f"❌ **{node.description}** 失败: {errmsg}")
            continue

        if node.tool_name == "code_executor":
            code_results.append({"node": node, "out": out})
        elif node.tool_name == "file_writer":
            file_result = {"node": node, "out": out}
        elif node.tool_name == "file_reader":
            content = out.get("content", "")
            nc = out.get("num_chars", len(content))
            parts.append(f"📄 **{node.description}**（{nc} 字符）:\n{content[:300]}")
        elif node.tool_name == "calculator":
            parts.append(f"🔢 **计算结果**: {out.get('result')}")

    # ── 第一部分：代码执行输出（Step 1）────────────────
    if code_results:
        first = code_results[0]
        rc = first["out"].get("returncode", -1)
        stdout = first["out"].get("stdout", "").strip()
        elapsed = first["out"].get("elapsed_ms", 0)
        if rc == 0:
            parts.append(
                f"💻 **代码执行输出**（{elapsed:.0f}ms）\n"
                f"```\n{stdout}\n```"
            )
        else:
            stderr = first["out"].get("stderr", "")[:200]
            parts.append(f"💻 **代码执行失败** (exit={rc})\nstderr: {stderr}")

    # ── 第二部分：文件真实路径（Step 2）────────────────
    if file_result:
        written = file_result["out"].get("written_path", "?")
        nb = file_result["out"].get("num_bytes", 0)
        nl = file_result["out"].get("num_lines", 0)
        parts.append(
            f"💾 **代码已自动保存**（{nl} 行, {nb} 字节）\n"
            f"📁 文件地址: `{written}`"
        )
    else:
        # 只有读文件/计算等非代码任务时不显示警告
        if code_results:
            parts.append("⚠️ 没有可用的 file_writer 工具，代码未保存到文件")

    # ── 第三部分：验证结论（Step 3 对比 Step 1）───────────
    if len(code_results) >= 2:
        step1_rc = code_results[0]["out"].get("returncode", -1)
        step3_rc = code_results[1]["out"].get("returncode", -1)
        step1_out = code_results[0]["out"].get("stdout", "").strip()
        step3_out = code_results[1]["out"].get("stdout", "").strip()

        if step3_rc == 0:
            if step1_out == step3_out:
                verdict = "✅ **验证通过**：两次执行结果一致，代码正确。"
            else:
                verdict = (
                    f"✅ **验证通过**：从文件重新执行成功（exit=0）。\n"
                    f"输出第二次：```\n{step3_out}\n```"
                )
        else:
            stderr3 = code_results[1]["out"].get("stderr", "")[:200]
            verdict = f"❌ **验证失败**：从文件执行出错 (exit={step3_rc})\nstderr: {stderr3}"
        parts.append(verdict)
    elif len(code_results) == 1:
        rc = code_results[0]["out"].get("returncode", -1)
        if rc == 0:
            parts.append("✅ **验证**：代码运行成功 (exit=0)。")
        else:
            parts.append(f"❌ **验证失败** (exit={rc})。")

    if not parts:
        return "所有任务均失败，请检查输入或工具配置。"
    return "代码任务已全部完成：\n\n" + "\n\n".join(parts)


def _output_summary(tool_name: str, output: dict) -> str:
    """Return a short human-readable summary of tool output."""
    if tool_name == "file_reader":
        num = output.get("num_chars", 0)
        return f"{num} chars"
    elif tool_name == "calculator":
        return f"result={output.get('result')}"
    elif tool_name == "local_file_search":
        n = len(output.get("results", []))
        return f"{n} results"
    elif tool_name == "table_analyzer":
        shape = output.get("shape")
        return f"shape={shape}"
    elif tool_name == "format_converter":
        n = len(output.get("content", ""))
        return f"{n} chars"
    elif tool_name == "code_executor":
        rc = output.get("returncode", -1)
        elapsed = output.get("elapsed_ms", 0)
        stdout_lines = len(output.get("stdout", "").strip().splitlines())
        if rc == 0:
            return f"exit=0, {stdout_lines} lines output ({elapsed:.0f}ms)"
        else:
            stderr_snippet = output.get("stderr", "")[:60].replace("\n", " ")
            return f"exit={rc}, stderr: {stderr_snippet}"
    elif tool_name == "file_writer":
        written = output.get("written_path", "?")
        nb = output.get("num_bytes", 0)
        nl = output.get("num_lines", 0)
        ow = " (overwritten)" if output.get("overwritten") else ""
        return f"{nl} lines, {nb} bytes → {written}{ow}"
    else:
        return str(output)[:60]


def _mock_skill_result(tool_name: str, args: dict) -> dict:
    """Fallback mock skill execution when b3 not available."""
    if tool_name == "file_reader":
        return make_skill_result(
            tool_name, "success", args,
            {
                "content": (
                    "Agent 系统通常由模型、工具、记忆和执行循环组成。\n"
                    "工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答。\n"
                    "Memory 为 Agent 提供全局知识和历史对话上下文。"
                ),
                "num_chars": 85,
                "source": args.get("path", "unknown"),
                "truncated": False,
            },
            None, 1.0
        )
    elif tool_name == "calculator":
        expr = args.get("expression", "0")
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
        except Exception:
            result = 0
        return make_skill_result(tool_name, "success", args, {"result": result}, None, 0.5)
    else:
        return make_skill_result(
            tool_name, "error", args, None,
            {"type": "NotImplemented", "message": f"Mock {tool_name} not implemented"}, 0.1
        )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HAL1000 多轮终端对话 REPL Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_path", default=None,
                   help="模型路径（优先级低于 --model_config），默认读 HAL_MODEL_PATH 环境变量")
    p.add_argument("--model_config", default=None,
                   help="直接指定 model.yaml（优先于 model_path）")
    p.add_argument("--tools_config", default=None,
                   help="工具配置 YAML，默认 ../configs/tools.yaml")
    p.add_argument("--memory_config", default=None,
                   help="记忆配置（保留参数，暂未使用）")
    p.add_argument("--toolset", default="basic_tools",
                   help="工具集名称，默认 basic_tools")
    p.add_argument("--mode", choices=["mock", "prompt_json"], default="mock",
                   help="LLM 模式，默认 mock")
    p.add_argument("--resume", default=None, metavar="SESSION_ID",
                   help="恢复历史会话")
    p.add_argument("--system_prompt", default=None,
                   help="自定义系统提示词文件路径")
    p.add_argument("--max_turns", type=int, default=10,
                   help="最大工具调用轮次，默认 10")
    p.add_argument("--verbose", action="store_true",
                   help="打印调试信息")
    p.add_argument("--auto_confirm", action="store_true",
                   help="跳过 DAG 任务计划人工确认，自动执行（批量/非交互场景用）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve model config
    model_config = args.model_config
    if model_config is None:
        model_path = args.model_path or os.environ.get("HAL_MODEL_PATH")
        if model_path:
            # Look for model.yaml in that directory
            candidate = Path(model_path) / "model.yaml"
            if candidate.exists():
                model_config = str(candidate)
        if model_config is None:
            model_config = str(_PROJECT_ROOT / "configs" / "model.yaml")

    # Resume session
    resume_messages = None
    resume_tool_rounds = 0
    session_id = None
    if args.resume:
        data = _load_session(args.resume)
        if data:
            resume_messages = data.get("messages", [])
            resume_tool_rounds = data.get("tool_rounds", 0)
            session_id = data.get("session_id", args.resume)
            print(C("blue", f"[系统] 已恢复会话 {session_id}（{len(resume_messages)} 条消息）"))
        else:
            print(C("red", f"[错误] 找不到会话: {args.resume}"))
            return 1

    chat = HALChat(
        mode=args.mode,
        tools_config=args.tools_config,
        model_config=model_config,
        toolset=args.toolset,
        max_turns=args.max_turns,
        verbose=args.verbose,
        system_prompt=args.system_prompt,
        session_id=session_id,
        resume_messages=resume_messages,
        resume_tool_rounds=resume_tool_rounds,
        auto_confirm=args.auto_confirm,
    )
    chat.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
