"""
task_executor.py — DAG 并发任务执行器

按 TaskDAG 执行：
- ready_nodes() 立即并发启动（各自一个子进程）
- 主进程 poll 循环（50ms）检查子进程完成情况
- 完成后解锁依赖它的节点，继续启动
- 参数占位符替换：
    __GENERATE__  → 调用 llm_generate_fn 生成代码，或用 mock 代码
    __FROM_CODE__ → 从上一个 code_executor 节点的 args_template["code"] 取代码内容
    __FROM_FILE__ → 从 file_writer 节点的 args_template["path"] 读文件内容
"""
from __future__ import annotations

import json
import multiprocessing
import re
import select
import sys
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

# 与 hal_chat 共享：DAG 执行期间后台线程监听 stdin，输入 /stop 则设此 flag
_DAG_STOP_REQUESTED: bool = False
_DAG_WATCHER_STOP: threading.Event = threading.Event()  # 通知 watcher 线程退出


def _stdin_watcher_thread(stop_event: threading.Event):
    """后台线程：监听 stdin，读到 /stop 就设标志。stop_event 置位时退出。"""
    global _DAG_STOP_REQUESTED
    import sys, select
    while not stop_event.is_set():
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if stop_event.is_set():
                return
            if r:
                line = sys.stdin.readline()
                if line.strip() == "/stop":
                    _DAG_STOP_REQUESTED = True
                    print("\n  \033[93m[stop] 收到停止指令，待当前节点完成后退出...\033[0m")
                    return
        except Exception:
            return

# ---------------------------------------------------------------------------
# ANSI 颜色（与 hal_chat 一致）
# ---------------------------------------------------------------------------
_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "cyan":    "\033[96m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "red":     "\033[91m",
    "gray":    "\033[90m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
}


def C(color: str, text: str) -> str:
    return f"{_C.get(color, '')}{text}{_C['reset']}"


# ---------------------------------------------------------------------------
# Mock 代码模板（按关键词选）
# ---------------------------------------------------------------------------
_MOCK_CODE_TEMPLATES = {
    "冒泡": """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
print('原始数组:', arr)
print('排序结果:', bubble_sort(arr[:]))
""",
    "fibonacci": """\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for i in range(10):
    print(f'fib({i}) = {fib(i)}')
""",
    "prime": """\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 50) if is_prime(i)]
print('50以内质数:', primes)
""",
}


def _pick_mock_code(description: str, user_input: str) -> str:
    combined = (description + user_input).lower()
    for kw, code in _MOCK_CODE_TEMPLATES.items():
        if kw in combined:
            return code
    return "result = list(range(1, 6))\nprint('result:', result)\n"

# ---------------------------------------------------------------------------
# 三阶段验证代码生成
# ---------------------------------------------------------------------------

# 静态检测：这些模式暗示代码会在验证时永久阻塞/挂起
# 分三类：(1)需要事件循环的库 (2)显式无限循环 (3)等待外部输入/连接
_BLOCKING_PATTERNS = [
    # 需要事件循环或显示设备的库
    r"pygame\.", r"tkinter", r"wx\.", r"PyQt", r"PySide", r"kivy",
    r"turtle\.", r"curses\.", r"cv2\.", r"matplotlib\.pyplot\.show",
    # 显式无限循环
    r"while\s+True", r"for\s+\w+\s+in\s+iter\(",
    # 等待外部输入或连接
    r"socket\.listen", r"server\.serve_forever",
    r"input\s*\(",
    r"asyncio\.run", r"loop\.run_forever",
]

def _is_blocking(code: str) -> bool:
    """静态判断代码在验证环境中是否会永久阻塞，无法自动退出。"""
    import re
    return any(re.search(pat, code) for pat in _BLOCKING_PATTERNS)

def _make_verify_code(file_path: str, file_code: str) -> str:
    """
    生成验证代码（三阶段）：
      Phase 1: 语法检查（必做，失败立刻报错退出）
      Phase 2: 如果是阻塞型代码 → 直接放行，打印提示
      Phase 3: 尝试运行（短 timeout 3s）
               → 成功：验证通过
               → 失败/超时：降级，打印原因，放行

    注意：Phase 3 的超时由外层 code_executor timeout 控制（设为短值），
    实际的"再 poll 更长时间"由框架调度层决定（见 _run_verify_node）。
    """
    path_repr = repr(file_path)

    if _is_blocking(file_code):
        # 阻塞型：只做语法检查
        return (
            "import ast, sys\n"
            f"_path = {path_repr}\n"
            "_src = open(_path, encoding=\'utf-8\').read()\n"
            "try:\n"
            "    ast.parse(_src)\n"
            "    _lines = len(_src.splitlines())\n"
            "    print(f\'[验证] 语法检查通过: {_path} ({_lines} 行)\')\n"
            "    print('[验证] 检测到阻塞型代码（会永久阻塞），跳过运行验证，语法已通过')\n"
            "except SyntaxError as _e:\n"
            "    print(f\'[验证] 语法错误: {_e}\', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
    else:
        # 非阻塞型：语法检查 + 用 exec() 在当前进程内运行（避免触发 subprocess 黑名单）
        # 外层 code_executor 的 timeout 会整体 kill，充当运行超时保护
        return (
            "import ast, sys, io, traceback\n"
            f"_path = {path_repr}\n"
            "_src = open(_path, encoding=\'utf-8\').read()\n"
            "# Phase 1: 语法检查\n"
            "try:\n"
            "    _tree = ast.parse(_src)\n"
            "except SyntaxError as _e:\n"
            "    print(f\'[验证] 语法错误: {_e}\', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "print(f\'[验证] 语法检查通过: {_path} ({len(_src.splitlines())} 行)\')\n"
            "# Phase 2: exec 运行（超时由外层 kill 控制）\n"
            "_stdout_buf = io.StringIO()\n"
            "_old_stdout = sys.stdout\n"
            "sys.stdout = _stdout_buf\n"
            "try:\n"
            "    exec(compile(_tree, _path, \'exec\'), {\'__name__\': \'__main__\', \'__file__\': _path})"
            "    sys.stdout = _old_stdout\n"
            "    _out = _stdout_buf.getvalue()\n"
            "    if not _out.strip():\n"
            "        _tops = [l for l in _src.splitlines() if l.strip() and not l[0] in (' ', '#') and not l.startswith(('def ', 'class ', 'import ', 'from ', '@'))]\n"
            "        _has_entry = bool(_tops) or 'if __name__' in _src\n"
            "        if not _has_entry:\n"
            "            print('[验证] 错误：代码只有定义没有入口（缺少 if __name__ 或顶层调用）', file=sys.stderr)\n"
            "            sys.exit(1)\n"
            "        else:\n"
            "            print('[验证] 运行成功（无可见输出）')\n"
            "    else:\n"
            "        print('[验证] 运行成功:')\n"
            "        print(_out[:1000])\n"
            "except Exception as _e:\n"
            "    sys.stdout = _old_stdout\n"
            "    print(f\'[验证] 运行失败: {type(_e).__name__}: {_e}\', file=sys.stderr)\\n"
            "    sys.exit(1)\n"
        )



# ---------------------------------------------------------------------------
# Worker（在子进程中运行）
# ---------------------------------------------------------------------------
def _worker(
    tool_name: str,
    args: dict,
    config_path: str,
    toolset: str,
    queue: multiprocessing.Queue,
) -> None:
    """子进程 worker：执行单个工具，结果放入 queue。"""
    try:
        import importlib
        import inspect
        from pathlib import Path as _P

        _here = _P(__file__).resolve().parent
        _root = _here.parent
        for p in [str(_here), str(_root / "skills")]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from common.schemas import make_skill_result
        from common.path_utils import resolve_from_file
        from b3_tool_layer import _load_tools_config, _resolve_toolset, _validate_args

        t0 = perf_counter()
        cfg_path, config = _load_tools_config(config_path)
        _, allowed = _resolve_toolset(config, toolset)
        data_root_setting = config.get("settings", {}).get("data_root", "../data")
        resolved_data_root = resolve_from_file(data_root_setting, cfg_path)
        definition = config["tools"].get(tool_name)
        if definition is None or tool_name not in allowed:
            raise ValueError(f"tool {tool_name!r} not in toolset {toolset!r}")
        _validate_args(args, definition)
        module = importlib.import_module(definition["module"])
        fn = getattr(module, definition["function"])
        kwargs = dict(args)
        sig = inspect.signature(fn)
        if "data_root" in sig.parameters:
            kwargs["data_root"] = str(resolved_data_root)
        output = fn(**kwargs)
        elapsed_ms = round((perf_counter() - t0) * 1000, 1)
        result = make_skill_result(tool_name, "success", args, output, None, elapsed_ms)
        queue.put(result)
    except Exception as exc:
        from common.schemas import make_skill_result
        result = make_skill_result(
            tool_name, "error", args, None,
            {"type": type(exc).__name__, "message": str(exc)},
            None,
        )
        queue.put(result)


# ---------------------------------------------------------------------------
# 占位符解析
# ---------------------------------------------------------------------------
def _resolve_args(node, dag, llm_generate_fn, verbose: bool) -> dict:
    """替换 args_template 里的占位符。"""
    args = dict(node.args_template)

    # __GENERATE__: 需要生成代码
    if args.get("code") == "__GENERATE__":
        if llm_generate_fn is not None:
            prompt = (
                f"请为以下任务编写Python代码，只输出代码，不要解释："
                f"{node.description}。用户原始需求：{dag.user_input}"
            )
            raw = llm_generate_fn(prompt)
            m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
            code = m.group(1).strip() if m else raw.strip()
            # 抒出太短或像 JSON 的垃圾输出，报错而不是静默写入 mock
            looks_like_json = code.startswith("{") or ("tool_calls" in code)
            too_short = len(code) < 30
            if looks_like_json or too_short:
                raise ValueError(
                    f"LLM 代码生成失败（输出 {len(code)} 字符，不像有效代码），"
                    f"\u62d2绝退化为 mock。原始输出: {raw[:80]!r}"
                )
            args["code"] = code
            # 检测入口点：如果只有函数/类定义，没有顶层可执行语句，直接报错触发重试
            _tops = [l for l in code.splitlines()
                     if l.strip() and not l[0] in (' ', '\t', '#')
                     and not l.startswith(('def ', 'class ', 'import ', 'from ', '@', '"""', "'''"))]
            _has_entry = bool(_tops) or 'if __name__' in code
            if not _has_entry:
                raise ValueError(
                    "生成的代码只有函数/类定义，缺少可运行的入口点 "
                    "(需要 if __name__ == '__main__': 或顶层调用语句)，"
                    "请重新生成包含完整运行逻辑的代码"
                )
        else:
            args["code"] = _pick_mock_code(node.description, dag.user_input)
        if verbose:
            print(C("gray", f"  [{node.task_id}] code({len(args['code'])} chars) ready"))

    # __FROM_CODE__: 从前驱 code_executor 节点取代码
    if args.get("content") == "__FROM_CODE__":
        for dep_id in node.depends_on:
            dep = next((n for n in dag.nodes if n.task_id == dep_id), None)
            if dep and dep.tool_name == "code_executor":
                code = dep.args_template.get("code", "")
                if code and code not in ("__GENERATE__", "__FROM_CODE__", "__FROM_FILE__"):
                    args["content"] = code
                    break
        if args.get("content") == "__FROM_CODE__":
            args["content"] = "# placeholder\nprint('no code')\n"

    # __FROM_FILE__: 从 file_writer 节点的 written_path 读文件，使用三阶段验证
    if args.get("code") == "__FROM_FILE__":
        for dep_id in node.depends_on:
            dep = next((n for n in dag.nodes if n.task_id == dep_id), None)
            if dep and dep.tool_name == "file_writer" and dep.result:
                written_path = dep.result.get("output", {}).get("written_path", "")
                if written_path:
                    try:
                        file_code = Path(written_path).read_text(encoding="utf-8")
                        # 替换为三阶段验证代码（由 _make_verify_code 生成）
                        args["code"] = _make_verify_code(written_path, file_code)
                    except Exception:
                        args["code"] = _pick_mock_code(node.description, dag.user_input)
                    break
        if args.get("code") == "__FROM_FILE__":
            # 没有 file_writer 前驱：尝试从 user_input / description 里提取已存在的文件路径
            _py_m = re.search(r'(?<![A-Za-z0-9_])/[^\s\'"]+\.py', dag.user_input)
            if not _py_m:
                _py_m = re.search(r'(?<![A-Za-z0-9_])/[^\s\'"]+\.py', node.description)
            if _py_m:
                _fpath = _py_m.group(0).strip()
                try:
                    file_code = Path(_fpath).read_text(encoding="utf-8")
                    args["code"] = _make_verify_code(_fpath, file_code)
                    if verbose:
                        print(C("gray", f"  [{node.task_id}] code_executor 从文件读取: {_fpath}"))
                except Exception as _fe:
                    args["code"] = _pick_mock_code(node.description, dag.user_input)
            else:
                args["code"] = _pick_mock_code(node.description, dag.user_input)

    # __REUSE__: 没有 file_writer 时，直接复用最近一个 code_executor 节点的代码
    if args.get("code") == "__REUSE__":
        for dep_id in reversed(node.depends_on):
            dep = next((n for n in dag.nodes if n.task_id == dep_id), None)
            if dep and dep.tool_name == "code_executor":
                reuse_code = dep.args_template.get("code", "")
                if reuse_code and reuse_code not in ("__GENERATE__", "__REUSE__", "__FROM_FILE__"):
                    args["code"] = reuse_code
                    break
        if args.get("code") == "__REUSE__":
            args["code"] = _pick_mock_code(node.description, dag.user_input)

    # file_reader path 为空时，从 node.description / dag.user_input 中自动提取
    if "path" in args and not args["path"] and node.tool_name == "file_reader":
        # 1. 优先从 user_input 里找绝对路径（最准确）
        extracted = ""
        abs_m = re.search(r'(?<![A-Za-z0-9_])/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,10}', dag.user_input)
        if abs_m:
            extracted = abs_m.group(0)
        # 2. 从 description 里找绝对路径
        if not extracted:
            abs_m2 = re.search(r'(?<![A-Za-z0-9_])/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,10}', node.description)
            if abs_m2:
                extracted = abs_m2.group(0)
        # 3. 从 description 里找相对路彤
        if not extracted:
            extracted = _extract_path_from_text(node.description)
        # 4. 从 user_input 里找相对路径
        if not extracted:
            extracted = _extract_path_from_text(dag.user_input)
        # 5. 最后尝试：recent_files
        if not extracted and getattr(dag, 'recent_files', None):
            import os as _os
            last_path = dag.recent_files[-1]
            data_root = str(Path(__file__).resolve().parent.parent / "data")
            if last_path.startswith(data_root):
                extracted = _os.path.relpath(last_path, data_root)
            else:
                extracted = _os.path.basename(last_path)
            if verbose:
                print(C("gray", f"  [{node.task_id}] file_reader path 从 recent_files 使用: {extracted}"))
        if extracted:
            args["path"] = extracted
            if verbose and not getattr(dag, '_printed_path_hint', False):
                print(C("gray", f"  [{node.task_id}] file_reader path 自动提取: {extracted}"))
        # 如果 path 是纯文件名（无路径分隔符），从前驱 local_file_search 结果里找完整路径
        _fp = args.get("path", "")
        if _fp and "/" not in _fp and not Path(_fp).is_absolute():
            for _dep_id in node.depends_on:
                _dep = next((n for n in dag.nodes if n.task_id == _dep_id), None)
                if _dep and _dep.tool_name == "local_file_search" and _dep.result:
                    _hits = _dep.result.get("output", {}).get("hits", [])
                    for _h in _hits:
                        _hpath = _h.get("path", "") if isinstance(_h, dict) else str(_h)
                        if _hpath.endswith(_fp) or Path(_hpath).name == _fp:
                            args["path"] = _hpath
                            if verbose:
                                print(C("gray", f"  [{node.task_id}] file_reader path 从 search 结果补全: {_hpath}"))
                            break
                if "/" in args.get("path", ""):
                    break
        # 如果还是空，留它为空，让 file_reader 自己报清楚的错误信息

    # image_qa path 为空时，从 description / user_input 里提取图片路径
    if "path" in args and not args["path"] and node.tool_name == "image_qa":
        # 只匹配 ASCII 路径字符（不含中文）
        img_ext = r'[A-Za-z0-9_./ -]+\.(?:png|jpg|jpeg|gif|bmp|webp)'
        m = re.search(img_ext, node.description, re.IGNORECASE)
        if not m:
            m = re.search(img_ext, dag.user_input, re.IGNORECASE)
        if m:
            raw_path = m.group(0).strip()
            # 如果包含 / 就从最后一个 / 左边开始截取（去掉前缀垃圾）
            if '/' in raw_path:
                slash_idx = raw_path.index('/')
                raw_path = raw_path[slash_idx:]
            args["path"] = raw_path
            if verbose:
                print(C("gray", f"  [{node.task_id}] image_qa path 自动提取: {args['path']}"))
        if not args.get("question"):
            args["question"] = dag.user_input or "请描述图片内容"

    # pdf_reader / docx_reader: path 为空时，从 user_input 里提取文件路径
    if node.tool_name in ("pdf_reader", "docx_reader") and not args.get("path"):
        _ext = ".pdf" if node.tool_name == "pdf_reader" else ".docx"
        # 匹配含中文的绝对路径，如 /root/siton-tmp/HAL1000/agent/data/科研训练学习总结.docx
        _pat_abs_unicode = r'(?<![A-Za-z0-9_])(/[^\s\"\']+('+re.escape(_ext)+r'))'
        # 匹配 ASCII 绝对路径
        _pat_abs = r"(?<![A-Za-z0-9_])(/[A-Za-z0-9_./ -]+" + re.escape(_ext) + r")"
        # 匹配含中文的相对路径或纯文件名，如 科研训练学习总结.docx
        _pat_rel = r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff./-]*" + re.escape(_ext)
        _m = re.search(_pat_abs_unicode, dag.user_input, re.IGNORECASE)
        if not _m:
            _m = re.search(_pat_abs, dag.user_input, re.IGNORECASE)
        if not _m:
            _m = re.search(_pat_rel, dag.user_input, re.IGNORECASE)
        if not _m:
            _m = re.search(_pat_abs_unicode, node.description, re.IGNORECASE)
        if not _m:
            _m = re.search(_pat_abs, node.description, re.IGNORECASE)
        if not _m:
            _m = re.search(_pat_rel, node.description, re.IGNORECASE)
        if _m:
            args["path"] = _m.group(0).strip()

    # local_file_search: 从 user_input 提取目录路径，如果用户是要“列出/浏览目录”就不要用描述作为 query
    # shell_exec: command 为空时从 description 兜底；同时修复常见的 xargs 路径错误
    if node.tool_name == "shell_exec":
        if not args.get("command"):
            # Planner 应该把具体命令写在 description 里；此处作为 fallback
            shell_cmds = ('ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc')
            desc_stripped = node.description.strip()
            if any(desc_stripped.startswith(c) for c in shell_cmds):
                args["command"] = desc_stripped
            else:
                dir_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./-]+)', dag.user_input)
                target = dir_m.group(1) if dir_m else "."
                args["command"] = f"ls -la {target}"
        else:
            # 修复 "ls /path/ | head -N | xargs cat" 没有目录前缀的问题
            # 错误: xargs cat 只拿到文件名，需要补成 xargs -I{} cat /path/{}
            m_bad = re.match(
                r'ls\s+(/[A-Za-z0-9_./-]+)/?\s*\|\s*(head\s+(?:-\d+|-n\s*\d+))\s*\|\s*xargs\s+(cat|head(?:\s+-\d+)?)$',
                args["command"].strip()
            )
            if m_bad:
                dir_path = m_bad.group(1).rstrip('/')
                head_part = m_bad.group(2)
                read_cmd = m_bad.group(3)
                fixed_cmd = f'ls {dir_path}/ | {head_part} | xargs -I{{}} {read_cmd} {dir_path}/{{}}'
                if verbose:
                    print(C("gray", f"  [{node.task_id}] xargs 路径修正: {args['command']!r} → {fixed_cmd!r}"))
                args["command"] = fixed_cmd
        if "workdir" not in args:
            args["workdir"] = "/"

    # local_file_search 已废弃，自动转换为 shell_exec
    if node.tool_name == "local_file_search":
        _lfs_root = args.get("root_dir", "") or ""
        if not _lfs_root:
            # 从 desc 或 user_input 提取路径
            _m = re.search(r'(/[A-Za-z0-9_./-]+)', node.description)
            if not _m: _m = re.search(r'(/[A-Za-z0-9_./-]+)', dag.user_input)
            _lfs_root = _m.group(1) if _m else "."
        node.tool_name = "shell_exec"
        _cmd = f"find {_lfs_root} -name '*.py' -o -name '*.md' -o -name '*.txt' | head -30"
        args = {"command": _cmd, "workdir": "/"}
        node.args_template = args
        if verbose:
            print(C("yellow", f"  [{node.task_id}] local_file_search → shell_exec: {_cmd}"))
    return args


def _extract_path_from_text(text: str) -> str:
    """从自然语言描述里提取文件路径。"""
    # 1. 绝对路径：找 /<ascii>。＜扩展名＞
    m_abs = re.search(r'(?<![A-Za-z0-9_])/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,10}', text)
    # 2. 相对路径：字母开头，含 / 和 .
    m_rel = re.search(r'[A-Za-z][A-Za-z0-9_./\-]*\.[A-Za-z0-9]{1,10}', text)
    if m_abs and m_rel:
        # 两者都找到：哪个在前面就用哪个
        return m_abs.group(0) if m_abs.start() <= m_rel.start() else m_rel.group(0)
    if m_abs:
        return m_abs.group(0)
    if m_rel:
        return m_rel.group(0)
    return ""


# ---------------------------------------------------------------------------
# 主执行函数
# ---------------------------------------------------------------------------
# 不可重试的错误类型（调用方逻辑错误，重试没意义）
_NODE_NO_RETRY_ERRORS = {"PathEscape", "ValidationError", "PermissionError", "FILE_NOT_FOUND", "INVALID_INPUT"}


def execute_dag(
    dag,
    tools_config: str,
    toolset: str,
    llm_generate_fn: Optional[Callable] = None,
    verbose: bool = False,
    poll_interval: float = 0.05,
    node_timeout: float = 30.0,
    node_max_retries: int = 2,
    node_retry_delays: tuple = (0.5, 1.0),
    tool_cache=None,
    cache_blacklist: set | None = None,
    cache_hit_counter: Optional[Callable] = None,
    cache_miss_counter: Optional[Callable] = None,
) -> object:
    """
    执行 TaskDAG。返回执行完毕的 dag（各节点 result 已填充）。
    tool_cache: ToolCache 实例（可选），命中时直接返回不开子进程
    节点失败时自动 retry（指数退避，最多 node_max_retries 次），
    不可重试错误（PathEscape/ValidationError/PermissionError/FILE_NOT_FOUND）直接失败。
    """
    _cache_bl = cache_blacklist or set()
    ctx = multiprocessing.get_context("fork")

    # 启动后台 stdin 监听线程，让用户可以在 DAG 执行期间输入 /stop
    global _DAG_STOP_REQUESTED
    _DAG_STOP_REQUESTED = False
    _watcher_stop = threading.Event()
    _watcher = threading.Thread(target=_stdin_watcher_thread, args=(_watcher_stop,), daemon=True)
    _watcher.start()

    # 运行中的子进程 {task_id: (Process, Queue, start_time)}
    running: dict = {}
    t_total = perf_counter()

    # 并发数统计
    concurrent = sum(1 for n in dag.nodes if not n.depends_on)
    print(C("blue", f"  [▶ DAG] 开始执行 {len(dag.nodes)} 个任务节点（{concurrent} 个可立即并发）"))

    while not dag.all_done():
        # ── 启动所有 ready 节点 ──────────────────────────────────
        for node in dag.ready_nodes():
            node.status = "running"
            try:
                args = _resolve_args(node, dag, llm_generate_fn, verbose)
            except ValueError as _resolve_err:
                # _resolve_args 内部生成代码时发现问题（如缺少入口点），
                # 把节点标记为 failed，让节点级 retry 重新生成代码
                from common.schemas import make_skill_result
                node.result = make_skill_result(
                    node.tool_name, "error", {},
                    None,
                    {"type": type(_resolve_err).__name__, "message": str(_resolve_err)},
                    0,
                )
                node.status = "pending"  # 重置为 pending 让 retry 能重新 resolve
                node._resolve_retry_count = getattr(node, "_resolve_retry_count", 0) + 1
                if node._resolve_retry_count <= 2:
                    print(C("yellow", f"  [resolve-retry {node._resolve_retry_count}/2 {node.task_id}] {_resolve_err}"))
                    import time as _time; _time.sleep(0.5 * node._resolve_retry_count)
                    continue
                else:
                    node.status = "failed"
                    print(C("red", f"  [✗ {node.task_id}] resolve 失败 3 次: {_resolve_err}"))
                    continue
            node.args_template = args

            # ── 缓存查找（副作用工具不缓存）─────────────────────
            if tool_cache is not None and node.tool_name not in _cache_bl:
                cached = tool_cache.get(node.tool_name, args)
                if cached is not None:
                    if cache_hit_counter:
                        cache_hit_counter()
                    node.result = cached["result"]
                    node.status = "done"
                    out = node.result.get("output") or {}
                    if node.tool_name == "file_reader":
                        summary = f"{out.get('num_chars', 0)} chars"
                    elif node.tool_name == "calculator":
                        summary = str(out.get("result", "?"))[:40]
                    else:
                        summary = str(out)[:40]
                    print(
                        C("yellow", f"  [✓ {node.task_id} {node.tool_name}]") +
                        C("magenta", f" {summary}") +
                        C("gray", " (0.0ms)") +
                        C("cyan", " [cache hit]")
                    )
                    continue
                else:
                    if cache_miss_counter:
                        cache_miss_counter()

            # image_qa 的视觉模型缓存在主进程全局变量，子进程 fork 后缓存丢失每次重载。
            # docx_reader / pdf_reader 子进程加载库耗时太長，且无副作用，也走主进程直接调用。
            if node.tool_name in ("image_qa", "docx_reader", "pdf_reader"):
                t_node = perf_counter()
                print(C("yellow", f"  [▶ {node.task_id} {node.tool_name}] {node.description}（主进程直接调用）"))
                try:
                    from b3_tool_layer import execute_tool_calls as _etc_dag
                    _results = _etc_dag(
                        [{"id": node.task_id, "name": node.tool_name, "args": args}],
                        tools_config, toolset,
                    )
                    _r = _results[0] if _results else {"status": "error", "error": {"type": "Empty", "message": "no result"}}
                except Exception as _e:
                    _r = {"status": "error", "error": {"type": type(_e).__name__, "message": str(_e)}}
                node.result = _r
                elapsed_ms = round((perf_counter() - t_node) * 1000, 1)
                if _r.get("status") == "success":
                    node.status = "done"
                    out = _r.get("output") or {}
                    print(C("green", f"  [✓ {node.task_id} {node.tool_name}]") + C("gray", f" ({elapsed_ms}ms)"))
                else:
                    node.status = "failed"
                    err = _r.get("error") or {}
                    print(C("red", f"  [✗ {node.task_id} {node.tool_name}] {err.get('message','unknown')}"))
                continue

            q: multiprocessing.Queue = ctx.Queue()
            p = ctx.Process(
                target=_worker,
                args=(node.tool_name, args, tools_config, toolset, q),
            )
            p.start()
            running[node.task_id] = (p, q, perf_counter())
            dep_desc = (
                f"（依赖: {', '.join(node.depends_on)}）"
                if node.depends_on
                else ""
            )
            print(C("yellow", f"  [▶ {node.task_id} {node.tool_name}] {node.description}{dep_desc}"))

        # ── poll：检查运行中的子进程 ─────────────────────────────
        for task_id in list(running.keys()):
            p, q, t_start = running[task_id]
            node = next(n for n in dag.nodes if n.task_id == task_id)

            # 超时检查
            elapsed = perf_counter() - t_start
            if elapsed > node_timeout:
                # ── 验证节点（code_executor + 描述含"验证"）：二次 poll ──
                is_verify = (
                    node.tool_name == "code_executor"
                    and any(kw in node.description.lower()
                            for kw in ["验证", "verify", "from file", "从文件"])
                )
                # 短超时（3s）后二次 poll 更长时间（15s）
                _SHORT = 3.0
                _LONG  = 15.0
                if is_verify and elapsed < _SHORT + _LONG and not getattr(node, '_second_poll', False):
                    if elapsed >= _SHORT:
                        # 进入二次 poll 阶段，打印一次提示
                        if not getattr(node, '_second_poll_notified', False):
                            node._second_poll_notified = True  # type: ignore
                            print(C("yellow", f"  [⏳ {task_id}] 初次超时({_SHORT:.0f}s)，延长等待({_LONG:.0f}s)中..."))
                    continue  # 继续等待，直到 _SHORT + _LONG 超时

                # 真正超时（或非验证节点）→ 降级处理
                p.terminate()
                p.join(timeout=2)
                del running[task_id]
                if is_verify:
                    # 验证节点超时 → 降级为"语法通过，运行超时"，不算失败
                    node.status = "done"
                    node.result = {
                        "status": "success",
                        "output": {
                            "stdout": f"[验证] 运行超时（>{node_timeout:.0f}s），降级为语法通过",
                            "stderr": "",
                            "returncode": 0,
                            "timed_out": True,
                            "elapsed_ms": round(elapsed * 1000, 1),
                        },
                        "error": None,
                    }
                    print(C("yellow", f"  [⚠ {task_id} {node.tool_name}] 运行超时，已降级（语法已通过）"))
                else:
                    # 普通节点超时 → 真失败
                    node.status = "failed"
                    node.result = {
                        "status": "error",
                        "output": None,
                        "error": {"type": "Timeout", "message": f"超时 ({node_timeout}s)"},
                    }
                    print(C("red", f"  [✗ {task_id} {node.tool_name}] 超时"))
                continue

            if not p.is_alive():
                p.join()
                try:
                    result = q.get_nowait()
                except Exception:
                    result = {
                        "status": "error",
                        "output": None,
                        "error": {"type": "Empty", "message": "子进程无结果"},
                    }
                node.result = result
                elapsed_ms = round((perf_counter() - t_start) * 1000, 1)

                if result.get("status") == "success":
                    out = result.get("output") or {}
                    # code_executor exit!=0 视为失败
                    # 例外：超时（timed_out=True）且是阻塞型代码（curses/pygame）→ 视为成功
                    if node.tool_name == "code_executor" and out.get("returncode", 0) != 0:
                        _is_timeout = out.get("timed_out", False) or out.get("returncode", 0) == -1
                        _code_str = node.args_template.get("code", "") if node.args_template else ""
                        _blocking = _is_timeout and _code_str and _is_blocking(_code_str)
                        if _blocking:
                            # 阻塞型代码超时是正常的，语法应该没问题
                            node.status = "done"
                            rc = out.get("returncode", -1)
                            print(
                                C("green", f"  [✓ {task_id} {node.tool_name}]")
                                + C("magenta", f" exit={rc} (阻塞型代码超时，语法已验证)")
                                + C("gray", f" ({elapsed_ms}ms)")
                            )
                        else:
                            node.status = "failed"
                            rc = out.get("returncode", -1)
                            stderr_preview = out.get("stderr", "")[:200]
                            errmsg = f"exit={rc}" + (f", stderr: {stderr_preview}" if stderr_preview else "")
                            print(C("red", f"  [✗ {task_id} {node.tool_name}] {errmsg} ({elapsed_ms}ms)"))
                            node.result = result
                    else:
                        node.status = "done"
                        # ── 写入缓存（副作用工具不缓存）──────────────
                        if tool_cache is not None and node.tool_name not in _cache_bl:
                            tool_cache.put(node.tool_name, args, result)
                        if node.tool_name == "code_executor":
                            rc = out.get("returncode", -1)
                            nlines = len(out.get("stdout", "").strip().splitlines())
                            summary = f"exit={rc}, {nlines} lines"
                        elif node.tool_name == "file_writer":
                            summary = f"{out.get('num_bytes', 0)} bytes → {out.get('written_path', '?')}"
                        elif node.tool_name == "file_reader":
                            summary = f"{out.get('num_chars', 0)} chars"
                        else:
                            summary = str(out)[:60]
                        print(
                            C("green", f"  [✓ {task_id} {node.tool_name}]")
                            + C("magenta", f" {summary}")
                            + C("gray", f" ({elapsed_ms}ms)")
                        )
                else:
                    err = result.get("error") or {}
                    err_type = err.get("type", "") if isinstance(err, dict) else ""
                    errmsg = err.get("message", "unknown") if isinstance(err, dict) else str(err)
                    # ── 节点级 retry（指数退避）──────────────────────
                    retry_count = getattr(node, '_retry_count', 0)
                    # file_reader 的 ValueError/FileNotFoundError 不重试：路径错了重试无意义
                    is_file_reader_logic_err = (
                        node.tool_name == "file_reader"
                        and err_type in ("ValueError", "FileNotFoundError")
                    )
                    if (err_type not in _NODE_NO_RETRY_ERRORS
                            and not is_file_reader_logic_err
                            and retry_count < node_max_retries):
                        node._retry_count = retry_count + 1  # type: ignore
                        wait = node_retry_delays[min(retry_count, len(node_retry_delays) - 1)]
                        print(C("yellow",
                            f"  [retry {node._retry_count}/{node_max_retries}"
                            f" {task_id} {node.tool_name}]"
                            f" 失败（{err_type or errmsg[:40]}），{wait:.1f}s 后重试..."))
                        time.sleep(wait)
                        # 重置为 pending，下一轮循环重新启动子进程
                        node.status = "pending"
                        node.result = None
                    else:
                        node.status = "failed"
                        node.result = result
                        retry_note = f"（已重试 {retry_count} 次）" if retry_count > 0 else ""
                        print(C("red", f"  [✗ {task_id} {node.tool_name}] {errmsg}{retry_note}"))

                del running[task_id]

        if not dag.all_done():
            # 检查用户是否输入了 /stop
            if _DAG_STOP_REQUESTED:
                # 杀掉所有运行中的子进程
                for tid, (p, q, _) in list(running.items()):
                    p.terminate()
                    print(C("yellow", f"  [stop] 已终止节点 {tid}"))
                _watcher_stop.set()   # 通知 watcher 线程退出，释放 stdin
                print(C("yellow", "  [stop] DAG 已中断，回到命令行"))
                raise InterruptedError("/stop")
            time.sleep(poll_interval)

    _watcher_stop.set()   # DAG 正常完成，通知 watcher 线程退出，释放 stdin
    total_ms = round((perf_counter() - t_total) * 1000, 1)
    done_count = sum(1 for n in dag.nodes if n.status == "done")
    fail_count = sum(1 for n in dag.nodes if n.status == "failed")
    if fail_count == 0:
        print(C("green", f"  [✓ DAG] 全部完成，耗时 {total_ms}ms"))
    else:
        print(C("yellow", f"  [DAG] 完成 {done_count}/{len(dag.nodes)}，失败 {fail_count}，耗时 {total_ms}ms"))
    return dag
