"""
task_planner.py — LLM 驱动的任务规划器

真实模型模式：调用 LLM（同一个 Qwen 模型，但用 task_planner.txt 的 system prompt）
  → 强制输出 JSON DAG
  → 解析失败自动 fallback 到空 DAG（走 FSM ReAct）

mock 模式：极简规则，只识别最明确的单任务，其余全走 FSM

设计原则：
  - 任务拆解是 LLM 的事，不是正则的事
  - Planner 和 Executor 是同一个模型的两个不同角色（不同 system prompt）
  - Planner 失败 = 不崩溃，直接 fallback 到 FSM（更保守，更鲁棒）
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PLANNER_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "task_planner.txt"


def _unique_fallback_path(ext: str = "py", subdir: str = "output") -> str:
    """
    生成唯一的典型默认文件名，避免多个任务都落到同一个固定路径上相互覆盖（如 output/result.py）。
    只在真正推断不出文件名时作为最后典型（正常情况下 LLM Planner 应该总是能推断出有意义的文件名）。
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{subdir}/task_{stamp}_{suffix}.{ext}"


# ---------------------------------------------------------------------------
# 数据结构（和 task_executor 共享）
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    task_id: str
    tool_name: str
    args_template: dict
    depends_on: list[str]
    description: str
    status: str = "pending"   # pending / running / done / failed
    result: Optional[dict] = None


@dataclass
class TaskDAG:
    nodes: list[TaskNode]
    user_input: str
    strategy: str = ""   # Planner 的一句话思路说明，用于人在环确认时展示
    recent_files: list = field(default_factory=list)  # 最近已写入的文件列表

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def all_done(self) -> bool:
        return all(n.status in ("done", "failed") for n in self.nodes)

    def ready_nodes(self) -> list[TaskNode]:
        done_ids = {n.task_id for n in self.nodes if n.status == "done"}
        return [
            n for n in self.nodes
            if n.status == "pending" and all(d in done_ids for d in n.depends_on)
        ]


# ---------------------------------------------------------------------------
# JSON → TaskDAG 转换
# ---------------------------------------------------------------------------

def _task_to_node(task: dict, user_input: str = "") -> Optional[TaskNode]:
    """把 Planner 输出的单个 task dict 转为 TaskNode。"""
    tid = task.get("id", "")
    tool = task.get("tool", "")
    desc = task.get("description", "")
    save_path = task.get("save_path", "")
    depends_on = task.get("depends_on", [])

    if not tid or not tool:
        return None

    # 根据 tool 类型构建 args_template
    if tool == "code_executor":
        args = {"code": "__GENERATE__", "language": "python"}
        # 判断是验证步骤（描述里有"验证"或"verify"）
        if any(kw in desc.lower() for kw in ["验证", "verify", "from file", "从文件"]):
            args["code"] = "__FROM_FILE__"
        # 检测到 user_input 里包含 .py 文件路径（用户想运行现有文件）
        elif re.search(r'(?<![A-Za-z0-9_])/[^\'"\s]+\.py', user_input):
            args["code"] = "__FROM_FILE__"
    elif tool == "file_writer":
        args = {
            "path": save_path or _unique_fallback_path("py"),
            "content": "__FROM_CODE__",
        }
    elif tool == "file_reader":
        file_path = ""
        if save_path and save_path.strip():
            file_path = save_path.strip()
        else:
            # 1. 含中文的绝对路径（最准）
            abs_m = re.search(r"(?<![A-Za-z0-9_])(/[^\\s\'\"]+\.[A-Za-z0-9]{1,10})", user_input)
            if abs_m:
                file_path = abs_m.group(1).strip()
            if not file_path:
                # 2. ASCII 绝对路径
                abs_m2 = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./ -]+\.[A-Za-z0-9]{1,10})', user_input)
                if abs_m2:
                    file_path = abs_m2.group(1).strip()
            if not file_path:
                # 3. 从 desc 找绝对路径
                abs_m3 = re.search(r"(?<![A-Za-z0-9_])(/[^\\s\'\"]+\.[A-Za-z0-9]{1,10})", desc)
                if abs_m3:
                    file_path = abs_m3.group(1).strip()
            if not file_path:
                # 4. 相对路径（支持中文）
                path_m = re.search(r'[\w\u4e00-\u9fff][\w\u4e00-\u9fff./-]*\.[A-Za-z0-9]{1,10}', desc)
                if path_m:
                    file_path = path_m.group(0)
        args = {"path": file_path, "max_chars": 5000}
    elif tool == "calculator":
        # 从描述里提取表达式
        expr_match = re.search(r'[\d\s+\-*/().]+', desc)
        args = {"expression": expr_match.group(0).strip() if expr_match else desc}
    elif tool == "local_file_search":
        # 从 desc / user_input 里提取目录路径
        dir_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./-]+)', desc)
        if not dir_m:
            dir_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./-]+)', user_input)
        root_dir = dir_m.group(1).rstrip('/') if dir_m else "."
        # 判断是列出目录还是搜索关键词
        list_kw = ['读取', '列出', '查看', '浏览', '显示', 'list', 'ls', 'dir']
        is_list = any(kw in desc or kw in user_input.lower() for kw in list_kw)
        query = "list" if is_list else desc
        args = {"query": query, "root_dir": root_dir, "top_k": 50}
    elif tool == "table_analyzer":
        args = {"path": save_path or "", "operation": "summary"}
    elif tool == "format_converter":
        args = {"path": save_path or "", "target_format": "json"}
    elif tool == "pdf_reader":
        # 从 user_input 或 description 里提取 PDF 路径
        _pdf_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./ -]+\.pdf)', user_input, re.IGNORECASE)
        if not _pdf_m:
            _pdf_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./ -]+\.pdf)', desc, re.IGNORECASE)
        if not _pdf_m:
            _pdf_m = re.search(r'[A-Za-z0-9_./ -]+\.pdf', user_input, re.IGNORECASE)
        _pdf_path = _pdf_m.group(0).strip() if _pdf_m else save_path or ""
        args = {"path": _pdf_path, "max_chars": 5000}
    elif tool == "docx_reader":
        # 从 user_input 或 description 里提取 docx 路径，支持中文文件名
        # 1. 含中文的绝对路径（最准确）
        _docx_m = re.search(r'(?<![A-Za-z0-9_])(/[^\s\'"]+\.docx)', user_input, re.IGNORECASE)
        if not _docx_m:
            # 2. ASCII 绝对路径
            _docx_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./ -]+\.docx)', user_input, re.IGNORECASE)
        if not _docx_m:
            # 3. 含中文的相对/纯文件名
            _docx_m = re.search(r'[\w\u4e00-\u9fff][\w\u4e00-\u9fff./-]*\.docx', user_input, re.IGNORECASE)
        if not _docx_m:
            # 4. 从 desc 里找
            _docx_m = re.search(r'(?<![A-Za-z0-9_])(/[^\s\'"]+\.docx)', desc, re.IGNORECASE)
        _docx_path = _docx_m.group(0).strip() if _docx_m else save_path or ""
        args = {"path": _docx_path, "max_chars": 5000}
    elif tool == "image_qa":
        # 从 description 或 save_path 里提取图片路径
        img_path = save_path or ""
        if not img_path:
            path_m = re.search(r'[A-Za-z0-9_./ -]+\.(?:png|jpg|jpeg|gif|bmp|webp)', desc, re.IGNORECASE)
            if path_m:
                raw = path_m.group(0).strip()
                if '/' in raw:
                    raw = raw[raw.index('/'):]
                img_path = raw
        # question 从 description 里提取，默认为简单描述
        question = desc if desc else "请描述这张图片的内容"
        args = {"path": img_path, "question": question}
    elif tool == "shell_exec":
        # Planner 的 description 字段直接就是 shell 命令（按提示词设计）
        # 如果 description 看起来是具体命令（以 ls/cat/grep/find/head/tail 开头），直接用
        # 否则从 user_input 提路径生成默认命令
        shell_cmds = ('ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'du', 'stat')
        desc_stripped = desc.strip()
        if any(desc_stripped.startswith(c) for c in shell_cmds):
            # description 就是命令
            cmd = desc_stripped
        else:
            # 从 user_input 提目录，生成默认 ls 命令
            dir_m = re.search(r'(?<![A-Za-z0-9_])(/[A-Za-z0-9_./-]+)', user_input)
            target_dir = dir_m.group(1) if dir_m else "."
            cmd = f"ls -la {target_dir}"
        args = {"command": cmd, "workdir": "/"}
    else:
        args = {}

    return TaskNode(
        task_id=tid,
        tool_name=tool,
        args_template=args,
        depends_on=depends_on,
        description=desc,
    )


def _parse_planner_output(raw: str, user_input: str) -> TaskDAG:
    """
    解析 LLM Planner 输出的 JSON。
    容错：提取第一个 {...} 块，解析失败返回空 DAG（走 FSM）。
    """
    # 去掉 markdown 代码块包裹
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw.strip(), flags=re.MULTILINE)

    # 找第一个完整 JSON 对象
    brace_start = raw.find('{')
    if brace_start == -1:
        return TaskDAG(nodes=[], user_input=user_input)

    # 尝试直接解析
    try:
        data = json.loads(raw[brace_start:])
    except json.JSONDecodeError:
        # 尝试找最后一个 } 截断后解析
        brace_end = raw.rfind('}')
        if brace_end == -1:
            return TaskDAG(nodes=[], user_input=user_input)
        try:
            data = json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            return TaskDAG(nodes=[], user_input=user_input)

    tasks_raw = data.get("tasks", [])
    if not tasks_raw:
        return TaskDAG(nodes=[], user_input=user_input)

    nodes = []
    for t in tasks_raw:
        node = _task_to_node(t, user_input=user_input)
        if node:
            nodes.append(node)

    strategy = data.get("strategy", "") or ""
    return TaskDAG(nodes=nodes, user_input=user_input, strategy=strategy)


# ---------------------------------------------------------------------------
# LLM Planner 调用
# ---------------------------------------------------------------------------

def _call_llm_planner(
    user_input: str,
    available_tools: list[str],
    llm_fn: Callable[[list[dict]], dict],
    recent_files: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> TaskDAG:
    """
    调用 LLM 做任务规划。
    llm_fn: 接受 messages list，返回 {"role": "assistant", "content": "..."}
    conversation_history: 最近几轮对话（只传 user/assistant text，不传工具调用细节）
    """
    try:
        system_prompt = _PLANNER_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return TaskDAG(nodes=[], user_input=user_input)

    # 在 system prompt 末尾注入当前可用工具列表
    tool_note = f"\n\n## Currently available tools\n{', '.join(available_tools)}"
    system_prompt += tool_note

    # 注入最近已写入的文件，让 Planner 知道可以引用哪些路径
    if recent_files:
        files_note = "\n\n## Recently written files (use these paths for file_reader if user asks to modify/read previous output)\n"
        files_note += "\n".join(f"- {p}" for p in recent_files[-5:])
        system_prompt += files_note

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 注入最近几轮对话历史（只保留 user/assistant 文本，过滤工具调用细节）
    if conversation_history:
        for msg in conversation_history[-8:]:  # 最多上4轮对话
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "user" and content.strip():
                messages.append({"role": "user", "content": content[:500]})
            elif role == "assistant" and content.strip() and not msg.get("tool_calls"):
                messages.append({"role": "assistant", "content": content[:500]})
    # 当前轮用户输入
    messages.append({"role": "user", "content": user_input})

    try:
        response = llm_fn(messages)
        raw = response.get("content", "") or ""
        if not raw.strip():
            return TaskDAG(nodes=[], user_input=user_input)
        dag = _parse_planner_output(raw, user_input)
        dag.recent_files = list(recent_files) if recent_files else []
        return dag
    except Exception as e:
        print(f"[Planner] LLM 调用失败: {e}，fallback 到 FSM")
        return TaskDAG(nodes=[], user_input=user_input)


# ---------------------------------------------------------------------------
# Mock Planner（极简规则，仅用于 mock 模式验证框架）
# ---------------------------------------------------------------------------

_SIMPLE_CODE_KW = [
    "写一个", "写个", "实现", "编写", "排序", "算法",
    "冒泡", "快排", "fibonacci", "斐波那契", "质数", "二分",
]

_ALGO_NAMES: list[tuple[list[str], str]] = [
    (["冒泡", "bubble"],           "algorithms/bubble_sort.py"),
    (["快排", "quicksort", "快速排序"], "algorithms/quick_sort.py"),
    (["归并", "merge"],            "algorithms/merge_sort.py"),
    (["堆排", "heap"],             "algorithms/heap_sort.py"),
    (["二分", "binary search"],    "algorithms/binary_search.py"),
    (["斐波那契", "fibonacci"],    "algorithms/fibonacci.py"),
    (["质数", "素数", "prime"],    "algorithms/prime_sieve.py"),
]

def _mock_plan(user_input: str, available_tools: list[str]) -> TaskDAG:
    """
    极简规则规划（mock 模式）：
    只识别最明确的单任务（一句话+一个算法关键词），其余返回空 DAG 走 FSM。
    目的是验证框架能跑通，不是模拟真实规划能力。
    """
    lower = user_input.lower()

    # 有分隔符（多任务）→ 直接 fallback，mock 模式不处理
    if re.search(r'[→，；;\n]', user_input) and len(user_input) > 20:
        return TaskDAG(nodes=[], user_input=user_input)

    # 纯计算
    if re.search(r'\d+\s*[+\-*/]\s*\d+', user_input) and "计算" in lower:
        expr_match = re.search(r'\d+(?:\s*[+\-*/]\s*\d+)+', user_input)
        if expr_match:
            return TaskDAG(nodes=[
                TaskNode("t1", "calculator",
                         {"expression": expr_match.group(0).strip()},
                         [], f"计算 {expr_match.group(0).strip()}")
            ], user_input=user_input)

    # 单代码任务
    if not any(kw in lower for kw in _SIMPLE_CODE_KW):
        return TaskDAG(nodes=[], user_input=user_input)

    # 推断文件名
    save_path = _unique_fallback_path("py")
    for keywords, path in _ALGO_NAMES:
        if any(kw in lower for kw in keywords):
            save_path = path
            break

    nodes = [
        TaskNode("t1", "code_executor",
                 {"code": "__GENERATE__", "language": "python"},
                 [], "生成并执行代码"),
    ]
    if "file_writer" in available_tools:
        nodes.append(TaskNode("t2", "file_writer",
                              {"path": save_path, "content": "__FROM_CODE__"},
                              ["t1"], f"保存到 {save_path}"))
        nodes.append(TaskNode("t3", "code_executor",
                              {"code": "__FROM_FILE__", "language": "python"},
                              ["t2"], "从文件验证"))
    else:
        nodes.append(TaskNode("t2", "code_executor",
                              {"code": "__REUSE__", "language": "python"},
                              ["t1"], "验证"))

    return TaskDAG(nodes=nodes, user_input=user_input, strategy="mock 规则识别到单任务管道（生成→保存→验证）")


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def plan(
    user_input: str,
    available_tools: list[str],
    llm_fn: Optional[Callable] = None,
    mock: bool = False,
    recent_files: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> TaskDAG:
    """
    主规划入口。

    Args:
        user_input:             用户原始输入
        available_tools:        当前可用工具名列表
        llm_fn:                 真实模型的调用函数（接受 messages，返回 dict）
                                为 None 时自动使用 mock 规划
        mock:                   强制使用 mock 规划（覆盖 llm_fn）
        recent_files:           最近已写入的文件路径列表（来自 Artifact Registry），
                                会被注入到 Planner 的 system prompt
        conversation_history:   最近几轮对话（自 self.messages），让 Planner 了解上下文

    Returns:
        TaskDAG，空 DAG 表示 fallback 到 FSM ReAct
    """
    if mock or llm_fn is None:
        return _mock_plan(user_input, available_tools)
    else:
        return _call_llm_planner(user_input, available_tools, llm_fn,
                                  recent_files=recent_files,
                                  conversation_history=conversation_history)
