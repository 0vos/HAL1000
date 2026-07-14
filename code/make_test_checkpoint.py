"""
手动写入一个 checkpoint，用于测试 b1_resume.py 的 resume 功能。
模拟：已完成第1步(calculator 123*456)，还剩第2步(calculator 789+321)。
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "code"))
from b1_checkpoint import save_checkpoint

outdir = Path(__file__).parent / "outputs" / "B1_checkpoint_test"
outdir.mkdir(parents=True, exist_ok=True)

# 模拟已经跑完第1轮工具调用的状态
messages = [
    {"role": "system", "content": "You are HAL1000, a local tool-using AI Agent."},
    {"role": "user", "content": "请分两步计算：第一步 123 * 456；第二步 789 + 321。"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_001", "name": "calculator", "args": {"expression": "123 * 456"}}
    ]},
    {"role": "tool", "tool_call_id": "call_001", "name": "calculator",
     "content": json.dumps({"status": "success", "output": {"result": 56088, "expression": "123 * 456"}}),
     "status": "success"},
]

checkpoint_state = {
    "conversation_id": "conv_checkpoint_test",
    "execution_mode": "integrated",
    "resume_from_turn": 1,
    "messages": messages,
    "tool_rounds": 1,
    "llm_calls": 1,
    "turns": [],
    "status": "running",
    "runtime_input": {
        "conversation_id": "conv_checkpoint_test",
        "execution_mode": "integrated",
        "user_input": "请分两步计算：第一步 123 * 456；第二步 789 + 321。",
        "system_prompt_path": "../prompts/local_tool_agent.txt",
        "toolset": "basic_tools",
        "max_turns": 8,
        "save_memory": "none",
    }
}

save_checkpoint(outdir, checkpoint_state)
print(f"✅ checkpoint 已写入: {outdir}/checkpoint.json")
print("现在可以跑 resume：")
print(f"  python code/b1_resume.py \\")
print(f"    --outdir outputs/B1_checkpoint_test \\")
print(f"    --tools_config configs/tools.yaml \\")
print(f"    --model_config configs/model.yaml \\")
print(f"    --memory_config configs/memory.yaml")
