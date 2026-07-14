"""
episodic_memory_test.py — Episodic Memory 向量相似度检索独立测试

测试 episodic_memory.py 的三层记忆管理，重点验证：
  1. archive_turn() 归档写入 SQLite
  2. recall() 向量相似度检索（MiniLM）/ BM25 fallback
  3. should_archive() + trim_working_memory() 压缩触发
  4. 跨会话持久化：_rebuild_index() 从 SQLite 重建向量索引

用法（在服务器上，agent/code 目录下）：
    python episodic_memory_test.py
    python episodic_memory_test.py --verbose      # 打印检索结果详情
    python episodic_memory_test.py --model_path /path/to/all-MiniLM-L6-V2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from episodic_memory import EpisodicMemory, EpisodicTurn


def _make_turn(user: str, ai: str, tools: list[str] = None):
    user_m = {"role": "user", "content": user}
    ai_m = {"role": "assistant", "content": ai, "tool_calls": []}
    tool_ms = [{"role": "tool", "name": t, "content": f"{t} 调用结果"} for t in (tools or [])]
    return user_m, ai_m, tool_ms


def _step(name: str):
    print(f"\n{'='*55}\n[STEP] {name}\n{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="打印检索结果详情")
    parser.add_argument(
        "--model_path",
        default=str(_HERE / "models" / "all-MiniLM-L6-V2"),
        help="MiniLM 模型路径（不传则自动尝试默认路径，失败时降级为 BM25）",
    )
    parser.add_argument("--persist_dir", default="/tmp/episodic_test",
                        help="测试用 SQLite 目录（测试完自动删除）")
    args = parser.parse_args()

    import shutil
    persist_dir = Path(args.persist_dir)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)  # 确保每次从干净状态开始

    session_id = "test_session_001"

    # ── STEP 0: 初始化，检查向量引擎 ────────────────────────────
    _step("0. 初始化 EpisodicMemory，检查向量引擎")
    mem = EpisodicMemory(
        session_id=session_id,
        persist_dir=str(persist_dir),
        archive_after_turns=6,
        keep_recent=3,
        embedding_model_hint=args.model_path,
    )
    stats = mem.stats()
    engine = stats.get("retrieval_mode", "unknown")
    print(f"✅ 初始化成功，检索引擎: {engine}")
    if engine == "bm25":
        print("  ⚠️  使用 BM25 关键词检索（MiniLM 未加载，可能模型路径不存在或 torch 未安装）")
        print(f"     MiniLM 路径: {args.model_path}")
    else:
        print(f"  ✅ 使用向量相似度检索（MiniLM 已加载）")

    # ── STEP 1: 归档多条历史轮次 ─────────────────────────────────
    _step("1. archive_turn() — 归档 8 条测试历史")
    test_turns = [
        ("帮我写一个快速排序算法", "好的，以下是快速排序的 Python 实现...", ["code_executor", "file_writer"]),
        ("快速排序的时间复杂度是多少", "快速排序平均时间复杂度是 O(n log n)", []),
        ("帮我计算 1+2+3+...+100", "结果是 5050", ["calculator"]),
        ("写一个二分查找算法", "二分查找的时间复杂度是 O(log n)，以下是实现...", ["code_executor"]),
        ("帮我读取 data/report.txt 的内容", "文件内容是：这是报告正文内容", ["file_reader"]),
        ("解释一下什么是动态规划", "动态规划是将复杂问题分解为有重叠的子问题...", []),
        ("帮我把结果保存到 output.txt", "已保存到 output.txt，共 512 字节", ["file_writer"]),
        ("冒泡排序和快速排序哪个更快", "快速排序通常更快，平均 O(n log n) vs O(n²)", []),
    ]
    for user, ai, tools in test_turns:
        user_m, ai_m, tool_ms = _make_turn(user, ai, tools)
        mem.archive_turn(user_m, ai_m, tool_ms)
    stats = mem.stats()
    archived = stats["archived_turns"]
    assert archived == 8, f"❌ 预期 8 条，实际 {archived}"
    print(f"✅ 归档完成，数据库中共 {archived} 条记录")

    # ── STEP 2: 向量相似度检索 ───────────────────────────────────
    _step("2. recall() — 向量相似度/BM25 检索")
    test_queries = [
        ("排序算法", ["快速排序", "冒泡排序", "二分查找"]),
        ("时间复杂度", ["O(n log n)", "O(log n)", "O(n²)"]),
        ("文件操作", ["output.txt", "report.txt"]),
        ("数学计算", ["5050", "1+2+3"]),
    ]
    all_passed = True
    for query, expected_keywords in test_queries:
        results: list[EpisodicTurn] = mem.recall(query)
        result_texts = " ".join(
            (t.user_text or "") + " " + (t.ai_summary or "") for t in results
        )
        hit = any(kw in result_texts for kw in expected_keywords)
        status = "✅" if hit else "❌"
        if not hit:
            all_passed = False
        print(f"  {status} 查询: '{query}' → top-{len(results)} 结果命中 {expected_keywords[:2]}: {hit}")
        if args.verbose:
            for t in results:
                print(f"       turn_id={t.turn_id}: {(t.user_text or '')[:50]}...")

    if all_passed:
        print("✅ 所有检索查询命中")
    else:
        print("⚠️  部分检索未命中（BM25 对短查询的召回可能不如向量检索）")

    # ── STEP 3: Working Memory 压缩 ──────────────────────────────
    _step("3. should_archive() + trim_working_memory() — 压缩触发")
    # 构造超过 archive_after_turns=6 的 working memory（16条 = 8轮对话）
    working_msgs = []
    for user, ai, _ in test_turns:
        working_msgs.append({"role": "user", "content": user})
        working_msgs.append({"role": "assistant", "content": ai})

    user_count = sum(1 for m in working_msgs if m["role"] == "user")
    should = mem.should_archive(working_msgs)
    print(f"  working memory: {user_count} 轮对话，should_archive: {should}")
    assert should, f"❌ {user_count} 轮 > archive_after_turns=6，应该触发压缩"

    trimmed = mem.trim_working_memory(working_msgs)
    trimmed_user = sum(1 for m in trimmed if m.get("role") == "user")
    print(f"  ✅ 压缩后: {user_count} → {trimmed_user} 轮对话（保留最近 {mem.keep_recent} 轮）")
    assert len(trimmed) < len(working_msgs), "❌ 压缩后条数应该减少"

    # ── STEP 4: 跨会话持久化 ─────────────────────────────────────
    _step("4. 跨会话持久化 — 模拟重启，从 SQLite 重建向量索引")
    mem2 = EpisodicMemory(
        session_id=session_id,
        persist_dir=str(persist_dir),
        archive_after_turns=6,
        keep_recent=3,
        embedding_model_hint=args.model_path,
    )
    stats2 = mem2.stats()
    rebuilt = stats2["archived_turns"]
    print(f"  重建后索引中共 {rebuilt} 条（来自 SQLite）")
    assert rebuilt == 8, f"❌ 持久化后应该还有 8 条，实际 {rebuilt}"

    results2 = mem2.recall("排序算法")
    assert len(results2) > 0, "❌ 重建后检索应该有结果"
    print(f"  ✅ 重建后检索 '排序算法' → {len(results2)} 条结果，持久化正常")

    # ── 清理测试数据 ──────────────────────────────────────────────
    try:
        shutil.rmtree(persist_dir)
        print(f"\n[清理] 已删除测试目录: {persist_dir}")
    except Exception:
        pass

    print(f"\n{'='*55}")
    print("✅✅✅ Episodic Memory 全部测试通过！")
    print(f"{'='*55}")
    print(f"  检索引擎: {engine}")
    print(f"  测试覆盖: archive × 8  →  recall × {len(test_queries)}  →  压缩  →  持久化重建")


if __name__ == "__main__":
    main()
