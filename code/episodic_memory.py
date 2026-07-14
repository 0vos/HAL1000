"""
episodic_memory.py — 分层上下文管理（Episodic Memory）

存储：SQLite（outputs/episodic/{session_id}.db）
检索：优先用 torch embedding 余弦相似度，fallback 到 BM25 关键词检索

自动选择模式：
  - 有 torch + transformers → EmbeddingRecall（语义向量，效果最好）
  - 仅有基础库            → BM25Recall（关键词，无依赖）
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# BM25（纯 Python，无依赖，fallback 用）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """英文按单词，中文按单字 + bigram。"""
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
             "and", "or", "for", "with", "that", "this", "it", "as", "on",
             "at", "be", "by", "from", "we", "you", "he", "she", "they",
             "我", "你", "他", "她", "的", "了", "是", "在", "和", "有",
             "不", "这", "那", "也", "都", "就", "还", "会", "能", "对"}
    tokens: list[str] = []
    for w in re.findall(r'[a-z0-9]+', text.lower()):
        if w not in _STOP and len(w) > 1:
            tokens.append(w)
    cjk = re.findall(r'[\u4e00-\u9fff]', text)
    for ch in cjk:
        if ch not in _STOP:
            tokens.append(ch)
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])
    return tokens


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs: list[list[str]] = []
        self._doc_ids: list[int] = []
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def add(self, doc_id: int, text: str) -> None:
        tokens = _tokenize(text)
        self._docs.append(tokens)
        self._doc_ids.append(doc_id)
        for t in set(tokens):
            self._df[t] = self._df.get(t, 0) + 1
        N = len(self._docs)
        self._idf = {t: math.log((N - df + 0.5) / (df + 0.5) + 1)
                     for t, df in self._df.items()}
        self._avgdl = sum(len(d) for d in self._docs) / max(N, 1)

    def score(self, query: str) -> list[tuple[int, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self._docs:
            return []
        scores = []
        for tokens in self._docs:
            dl = len(tokens)
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            s = sum(
                self._idf.get(qt, 0) * (tf_map.get(qt, 0) * (self.k1 + 1)) /
                (tf_map.get(qt, 0) + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1)))
                for qt in q_tokens
            )
            scores.append(s)
        ranked = sorted(zip(self._doc_ids, scores), key=lambda x: -x[1])
        return [(did, sc) for did, sc in ranked if sc > 0]


# ---------------------------------------------------------------------------
# Embedding 检索（有 torch 时使用）
# ---------------------------------------------------------------------------

class EmbeddingRecall:
    """
    用 torch + transformers 做语义 embedding 检索。
    优先加载项目内的 all-MiniLM-L6-V2（/models/），
    fallback 用 Qwen tokenizer 的 last-hidden-state 均值。
    """

    def __init__(self, model_hint: str | None = None):
        self._model = None
        self._tokenizer = None
        self._doc_ids: list[int] = []
        self._vecs: list = []   # list of 1D tensors
        self._ready = False
        self._load(model_hint)

    def _load(self, model_hint: str | None) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            # 候选路径
            candidates = []
            if model_hint:
                candidates.append(model_hint)
            # 项目内 MiniLM
            here = Path(__file__).resolve().parent
            candidates += [
                str(here / "models" / "all-MiniLM-L6-V2"),
                str(here.parent / "models" / "all-MiniLM-L6-V2"),
                "/root/siton-tmp/HAL1000/agent/code/models/all-MiniLM-L6-V2",
            ]
            for path in candidates:
                if Path(path).exists():
                    self._tokenizer = AutoTokenizer.from_pretrained(path)
                    self._model = AutoModel.from_pretrained(path)
                    self._model.eval()
                    self._ready = True
                    print(f"[episodic] embedding 模型已加载: {path}")
                    return
            # 没有 MiniLM，用 Qwen 的 tokenizer 做简单 bag-of-words embedding
            print("[episodic] 未找到 MiniLM，降级到 BM25")
        except Exception as e:
            print(f"[episodic] embedding 加载失败（{e}），使用 BM25")

    def _embed(self, text: str):
        """返回归一化 embedding 向量（torch.Tensor 1D）。"""
        import torch
        inputs = self._tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            out = self._model(**inputs)
        # mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        vec = vec.squeeze(0)
        # L2 归一化
        vec = vec / (vec.norm() + 1e-8)
        return vec

    def add(self, doc_id: int, text: str) -> None:
        if not self._ready:
            return
        import torch
        vec = self._embed(text)
        self._doc_ids.append(doc_id)
        self._vecs.append(vec)

    def score(self, query: str) -> list[tuple[int, float]]:
        if not self._ready or not self._vecs:
            return []
        import torch
        q_vec = self._embed(query)
        sims = [(did, float(q_vec @ v)) for did, v in zip(self._doc_ids, self._vecs)]
        return sorted(sims, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# EpisodicTurn
# ---------------------------------------------------------------------------

@dataclass
class EpisodicTurn:
    turn_id: int
    session_id: str
    created_at: str
    user_text: str
    ai_summary: str
    tool_names: list[str]
    artifact_ids: list[str]
    full_json: str

    def to_memory_block(self) -> str:
        tools_str = ", ".join(self.tool_names) if self.tool_names else "无"
        arts_str = " ".join(self.artifact_ids) if self.artifact_ids else ""
        return (
            f"[Memory #{self.turn_id}] {self.created_at[:16]}\n"
            f"用户: {self.user_text[:100]}\n"
            f"AI: {self.ai_summary[:200]}\n"
            f"工具: {tools_str}"
            + (f"\n产物: {arts_str}" if arts_str else "")
        )


# ---------------------------------------------------------------------------
# EpisodicMemory — 主类
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    三层内存管理器：
      Working Memory  → self.messages（最近 keep_recent 轮）
      Episodic Memory → SQLite 归档 + BM25/Embedding 检索
      Artifact Refs   → 由 artifact_registry 管理

    自动检测运行环境，选择最佳检索引擎：
      有 torch + MiniLM → EmbeddingRecall（语义）
      否则              → BM25（关键词）
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS turns (
        turn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        user_text    TEXT NOT NULL,
        ai_summary   TEXT NOT NULL,
        tool_names   TEXT NOT NULL,
        artifact_ids TEXT NOT NULL,
        full_json    TEXT NOT NULL,
        search_text  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_session ON turns(session_id);
    """

    def __init__(
        self,
        session_id: str,
        persist_dir: str | Path,
        max_working_tokens: int = 2000,
        archive_after_turns: int = 6,
        keep_recent: int = 3,
        embedding_model_hint: str | None = None,
    ):
        self.session_id = session_id
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.max_working_tokens = max_working_tokens
        self.archive_after_turns = archive_after_turns
        self.keep_recent = keep_recent
        self._lock = threading.Lock()

        # SQLite
        self._db_path = self.persist_dir / f"{session_id}.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

        # 检索引擎：优先 Embedding，fallback BM25
        self._embed_recall = EmbeddingRecall(model_hint=embedding_model_hint)
        if not self._embed_recall._ready:
            self._embed_recall = None  # type: ignore
        self._bm25 = BM25()
        self._mode = "embedding" if self._embed_recall else "bm25"

        # 从 DB 重建内存索引
        self._rebuild_index()

    # ------------------------------------------------------------------
    # 归档
    # ------------------------------------------------------------------
    def archive_turn(
        self,
        user_msg: dict,
        ai_msg: dict,
        tool_msgs: list[dict],
        artifact_ids: list[str] | None = None,
    ) -> int:
        user_text = (user_msg.get("content") or "")[:500]
        ai_content = (ai_msg.get("content") or "")
        ai_summary = ai_content[:300]
        tool_names = [m.get("name", "") for m in tool_msgs if m.get("role") == "tool"]
        artifact_ids = artifact_ids or []
        created_at = _now_iso()
        full_json = json.dumps([user_msg, ai_msg] + tool_msgs, ensure_ascii=False)
        search_text = f"{user_text} {ai_summary} {' '.join(tool_names)}"

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO turns
                   (session_id, created_at, user_text, ai_summary,
                    tool_names, artifact_ids, full_json, search_text)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (self.session_id, created_at, user_text, ai_summary,
                 json.dumps(tool_names), json.dumps(artifact_ids),
                 full_json, search_text),
            )
            turn_id = cur.lastrowid
            self._conn.commit()

        # 更新内存索引
        if self._embed_recall:
            self._embed_recall.add(turn_id, search_text)
        self._bm25.add(turn_id, search_text)
        return turn_id

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def recall(self, query: str, top_k: int = 3) -> list[EpisodicTurn]:
        if self._mode == "embedding" and self._embed_recall:
            ranked = self._embed_recall.score(query)[:top_k]
        else:
            ranked = self._bm25.score(query)[:top_k]

        if not ranked:
            return []
        turn_ids = [tid for tid, _ in ranked]
        placeholders = ",".join("?" * len(turn_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM turns WHERE turn_id IN ({placeholders})",
                turn_ids,
            ).fetchall()
        turns = [self._row_to_turn(r) for r in rows]
        turns.sort(key=lambda t: t.turn_id)
        return turns

    # ------------------------------------------------------------------
    # Working memory 修剪
    # ------------------------------------------------------------------
    def should_archive(self, messages: list[dict]) -> bool:
        user_count = sum(1 for m in messages if m.get("role") == "user")
        return user_count > self.archive_after_turns

    def trim_working_memory(
        self,
        messages: list[dict],
        recalled_turns: list[EpisodicTurn] | None = None,
    ) -> list[dict]:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        memory_injection: list[dict] = []
        if recalled_turns:
            memory_text = "\n\n".join(t.to_memory_block() for t in recalled_turns)
            memory_injection = [{
                "role": "system",
                "content": f"[以下是从历史对话中召回的相关记忆，供参考]\n{memory_text}",
            }]

        # 按轮分组（每个 user 消息开始新一轮）
        rounds: list[list[dict]] = []
        current: list[dict] = []
        for m in non_system:
            if m.get("role") == "user" and current:
                rounds.append(current)
                current = [m]
            else:
                current.append(m)
        if current:
            rounds.append(current)

        kept = rounds[-self.keep_recent:] if len(rounds) > self.keep_recent else rounds
        kept_msgs = [m for r in kept for m in r]
        return system_msgs + memory_injection + kept_msgs

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id=?",
                (self.session_id,),
            ).fetchone()[0]
        return {
            "archived_turns": count,
            "session_id": self.session_id,
            "retrieval_mode": self._mode,
            "db_path": str(self._db_path),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        rows = self._conn.execute(
            "SELECT turn_id, search_text FROM turns WHERE session_id=?",
            (self.session_id,),
        ).fetchall()
        for turn_id, search_text in rows:
            if self._embed_recall:
                self._embed_recall.add(turn_id, search_text)
            self._bm25.add(turn_id, search_text)

    def _row_to_turn(self, row) -> EpisodicTurn:
        (turn_id, session_id, created_at, user_text, ai_summary,
         tool_names_json, artifact_ids_json, full_json, _) = row
        return EpisodicTurn(
            turn_id=turn_id, session_id=session_id, created_at=created_at,
            user_text=user_text, ai_summary=ai_summary,
            tool_names=json.loads(tool_names_json),
            artifact_ids=json.loads(artifact_ids_json),
            full_json=full_json,
        )

    def close(self) -> None:
        self._conn.close()


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        mem = EpisodicMemory("sess_test", tmp, archive_after_turns=2, keep_recent=2)
        print(f"检索模式: {mem._mode}")
        for i in range(5):
            mem.archive_turn(
                {"role": "user", "content": f"问题{i}: {'如何排序数组？' if i%2==0 else '斐波那契数列'}"},
                {"role": "assistant", "content": f"回答{i}: {'冒泡排序...' if i%2==0 else 'fib(n)=fib(n-1)+fib(n-2)'}"},
                [], artifact_ids=[f"art_{i:04x}"],
            )
        print("归档 5 轮完成")
        results = mem.recall("排序算法", top_k=2)
        print(f"召回 {len(results)} 条（排序算法）:", [t.user_text[:20] for t in results])
        results2 = mem.recall("斐波那契", top_k=2)
        print(f"召回 {len(results2)} 条（斐波那契）:", [t.user_text[:20] for t in results2])
        fake = [{"role": "system", "content": "sys"}] + [
            m for i in range(6)
            for m in [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
        ]
        trimmed = mem.trim_working_memory(fake, recalled_turns=results)
        print(f"修剪: {len(fake)} → {len(trimmed)} 条")
        assert len(trimmed) < len(fake)
        print(f"统计: {mem.stats()}")
        print("ALL PASS")
    finally:
        shutil.rmtree(tmp)
