"""
artifact_registry.py — 产物版本管理（DIBP 风格）

每次工具执行产生输出时，注册为一个带版本的 Artifact。
支持：
  - register(tool_name, args, output, session_id) → artifact_id
  - get(artifact_id) → 最新版本的 Artifact
  - get_version(artifact_id, version) → 指定版本
  - rollback(artifact_id, version) → 真正回滚：还原文件内容 + 移动 current_version 指针
  - diff(artifact_id, v1, v2) → 对比两个版本的文件内容（file_writer）或 output dict（其他工具）
  - history(artifact_id) → 所有版本列表
  - list_session(session_id) → 本次会话所有 artifact
  - to_ref(artifact_id) → 返回可嵌入 prompt 的简短引用字符串，如 "[art_a1b2 file_writer v3]"

设计：
  - artifact_id = "art_" + sha256(tool_name+json(args))[:8]
    同一工具+同参数的多次调用 → 同一 artifact_id，版本递增
  - 持久化到 outputs/artifacts/{session_id}.json
  - 线程安全（threading.Lock）
  - file_writer 产物额外持久化文件内容快照，支持真正的文件级回滚

数据结构：
  ArtifactVersion:
    version: int
    created_at: str (ISO)
    output: dict          # 工具的原始 output 字段
    summary: str          # 一行人类可读摘要（由 _summarize 生成）
    content_snapshot: str | None
                          # 仅 file_writer：文件内容快照（用于真正回滚）

  Artifact:
    artifact_id: str
    tool_name: str
    args: dict
    session_id: str
    current_version: int
    versions: list[ArtifactVersion]
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArtifactVersion:
    version: int
    created_at: str           # ISO 8601
    output: dict              # 工具的原始 output 字段
    summary: str              # 一行人类可读摘要
    content_snapshot: Optional[str] = None
    # 仅 file_writer 填充：文件内容快照，用于真正的文件级回滚


@dataclass
class Artifact:
    artifact_id: str
    tool_name: str
    args: dict
    session_id: str
    current_version: int
    versions: List[ArtifactVersion] = field(default_factory=list)

    def current(self) -> Optional[ArtifactVersion]:
        """返回当前版本的 ArtifactVersion。"""
        return self.get_version(self.current_version)

    def get_version(self, version: int) -> Optional[ArtifactVersion]:
        """按版本号查找 ArtifactVersion。"""
        for v in self.versions:
            if v.version == version:
                return v
        return None


# ---------------------------------------------------------------------------
# Summarize helper
# ---------------------------------------------------------------------------

def _summarize(tool_name: str, output: dict) -> str:
    """根据工具类型生成一行人类可读摘要。"""
    if tool_name == "file_writer":
        path = output.get("written_path") or output.get("relative_path") or output.get("path") or "?"
        num_bytes = output.get("num_bytes", 0)
        return f"写入 {path}，{num_bytes} 字节"
    elif tool_name == "file_reader":
        source = output.get("source") or output.get("path") or "?"
        num_chars = output.get("num_chars", 0)
        truncated = output.get("truncated", False)
        trunc_mark = "（已截断）" if truncated else ""
        return f"读取 {source}，{num_chars} 字符{trunc_mark}"
    elif tool_name == "calculator":
        result = output.get("result")
        return f"计算结果：{result}"
    elif tool_name == "code_executor":
        rc = output.get("returncode", -1)
        elapsed = output.get("elapsed_ms", 0)
        stdout_lines = len((output.get("stdout") or "").strip().splitlines())
        if rc == 0:
            return f"执行成功，{stdout_lines} 行输出，耗时 {elapsed:.0f}ms"
        else:
            stderr_snip = (output.get("stderr") or "")[:60].replace("\n", " ")
            return f"执行失败 (exit={rc})，stderr: {stderr_snip}"
    elif tool_name == "local_file_search":
        results = output.get("results", [])
        return f"搜索命中 {len(results)} 条结果"
    elif tool_name == "table_analyzer":
        shape = output.get("shape")
        if shape:
            return f"分析表格，维度：{shape}"
        rows = output.get("num_rows", "?")
        cols = output.get("num_columns", "?")
        return f"分析表格，{rows} 行 × {cols} 列"
    elif tool_name == "format_converter":
        target_format = output.get("target_format") or "?"
        num_chars = len(output.get("content") or output.get("formatted_text") or "")
        return f"格式转换为 {target_format}，{num_chars} 字符"
    elif tool_name == "pdf_reader":
        path = output.get("path") or "?"
        num_pages = output.get("num_pages", "?")
        num_chars = output.get("num_chars", 0)
        return f"读取 PDF {path}，{num_pages} 页，{num_chars} 字符"
    elif tool_name == "docx_reader":
        path = output.get("path") or "?"
        num_paragraphs = output.get("num_paragraphs", "?")
        num_chars = output.get("num_chars", 0)
        return f"读取 DOCX {path}，{num_paragraphs} 段落，{num_chars} 字符"
    else:
        snippet = str(output)[:80].replace("\n", " ")
        return f"{tool_name} 输出：{snippet}"


def _read_content_snapshot(output: dict) -> Optional[str]:
    """
    尝试读取 file_writer 写入文件的内容，作为版本快照保存。
    失败时（文件已被删除、权限问题等）静默返回 None。
    """
    written_path = output.get("written_path")
    if not written_path:
        return None
    try:
        return Path(written_path).read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _artifact_version_to_dict(v: ArtifactVersion) -> dict:
    d = {
        "version": v.version,
        "created_at": v.created_at,
        "output": v.output,
        "summary": v.summary,
    }
    if v.content_snapshot is not None:
        d["content_snapshot"] = v.content_snapshot
    return d


def _artifact_to_dict(a: Artifact) -> dict:
    return {
        "artifact_id": a.artifact_id,
        "tool_name": a.tool_name,
        "args": a.args,
        "session_id": a.session_id,
        "current_version": a.current_version,
        "versions": [_artifact_version_to_dict(v) for v in a.versions],
    }


def _artifact_from_dict(d: dict) -> Artifact:
    versions = [
        ArtifactVersion(
            version=v["version"],
            created_at=v["created_at"],
            output=v["output"],
            summary=v["summary"],
            content_snapshot=v.get("content_snapshot"),
        )
        for v in d.get("versions", [])
    ]
    return Artifact(
        artifact_id=d["artifact_id"],
        tool_name=d["tool_name"],
        args=d["args"],
        session_id=d["session_id"],
        current_version=d["current_version"],
        versions=versions,
    )


# ---------------------------------------------------------------------------
# ArtifactRegistry
# ---------------------------------------------------------------------------

class ArtifactRegistry:
    """
    产物版本管理注册表。

    参数：
        session_id:   当前会话 ID
        persist_dir:  持久化目录，文件名 = {session_id}.json
                      默认为 None（不持久化，仅内存）
    """

    def __init__(self, session_id: str, persist_dir: Optional[str] = None):
        self.session_id = session_id
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._lock = threading.Lock()
        self._store: dict[str, Artifact] = {}
        self._session_index: dict[str, set] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _artifact_id(self, tool_name: str, args: dict) -> str:
        """
        生成 artifact_id。
        file_writer 特殊处理：只用 path 作为 key，忽略 content。
        这样对同一个文件的多次写入才能产生同一 artifact_id 并版本递增。
        其他工具：使用全部 args。
        """
        if tool_name == "file_writer":
            key = tool_name + json.dumps({"path": args.get("path", "")}, sort_keys=True, ensure_ascii=False)
        else:
            key = tool_name + json.dumps(args, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return "art_" + digest[:8]

    def _persist_path(self) -> Optional[Path]:
        if self.persist_dir is None:
            return None
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return self.persist_dir / f"{self.session_id}.json"

    def _save(self) -> None:
        """原子写入持久化文件。"""
        path = self._persist_path()
        if path is None:
            return
        data = {
            "session_id": self.session_id,
            "artifacts": {
                aid: _artifact_to_dict(artifact)
                for aid, artifact in self._store.items()
                if artifact.session_id == self.session_id
            },
        }
        text = json.dumps(data, ensure_ascii=False, indent=2)
        tmp = path.with_name("." + path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        path = self._persist_path()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for aid, art_dict in raw.get("artifacts", {}).items():
                artifact = _artifact_from_dict(art_dict)
                self._store[aid] = artifact
                self._session_index.setdefault(artifact.session_id, set()).add(aid)
        except (json.JSONDecodeError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        tool_name: str,
        args: dict,
        output: dict,
        session_id: Optional[str] = None,
    ) -> str:
        """
        注册一次工具执行产物。版本递增，file_writer 额外保存文件内容快照。
        """
        sid = session_id or self.session_id
        aid = self._artifact_id(tool_name, args)
        now = datetime.now(timezone.utc).isoformat()
        summary = _summarize(tool_name, output)

        # file_writer：注册时立即读取文件内容作为快照
        snapshot = _read_content_snapshot(output) if tool_name == "file_writer" else None

        with self._lock:
            if aid in self._store:
                artifact = self._store[aid]
                new_version = artifact.current_version + 1
                av = ArtifactVersion(
                    version=new_version,
                    created_at=now,
                    output=output,
                    summary=summary,
                    content_snapshot=snapshot,
                )
                artifact.versions.append(av)
                artifact.current_version = new_version
            else:
                av = ArtifactVersion(
                    version=1,
                    created_at=now,
                    output=output,
                    summary=summary,
                    content_snapshot=snapshot,
                )
                artifact = Artifact(
                    artifact_id=aid,
                    tool_name=tool_name,
                    args=args,
                    session_id=sid,
                    current_version=1,
                    versions=[av],
                )
                self._store[aid] = artifact
                self._session_index.setdefault(sid, set()).add(aid)
            self._save()

        return aid

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """返回 Artifact 对象（含所有版本）。"""
        with self._lock:
            return self._store.get(artifact_id)

    def get_version(self, artifact_id: str, version: int) -> Optional[ArtifactVersion]:
        """返回指定版本的 ArtifactVersion。"""
        with self._lock:
            artifact = self._store.get(artifact_id)
            if artifact is None:
                return None
            return artifact.get_version(version)

    def rollback(self, artifact_id: str, version: int) -> dict:
        """
        真正的回滚：
        1. 移动 current_version 指针到目标版本
        2. 如果是 file_writer 且有内容快照，原子写回磁盘文件
        3. 其他工具（code_executor 等）无法回滚执行结果，仅移动指针并说明

        返回：
          {"ok": True,  "restored_file": path_str | None, "note": "..."}
          {"ok": False, "reason": "..."}
        """
        with self._lock:
            artifact = self._store.get(artifact_id)
            if artifact is None:
                return {"ok": False, "reason": f"artifact {artifact_id!r} not found"}

            target_av = artifact.get_version(version)
            if target_av is None:
                return {"ok": False, "reason": f"version {version} not found in {artifact_id}"}

            # 移动版本指针
            artifact.current_version = version
            self._save()

        # file_writer：有快照则原子写回文件
        restored_file = None
        note = ""

        if artifact.tool_name == "file_writer":
            if target_av.content_snapshot is not None:
                written_path = target_av.output.get("written_path")
                if written_path:
                    p = Path(written_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_name("." + p.name + ".tmp")
                    try:
                        tmp.write_text(target_av.content_snapshot, encoding="utf-8")
                        os.replace(tmp, p)
                        restored_file = written_path
                        note = f"已将 {written_path} 还原到 v{version} 的内容"
                    except BaseException as e:
                        try:
                            tmp.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return {"ok": False, "reason": f"文件写回失败: {e}"}
                else:
                    note = f"版本指针已移到 v{version}，但 output 中无 written_path，无法还原文件"
            else:
                note = (
                    f"版本指针已移到 v{version}，但该版本无内容快照"
                    f"（可能是旧版本 registry 注册的，未保存快照），无法还原文件"
                )
        else:
            note = (
                f"版本指针已移到 v{version}。"
                f"{artifact.tool_name} 的执行结果无法回滚（代码已跑、计算已完成），"
                f"但历史记录可通过 get_version() 查阅。"
            )

        return {"ok": True, "restored_file": restored_file, "note": note}

    def diff(self, artifact_id: str, v1: int, v2: int) -> str:
        """
        对比两个版本：
        - file_writer：对比文件内容快照（更直观）
        - 其他工具：对比 output dict 的 JSON 表示
        """
        with self._lock:
            artifact = self._store.get(artifact_id)
            if artifact is None:
                return f"artifact {artifact_id!r} not found"
            av1 = artifact.get_version(v1)
            av2 = artifact.get_version(v2)
            tool_name = artifact.tool_name

        if av1 is None:
            return f"version {v1} not found in {artifact_id}"
        if av2 is None:
            return f"version {v2} not found in {artifact_id}"

        # file_writer 优先用内容快照做 diff，内容更直观
        if tool_name == "file_writer" and av1.content_snapshot is not None and av2.content_snapshot is not None:
            text1 = av1.content_snapshot.splitlines(keepends=True)
            text2 = av2.content_snapshot.splitlines(keepends=True)
            from_label = f"{artifact_id} v{v1} (文件内容)"
            to_label   = f"{artifact_id} v{v2} (文件内容)"
        else:
            text1 = json.dumps(av1.output, ensure_ascii=False, indent=2).splitlines(keepends=True)
            text2 = json.dumps(av2.output, ensure_ascii=False, indent=2).splitlines(keepends=True)
            from_label = f"{artifact_id} v{v1} (output)"
            to_label   = f"{artifact_id} v{v2} (output)"

        diff_lines = list(difflib.unified_diff(text1, text2, fromfile=from_label, tofile=to_label))
        if not diff_lines:
            return "(no difference)"
        return "".join(diff_lines)

    def history(self, artifact_id: str) -> List[ArtifactVersion]:
        """返回所有版本列表（按版本号升序）。"""
        with self._lock:
            artifact = self._store.get(artifact_id)
            if artifact is None:
                return []
            return sorted(artifact.versions, key=lambda v: v.version)

    def list_session(self, session_id: Optional[str] = None) -> List[Artifact]:
        """返回指定会话的所有 Artifact。"""
        sid = session_id or self.session_id
        with self._lock:
            aids = self._session_index.get(sid, set())
            return [self._store[aid] for aid in aids if aid in self._store]

    def to_ref(self, artifact_id: str) -> str:
        """返回可嵌入 prompt 的简短引用字符串，如 '[art_a1b2 file_writer v3]'。"""
        with self._lock:
            artifact = self._store.get(artifact_id)
            if artifact is None:
                return f"[{artifact_id} unknown]"
            return f"[{artifact_id} {artifact.tool_name} v{artifact.current_version}]"


# ---------------------------------------------------------------------------
# Module-level convenience singleton (optional usage)
# ---------------------------------------------------------------------------

_default_registry: Optional[ArtifactRegistry] = None


def get_default_registry(session_id: str = "default", persist_dir: Optional[str] = None) -> ArtifactRegistry:
    """获取（或创建）模块级默认 registry。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ArtifactRegistry(session_id=session_id, persist_dir=persist_dir)
    return _default_registry
