from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file

# 向量检索相关（可选导入，如果库不存在则降级）
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    print("Warning: sentence-transformers or chromadb not installed. Vector search disabled.", file=sys.stderr)


# 全局向量检索器（单例模式）
_vector_client = None
_vector_collection = None
_sentence_transformer = None


def _get_vector_embedding(text: str) -> list[float]:
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            # 使用项目内的本地模型
            local_path = "/root/siton-tmp/HAL1000/agent/code/models/all-MiniLM-L6-V2"
            _sentence_transformer = SentenceTransformer(local_path)
            print(f"✅ 本地模型加载成功: {local_path}")
        except Exception as e:
            print(f"⚠️ 本地模型加载失败: {e}, 尝试在线下载...")
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-V2')
    return _sentence_transformer.encode(text).tolist()


def _init_vector_db(persist_path: Path) -> bool:
    """初始化向量数据库"""
    global _vector_client, _vector_collection
    
    if not VECTOR_AVAILABLE:
        return False
    
    try:
        # 创建持久化目录
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化客户端
        _vector_client = chromadb.PersistentClient(path=str(persist_path))
        
        # 获取或创建集合
        try:
            _vector_collection = _vector_client.get_collection("memory_embeddings")
        except:
            _vector_collection = _vector_client.create_collection(
                name="memory_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
        return True
    except Exception as e:
        print(f"Vector DB init warning: {e}", file=sys.stderr)
        return False


def _index_memory_for_vector(
    config_path: str,
    memory_id: str,
    content: str,
    metadata: dict,
) -> bool:
    """将记忆文档索引到向量数据库"""
    if not VECTOR_AVAILABLE or _vector_collection is None:
        return False
    
    try:
        # 生成向量
        embedding = _get_vector_embedding(content[:1000])  # 限制长度提升性能
        
        # 存储到向量数据库
        _vector_collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[content[:1000]]
        )
        return True
    except Exception as e:
        print(f"Vector indexing warning for {memory_id}: {e}", file=sys.stderr)
        return False


def _vector_search(query: str, top_k: int = 5, paths: dict = None) -> list[dict]:
    """
    向量检索：根据语义相似度返回最相关的记忆文档
    
    Args:
        query: 查询文本
        top_k: 返回数量
        paths: 记忆路径配置
        
    Returns:
        包含 memory_id 和相似度分数的列表
    """
    if not VECTOR_AVAILABLE or _vector_collection is None:
        return []
    
    try:
        # 生成查询向量
        query_embedding = _get_vector_embedding(query)
        
        # 执行向量检索
        results = _vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results['ids']:
            return []
        
        # 构建结果列表
        docs = []
        for i, doc_id in enumerate(results['ids'][0]):
            docs.append({
                "memory_id": doc_id,
                "content": results['documents'][0][i] if results['documents'] else "",
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0.0
            })
        return docs
    except Exception as e:
        print(f"Vector search warning: {e}", file=sys.stderr)
        return []


def _memory_paths(config_path: str | Path) -> dict[str, Path | int]:
    """读取 memory.yaml 配置，返回记忆相关路径"""
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define a memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    return {
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "max_chars": max_chars,
    }


def _read_index(index_path: Path) -> dict:
    """读取 memory_index.json，返回索引字典"""
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    return index


def _calculate_relevance(content: str, query: str | None) -> float:
    """计算关键词相关性分数"""
    if not query:
        return 0.0
    
    query_words = [w.lower() for w in re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", query) if len(w) > 0]
    if not query_words:
        return 0.0
    
    content_lower = content.lower()
    score = 0.0
    for word in query_words:
        count = content_lower.count(word)
        if count > 0:
            score += count * (1.0 / (len(word) + 1))
            if word in content_lower[:300]:
                score += 2.0
    return score


def _build_memory_markdown(
    title: str,
    memory_id: str,
    conversation_id: str,
    answer: str,
    messages: list,
    trace: dict,
    timestamp: str,
) -> str:
    """构建记忆文档的 Markdown 内容"""
    return (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- timestamp: `{timestamp}`\n\n"
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
    top_k: int | None = None,
    use_vector_search: bool = False,
    enable_error_analysis: bool = False,
) -> dict:
    """
    查找并返回记忆文档。
    
    Args:
        config_path: memory.yaml 配置文件路径
        selected_memory_ids: 用户选择的记忆 ID 列表
        use_global_memory: 是否加载全局记忆
        query: 查询关键词，用于相关性排序
        outdir: 输出目录
        top_k: 返回最相关的前 k 个文档
        use_vector_search: 是否使用向量检索（进阶功能）
        enable_error_analysis: 是否启用错误记忆分析
    
    Returns:
        包含记忆文档、状态信息和错误分析的字典
    """
    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    # 【进阶亮点】：向量检索优先
    vector_results = []
    if use_vector_search and query and VECTOR_AVAILABLE:
        # 初始化向量数据库
        vector_db_path = paths["root"] / ".vector_db"
        if _init_vector_db(vector_db_path):
            # 确保所有记忆都被索引
            for memory_id, metadata in index.items():
                doc_path = paths["root"] / metadata.get("path", "")
                if doc_path.exists():
                    content = read_text(doc_path)
                    _index_memory_for_vector(
                        config_path,
                        memory_id,
                        content,
                        {"memory_id": memory_id, "title": metadata.get("title", "")}
                    )
            
            # 执行向量检索
            vector_results = _vector_search(query, top_k=top_k or 5, paths=paths)
    
    # 构建有序的 memory_id 列表
    ordered_ids = []
    if use_global_memory:
        ordered_ids.extend(sorted(key for key, item in index.items() if item.get("memory_type") == "global"))
    
    # 向量检索结果优先
    vector_ids = [r["memory_id"] for r in vector_results]
    ordered_ids.extend(vector_ids)
    ordered_ids.extend(selected_memory_ids)
    ordered_ids = list(dict.fromkeys(ordered_ids))
    
    candidate_docs = []
    errors = []
    
    # 第一阶段：安全的物理文件读取
    for memory_id in ordered_ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        
        relative_path = metadata.get("path")
        if not isinstance(relative_path, str):
            errors.append({"memory_id": memory_id, "type": "InvalidMetadata", "message": "memory path is missing"})
            continue
        
        document_path = (paths["root"] / relative_path).resolve()
        try:
            document_path.relative_to(paths["root"].resolve())
        except ValueError:
            errors.append({"memory_id": memory_id, "type": "InvalidPath", "message": "memory path escapes root"})
            continue
        
        if not document_path.is_file():
            errors.append({
                "memory_id": memory_id,
                "type": "FileNotFoundError",
                "message": f"memory file not found: {relative_path}"
            })
            continue
        
        original_content = read_text(document_path)
        
        # 计算相关性得分
        keyword_score = _calculate_relevance(original_content, query)
        
        # 如果有向量检索结果，使用向量相似度
        vector_score = 0.0
        for vr in vector_results:
            if vr["memory_id"] == memory_id:
                vector_score = vr.get("similarity_score", 0.0)
                break
        
        # 综合得分：向量检索权重更高
        combined_score = vector_score * 0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score
        
        candidate_docs.append({
            "memory_id": memory_id,
            "metadata": metadata,
            "relative_path": relative_path,
            "original_content": original_content,
            "keyword_score": keyword_score,
            "vector_score": vector_score,
            "relevance_score": combined_score,
            "original_order": len(candidate_docs)
        })
    
    # 智能预算裁剪
    total_raw_chars = sum(len(d["original_content"]) for d in candidate_docs)
    max_chars = int(paths["max_chars"])
    
    if total_raw_chars > max_chars and query:
        candidate_docs.sort(key=lambda x: (-x["relevance_score"], x["original_order"]))
    
    docs = []
    remaining = max_chars
    any_truncated = False
    
    for doc in candidate_docs:
        original = doc["original_content"]
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        
        if included:
            docs.append({
                "memory_id": doc["memory_id"],
                "memory_type": doc["metadata"].get("memory_type"),
                "title": doc["metadata"].get("title", doc["memory_id"]),
                "path": doc["relative_path"],
                "content": included,
                "original_chars": len(original),
                "included_chars": len(included),
                "truncated": truncated,
                "keyword_score": round(doc["keyword_score"], 2),
                "vector_score": round(doc["vector_score"], 4),
                "relevance_score": round(doc["relevance_score"], 2)
            })
            remaining -= len(included)
    
    # Top-K 返回
    if top_k and len(docs) > top_k:
        docs.sort(key=lambda x: -x.get("relevance_score", 0))
        docs = docs[:top_k]
    
    # 恢复原始顺序
    docs.sort(key=lambda x: next(
        (d["original_order"] for d in candidate_docs if d["memory_id"] == x["memory_id"]),
        0
    ))
    
    status = "success"
    if errors and docs:
        status = "partial"
    elif errors:
        status = "error"
    
    # 【进阶亮点】：错误记忆分析
    error_analysis = None
    if enable_error_analysis and docs:
        error_analysis = _analyze_all_memories(docs, query)
    
    result = {
        "status": status,
        "query": query,
        "top_k": top_k,
        "use_vector_search": use_vector_search,
        "selected_memory_docs": docs,
        "max_memory_chars": max_chars,
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": errors,
        "error_analysis": error_analysis,
    }
    
    if outdir:
        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "selected_ids": [item["memory_id"] for item in docs],
                "query": query,
                "top_k": top_k,
                "use_vector_search": use_vector_search,
                "enable_error_analysis": enable_error_analysis,
                "errors": errors,
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result


def _safe_conversation_id(conversation_id: str) -> str:
    """验证 conversation_id 只包含安全字符"""
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def _generate_smart_summary(answer: str) -> str:
    """智能摘要提取器"""
    clean_answer = answer.strip()
    if not clean_answer:
        return ""
    
    lines = [line.strip() for line in clean_answer.split("\n") if line.strip()]
    
    key_points = [l for l in lines if re.match(r"^(\d+\.|-|•|\*)\s*", l)]
    if key_points:
        summary_candidate = " ".join(key_points[:3])
        if len(summary_candidate) <= 200:
            return summary_candidate
    
    first_paragraph = clean_answer.split("\n\n")[0] if "\n\n" in clean_answer else clean_answer
    if len(first_paragraph) <= 200:
        return first_paragraph
    
    return clean_answer[:200]


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict:
    """保存当前对话为记忆文档"""
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    
    paths = _memory_paths(config_path)
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    
    title = f"{save_type.title()} {conversation_id}"
    summary = _generate_smart_summary(answer)
    
    markdown = _build_memory_markdown(
        title=title,
        memory_id=memory_id,
        conversation_id=conversation_id,
        answer=answer,
        messages=messages,
        trace=trace,
        timestamp=now,
    )
    write_text(markdown, target_path)
    
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    created_at = existing.get("created_at", now)
    
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
    }
    write_json(index, paths["index"])
    
    # 【进阶亮点】：保存时自动索引到向量数据库
    if VECTOR_AVAILABLE:
        vector_db_path = paths["root"] / ".vector_db"
        if _init_vector_db(vector_db_path):
            _index_memory_for_vector(
                config_path,
                memory_id,
                markdown,
                {
                    "memory_id": memory_id,
                    "title": title,
                    "memory_type": save_type,
                    "conversation_id": conversation_id
                }
            )
    
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    
    if outdir:
        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "save",
                "status": "success",
                "memory_id": memory_id,
                "save_type": save_type,
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result


def update_memory(
    config_path: str,
    memory_id: str,
    new_messages_path: str,
    new_trace_path: str,
    new_answer_path: str,
    outdir: str | None = None,
    merge_mode: str = "replace",
) -> dict:
    """更新已存在的记忆文档"""
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    
    if memory_id not in index:
        raise ValueError(f"Memory {memory_id} does not exist")
    
    old_metadata = index[memory_id]
    old_path = paths["root"] / old_metadata["path"]
    
    old_content = read_text(old_path) if old_path.exists() else ""
    old_answer = ""
    if old_content:
        match = re.search(r"## Final Answer\s*\n+(.*?)(?=\n##|$)", old_content, re.DOTALL)
        if match:
            old_answer = match.group(1).strip()
    
    new_messages = read_json(new_messages_path)
    new_trace = read_json(new_trace_path)
    new_answer = read_text(new_answer_path).strip()
    
    if not isinstance(new_messages, list) or not isinstance(new_trace, dict):
        raise ValueError("messages must be an array and trace must be a dict")
    
    now = now_iso()
    conversation_id = old_metadata.get("conversation_id", "")
    
    if merge_mode == "append" and old_answer:
        final_answer = old_answer + "\n\n---\n\n## Update at " + now + "\n\n" + new_answer
    else:
        final_answer = new_answer
    
    title = old_metadata.get("title", f"Memory {memory_id}")
    markdown = _build_memory_markdown(
        title=title,
        memory_id=memory_id,
        conversation_id=conversation_id,
        answer=final_answer,
        messages=new_messages,
        trace=new_trace,
        timestamp=now,
    )
    
    write_text(markdown, old_path)
    
    index[memory_id]["updated_at"] = now
    index[memory_id]["summary"] = _generate_smart_summary(final_answer)
    write_json(index, paths["index"])
    
    # 【进阶亮点】：更新时重新索引向量数据库
    if VECTOR_AVAILABLE:
        vector_db_path = paths["root"] / ".vector_db"
        if _init_vector_db(vector_db_path):
            _index_memory_for_vector(
                config_path,
                memory_id,
                markdown,
                {
                    "memory_id": memory_id,
                    "title": title,
                    "memory_type": old_metadata.get("memory_type", ""),
                    "conversation_id": conversation_id
                }
            )
    
    conflict_detected = False
    if old_answer and new_answer:
        old_keywords = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]{3,}", old_answer[:200]))
        new_keywords = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]{3,}", new_answer[:200]))
        overlap = old_keywords & new_keywords
        if len(overlap) > 3:
            conflict_detected = True
    
    result = {
        "status": "success",
        "memory_id": memory_id,
        "action": "updated",
        "merge_mode": merge_mode,
        "conflict_detected": conflict_detected,
        "updated_at": now,
        "path": old_metadata["path"],
        "old_answer_preview": old_answer[:100] + "..." if len(old_answer) > 100 else old_answer,
        "new_answer_preview": new_answer[:100] + "..." if len(new_answer) > 100 else new_answer,
    }
    
    if outdir:
        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(result, output_dir / "updated_memory.json")
        append_jsonl(
            {
                "timestamp": now,
                "operation": "update",
                "status": "success",
                "memory_id": memory_id,
                "merge_mode": merge_mode,
                "conflict_detected": conflict_detected,
            },
            output_dir / "memory_log.jsonl",
        )
    
    return result


def parse_bool(value: str) -> bool:
    """解析布尔值字符串"""
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Advanced local memory engine for AI Agents with vector search and error analysis.",
        epilog="""
Examples:
  # 关键词检索
  python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_conversation_conv_000 --use_global_memory true --query "Agent 系统如何调用工具？" --top_k 3 --outdir ../outputs/B5_memory

  # 向量检索（语义搜索）
  python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_conversation_conv_000 --use_global_memory true --query "智能体如何利用工具完成任务" --top_k 2 --use_vector_search true --outdir ../outputs/B5_memory/vector_test

  # 向量检索 + 错误记忆分析
  python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_conversation_conv_000 --use_global_memory true --query "智能体如何利用工具完成任务" --top_k 2 --use_vector_search true --enable_error_analysis true --outdir ../outputs/B5_memory/error_analysis_test
        """
    )
    parser.add_argument("--config", required=True, help="Path to memory.yaml")
    
    # 查找模式参数
    parser.add_argument("--select_memory_ids", nargs="*", help="Memory IDs to select")
    parser.add_argument("--use_global_memory", type=parse_bool, help="Whether to load global memory")
    parser.add_argument("--query", help="Search query for relevance ranking")
    parser.add_argument("--top_k", type=int, help="Return top k most relevant documents")
    parser.add_argument("--use_vector_search", type=parse_bool, default=False,
                        help="Use vector search (semantic search) for better relevance")
    parser.add_argument("--enable_error_analysis", type=parse_bool, default=False,
                        help="Enable error memory analysis")
    
    # 保存模式参数
    parser.add_argument("--save_type", choices=["conversation", "global"], help="Save type")
    parser.add_argument("--save_input_path", help="Path to memory_save_input.json")
    
    # 更新模式参数
    parser.add_argument("--update_memory_id", help="Memory ID to update")
    parser.add_argument("--update_input_path", help="Path to update input JSON")
    parser.add_argument("--merge_mode", choices=["replace", "append"], default="replace",
                        help="Merge mode for update: replace or append")
    
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行主入口"""
    args = build_parser().parse_args(argv)
    
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        
        # 模式1：更新记忆
        if args.update_memory_id and args.update_input_path:
            input_path = resolve_cli_path(args.update_input_path)
            payload = read_json(input_path)
            base = input_path.parent
            
            result = update_memory(
                str(config_path),
                args.update_memory_id,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
                merge_mode=args.merge_mode,
            )
            print(f"Updated memory: {outdir / 'updated_memory.json'}")
            return 0
        
        # 模式2：保存记忆
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            
            base = input_path.parent
            result = save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(f"Saved memory: {outdir / 'saved_memory.json'}")
            return 0
        
        # 模式3：查找记忆
        if args.select_memory_ids is None and args.use_global_memory is None:
            raise ValueError("Select mode requires --select_memory_ids or --use_global_memory")
        
        result = load_memory(
            str(config_path),
            args.select_memory_ids or [],
            bool(args.use_global_memory),
            args.query,
            str(outdir),
            top_k=args.top_k,
            use_vector_search=bool(args.use_vector_search),
            enable_error_analysis=bool(args.enable_error_analysis),  # 新增
        )
        print(f"Selected memory: {outdir / 'selected_memory.json'}")
        return 0
    
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _analyze_memory_errors(
    memory_content: str,
    memory_id: str,
    query: str | None = None
) -> dict:
    """
    错误记忆分析功能
    
    检测记忆中的常见问题：
    1. 时间敏感信息（可能过时）
    2. 不确定性表述（可能不准确）
    3. 与当前查询冲突的内容
    4. 内容长度异常（可能不完整）
    
    Returns:
        {
            "has_errors": bool,
            "severity": "low" | "medium" | "high",
            "warnings": list[str],
            "suggestions": list[str],
            "impact_score": float  # 0-1, 越高表示对回答影响越大
        }
    """
    warnings = []
    suggestions = []
    severity = "low"
    impact_score = 0.0
    
    # 1. 检测时间敏感关键词（可能过时）
    time_keywords = [
        "最新", "当前", "现在", "目前", "最近", "今年", "本月", "今日",
        "latest", "current", "now", "recent", "2024", "2025", "2026"
    ]
    time_hits = [kw for kw in time_keywords if kw in memory_content.lower()]
    if time_hits:
        warnings.append(f"包含时间敏感词: {', '.join(time_hits[:3])}，信息可能已过时")
        impact_score += 0.15
        severity = "medium"
        suggestions.append("建议验证信息时效性，特别是日期和版本信息")
    
    # 2. 检测不确定性表述（可能不准确）
    uncertainty_keywords = [
        "可能", "也许", "大概", "似乎", "据说", "听说", "传闻",
        "maybe", "perhaps", "probably", "might", "seems", "apparently"
    ]
    uncertainty_hits = [kw for kw in uncertainty_keywords if kw in memory_content.lower()]
    if uncertainty_hits:
        warnings.append(f"包含不确定表述: {', '.join(uncertainty_hits[:3])}")
        impact_score += 0.1
        if severity == "low":
            severity = "medium"
        suggestions.append("建议寻找更确定的来源验证这些信息")
    
    # 3. 检测内容长度异常
    content_length = len(memory_content)
    if content_length < 50:
        warnings.append(f"内容过短 ({content_length} 字符)，可能信息不完整")
        impact_score += 0.2
        severity = "medium"
        suggestions.append("建议补充更多上下文信息")
    elif content_length > 5000:
        warnings.append(f"内容过长 ({content_length} 字符)，可能包含冗余信息")
        impact_score += 0.05
        suggestions.append("建议精简内容，提取核心要点")
    
    # 4. 检测与查询的语义冲突（简单版本）
    if query and query.lower() in memory_content.lower():
        # 查询词出现在记忆中，但不一定冲突，只是标记
        pass
    
    # 5. 检测缺失的关键信息
    required_fields = ["模型", "工具", "记忆", "执行", "循环"]
    missing_fields = [f for f in required_fields if f not in memory_content]
    if missing_fields:
        warnings.append(f"可能缺少关键概念: {', '.join(missing_fields[:3])}")
        impact_score += 0.1
        suggestions.append("建议补充缺失的概念说明")
    
    # 6. 检测矛盾表述（简单版本）
    contradiction_pairs = [
        ("支持", "不支持"),
        ("可以", "不可以"),
        ("是", "不是"),
        ("有", "没有"),
        ("正确", "错误"),
        ("true", "false")
    ]
    contradictions = []
    for pos, neg in contradiction_pairs:
        if pos in memory_content.lower() and neg in memory_content.lower():
            contradictions.append(f"{pos}/{neg}")
    if contradictions:
        warnings.append(f"检测到可能的矛盾表述: {', '.join(contradictions[:2])}")
        impact_score += 0.25
        severity = "high"
        suggestions.append("建议检查并统一表述，避免前后矛盾")
    
    # 计算最终严重程度
    if impact_score >= 0.4:
        severity = "high"
    elif impact_score >= 0.2:
        severity = "medium"
    
    # 限制 impact_score 不超过 1.0
    impact_score = min(impact_score, 1.0)
    
    return {
        "has_errors": len(warnings) > 0,
        "severity": severity,
        "warnings": warnings,
        "suggestions": suggestions,
        "impact_score": round(impact_score, 2),
        "memory_id": memory_id
    }


def _analyze_all_memories(
    docs: list,
    query: str | None = None
) -> dict:
    """
    分析所有选中的记忆文档，汇总错误信息
    """
    if not docs:
        return {
            "has_errors": False,
            "overall_severity": "none",
            "memory_analyses": [],
            "summary": "没有记忆需要分析",
            "recommendations": []
        }
    
    analyses = []
    overall_impact = 0.0
    all_warnings = []
    all_suggestions = []
    
    for doc in docs:
        analysis = _analyze_memory_errors(
            doc.get("content", ""),
            doc.get("memory_id", ""),
            query
        )
        analyses.append(analysis)
        overall_impact += analysis.get("impact_score", 0)
        all_warnings.extend(analysis.get("warnings", []))
        all_suggestions.extend(analysis.get("suggestions", []))
    
    # 计算总体严重程度
    avg_impact = overall_impact / len(docs) if docs else 0
    if avg_impact >= 0.4:
        overall_severity = "high"
    elif avg_impact >= 0.2:
        overall_severity = "medium"
    else:
        overall_severity = "low"
    
    # 生成总结和建议
    summary = f"分析了 {len(docs)} 个记忆文档"
    if all_warnings:
        summary += f"，发现 {len(all_warnings)} 个潜在问题"
    else:
        summary += "，未发现明显问题"
    
    recommendations = []
    if all_warnings:
        recommendations.append("建议优先处理高影响的问题")
        recommendations.append("使用更新后的信息替换过时内容")
    
    # 如果有高严重程度的问题，添加额外建议
    high_severity_count = sum(1 for a in analyses if a.get("severity") == "high")
    if high_severity_count > 0:
        recommendations.append(f"有 {high_severity_count} 个记忆存在高风险问题，建议立即更新")
    
    return {
        "has_errors": len(all_warnings) > 0,
        "overall_severity": overall_severity,
        "memory_analyses": analyses,
        "summary": summary,
        "recommendations": recommendations,
        "total_warnings": len(all_warnings)
    }


if __name__ == "__main__":
    raise SystemExit(main())
