"""Auto-generate OpenAI function schema from a Python function.

Reads the function's signature and docstring to produce a JSON schema that
matches the format produced by b3_tool_layer.get_tools_schema.

Supported type annotations:
    int, float, bool, str, list, dict, optional types via Union[T, None]

Description extraction:
    - If the docstring starts with a one-line summary, that becomes the
      function description.
    - If a `Parameters` or `Args` section exists, each parameter is annotated
      with its description from that section.
    - Returns are described from a `Returns` section when present.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, get_args, get_origin


_PYTHON_TYPE_TO_JSON = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    list: "array",
    dict: "object",
}


def _python_type_to_json(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    # Optional[X] / Union[X, None] -> unwrap to X
    if origin is type(None):
        return "object"
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
            origin = get_origin(annotation)
            args = get_args(annotation)
    if annotation in _PYTHON_TYPE_TO_JSON:
        return _PYTHON_TYPE_TO_JSON[annotation]
    if origin in (list, dict):
        return _PYTHON_TYPE_TO_JSON[origin]
    if origin is None and hasattr(annotation, "__name__") and annotation.__name__ in _PYTHON_TYPE_TO_JSON:
        return _PYTHON_TYPE_TO_JSON[annotation.__name__]
    return "string"


def _parse_docstring(doc: str | None) -> dict:
    if not doc:
        return {"summary": "", "params": {}, "returns": ""}
    doc = doc.strip()
    summary = doc.split("\n\n", 1)[0].strip()
    params: dict[str, str] = {}
    returns = ""
    sections = re.split(r"\n(?=(?:Parameters|Args|Returns|Yields|Notes|Examples)\b)", doc)
    for section in sections:
        head, _, body = section.partition("\n")
        head = head.strip().rstrip(":")
        body = body.strip()
        if head in {"Parameters", "Args"}:
            for line in body.splitlines():
                m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
                if m:
                    params[m.group(1)] = m.group(2).strip()
        elif head in {"Returns", "Yields"}:
            returns = body
    return {"summary": summary, "params": params, "returns": returns}


def schema_from_function(func: Callable, name: str | None = None) -> dict:
    """Generate an OpenAI-style function schema from a Python callable."""
    name = name or func.__name__
    sig = inspect.signature(func)
    try:
        resolved_hints = getattr(func, "__annotations__", {}) or {}
        try:
            resolved_hints = inspect.get_annotations(func, eval_str=True)
        except Exception:
            # Fallback for Python < 3.10 / no future-annotations
            try:
                resolved_hints = inspect.get_annotations(func)
            except Exception:
                resolved_hints = {}
    except Exception:
        resolved_hints = {}
    doc = _parse_docstring(inspect.getdoc(func))
    properties: dict[str, dict] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = resolved_hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty or annotation is None:
            annotation = str
        json_type = _python_type_to_json(annotation)
        description = doc["params"].get(param_name, "")
        prop: dict[str, Any] = {"type": json_type, "description": description or f"Parameter {param_name}"}
        if json_type == "array":
            item_annotation = getattr(annotation, "__args__", None) or [str]
            if isinstance(item_annotation, tuple) and item_annotation:
                prop["items"] = {"type": _python_type_to_json(item_annotation[0])}
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc["summary"] or f"Function {name}",
            "parameters": parameters,
        },
    }


def schemas_from_module(module, names: list[str] | None = None) -> list[dict]:
    """Generate schemas for every public callable in a module."""
    targets = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        if names and attr_name not in names:
            continue
        attr = getattr(module, attr_name)
        if not callable(attr):
            continue
        if not inspect.isfunction(attr):
            continue
        targets.append(attr)
    return [schema_from_function(func, name=func.__name__) for func in targets]