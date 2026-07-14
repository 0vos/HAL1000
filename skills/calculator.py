from __future__ import annotations

import ast
import math
import operator

from skills_error_codes import ErrorCode, attach_error_code


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 30:
            raise attach_error_code(ValueError("exponent magnitude must not exceed 30"), ErrorCode.OVERFLOW)
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(result, complex) or not math.isfinite(float(result)) or abs(result) > 1e100:
            raise attach_error_code(ValueError("calculation result is out of range"), ErrorCode.OVERFLOW)
        return result
    raise attach_error_code(ValueError(f"unsupported expression element: {type(node).__name__}"), ErrorCode.UNSUPPORTED_TYPE)


def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise attach_error_code(ValueError("expression must be a non-empty string"), ErrorCode.INVALID_INPUT)
    if len(expression) > 200:
        raise attach_error_code(ValueError("expression is too long"), ErrorCode.INVALID_INPUT)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise attach_error_code(ValueError("invalid arithmetic expression"), ErrorCode.INVALID_INPUT) from exc
    return {"result": _evaluate(tree)}
