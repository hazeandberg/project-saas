from __future__ import annotations

import ast
from typing import Any, Dict, Optional

from src.agent.rules_loader import Policy, Rule


class UnsafeExpressionError(Exception):
    pass


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.NotIn,
    ast.List,
    ast.Tuple,
)


def _validate_ast(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise UnsafeExpressionError(
                f"Disallowed expression node: {type(child).__name__}"
            )


def _eval_expr(expr: str, features: Dict[str, Any]) -> bool:
    """
    Safely evaluates a boolean expression against a features dict.

    Supported:
      - ==, !=
      - and, or
      - in [...]
      - true/false/null (YAML-style) are supported as aliases
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Invalid expression syntax: {expr}") from e

    _validate_ast(tree)
    compiled = compile(tree, filename="<policy>", mode="eval")

    # YAML-style literals compatibility
    ctx: Dict[str, Any] = {
        "__builtins__": {},
        "true": True,
        "false": False,
        "null": None,
    }
    ctx.update(features)

    return bool(eval(compiled, ctx, {}))


def apply_policy(policy: Policy, features: Dict[str, Any]) -> Optional[Rule]:
    """
    Returns the first matching rule by ascending priority.
    """
    for rule in policy.rules:
        try:
            if _eval_expr(rule.expr, features):
                return rule
        except UnsafeExpressionError:
            continue
    return None
