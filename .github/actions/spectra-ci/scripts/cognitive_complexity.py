"""Cognitive complexity of Python functions (SPCTR-305). Standard library only.

Vendored from ``cognitive_complexity`` 1.3.0 (https://github.com/Melevir/cognitive_complexity,
MIT License, Copyright (c) 2020 Ilya Lebedev), merged into one module and typed.
The package only ships an sdist, unreleased since 2022, which a hash-pinned
install cannot build; the algorithm is small enough to own.

It follows SonarQube's Python implementation of cognitive complexity
(rule python:S3776): +1 per if / for / while / except / ternary, plus
their nesting level; +1 per else; +1 per boolean operator sequence;
nested functions add a nesting level. Three changes from the vendored
package align it with SonarQube: recursion adds nothing, a lambda adds no
nesting level, and an ``elif`` costs a flat 1 (the package counted it as an
``if`` nested in the ``else``). With them, the functions above 15 on
Spectra's ``app/`` are exactly the four SonarQube reports, with the same
values (tests/test_ci_cognitive_complexity.py). Spectra's
linters gate uses it for Python so the verdict matches Sonar's instead of
a cyclomatic proxy.

This file is the single source: the CI runner imports it next to
cognitive_report.py, and Spectra (manual scans, gate tests) loads it in
place through ``app/ci/cognitive_complexity.py``. It must stay standard
library only and must not import ``app``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import NamedTuple, Union

AnyFuncdef = Union[ast.FunctionDef, ast.AsyncFunctionDef]

_CONTROL_FLOW_BREAKERS = (ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler)
_NESTING_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class FunctionComplexity(NamedTuple):
    """Cognitive complexity of one function of a source file."""

    name: str
    start: int
    end: int
    cognitive: int


def cognitive_complexity(funcdef: AnyFuncdef) -> int:
    """Cognitive complexity of a function definition."""
    if _is_decorator(funcdef):
        return cognitive_complexity(funcdef.body[0])  # type: ignore[arg-type]
    return sum(_node_complexity(node, 0) for node in funcdef.body)


def file_functions(source: str) -> Iterator[FunctionComplexity]:
    """Every function and method of a Python source, outermost first.

    Raises SyntaxError when the source does not parse.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, 'end_lineno', None) or node.lineno
            yield FunctionComplexity(node.name, node.lineno, end, cognitive_complexity(node))


def _node_complexity(node: ast.AST, increment_by: int) -> int:
    if isinstance(node, ast.If):
        return _if_chain_complexity(node, increment_by)
    increment_by, base_complexity, iterate_children = _process_node(node, increment_by)
    if not iterate_children:
        return base_complexity
    return base_complexity + sum(_node_complexity(child, increment_by) for child in ast.iter_child_nodes(node))


def _process_node(node: ast.AST, increment_by: int) -> tuple[int, int, bool]:
    if isinstance(node, _CONTROL_FLOW_BREAKERS):
        return _process_control_flow_breaker(node, increment_by)
    if isinstance(node, _NESTING_NODES):
        return increment_by + 1, 0, True
    if isinstance(node, ast.BoolOp):
        return increment_by, sum(1 for inner in ast.walk(node) if isinstance(inner, ast.BoolOp)), False
    return increment_by, 0, True


def _if_chain_complexity(node: ast.If, nesting: int) -> int:
    """An if / elif / else chain, as SonarQube counts it.

    The ``if`` costs 1 plus its nesting level; each ``elif`` and the final
    ``else`` cost a flat 1 (the vendored package counted an ``elif`` as an
    ``if`` nested in the ``else``: +3 instead of +1 at nesting level 2).
    Every branch body is one nesting level deeper than the ``if``.
    """
    total = nesting + 1 + _branch_complexity(node.test, node.body, nesting + 1)
    orelse = node.orelse
    while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        elif_node = orelse[0]
        total += 1 + _branch_complexity(elif_node.test, elif_node.body, nesting + 1)
        orelse = elif_node.orelse
    if orelse:
        total += 1 + sum(_node_complexity(statement, nesting + 1) for statement in orelse)
    return total


def _branch_complexity(test: ast.expr, body: list[ast.stmt], nesting: int) -> int:
    return _node_complexity(test, nesting) + sum(_node_complexity(statement, nesting) for statement in body)


def _process_control_flow_breaker(node: ast.AST, increment_by: int) -> tuple[int, int, bool]:
    """Loops, ternaries and except handlers cost 1 plus their nesting, and nest.

    A loop's ``else`` (``for``/``while`` … ``else``) adds 1 more; a ternary's
    ``else`` is part of the ternary.
    """
    nesting = increment_by + 1
    loop_else = isinstance(node, (ast.For, ast.AsyncFor, ast.While)) and bool(node.orelse)
    return nesting, nesting + (1 if loop_else else 0), True


def _is_decorator(funcdef: AnyFuncdef) -> bool:
    return (
        isinstance(funcdef, ast.FunctionDef)
        and len(funcdef.body) == 2
        and isinstance(funcdef.body[0], ast.FunctionDef)
        and isinstance(funcdef.body[1], ast.Return)
    )
