from __future__ import annotations

import ast
import math

import pytest

from counterpatch.analyzer import collect_functions
from counterpatch.candidates import (
    INT,
    STR,
    CallPlan,
    TypeSpec,
    build_call_plan,
    discover_hints,
    parse_annotation,
    seed_examples,
    value_pool,
)
from counterpatch.models import ChangedFunction


def annotation(text: str) -> TypeSpec | None:
    return parse_annotation(ast.parse(text, mode="eval").body)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("int", INT),
        ("str", STR),
        ("'int'", INT),
        ("Optional[int]", TypeSpec("optional", (INT,))),
        ("int | None", TypeSpec("optional", (INT,))),
        ("list[str]", TypeSpec("list", (STR,))),
        ("typing.List[int]", TypeSpec("list", (INT,))),
        ("dict[str, int]", TypeSpec("dict", (STR, INT))),
        ("tuple[int, str]", TypeSpec("tuple", (INT, STR))),
        ("tuple[int, ...]", TypeSpec("tuple", (INT,), variadic=True)),
        ("Literal['a', 'b']", TypeSpec("literal", values=("a", "b"))),
        ("int | str", TypeSpec("union", (INT, STR))),
    ],
)
def test_parse_annotation_supported(text: str, expected: TypeSpec) -> None:
    assert annotation(text) == expected


@pytest.mark.parametrize("text", ["PaymentGateway", "Callable[[int], int]", "list[Decimal]", "dict[Foo, int]"])
def test_parse_annotation_unsupported(text: str) -> None:
    assert annotation(text) is None


def function(source: str, name: str):
    return collect_functions(source)[name]


def test_numeric_boundaries_are_discovered() -> None:
    node = function("def f(length: int) -> bool:\n    if length <= 32:\n        return True\n    return False\n", "f").node
    hints = discover_hints([node], {"length"})
    assert hints.numbers == {"length": {32}}
    pool = value_pool(INT, "length", hints)
    assert pool[:3] == [0, 1, -1]
    assert {31, 32, 33} <= set(pool)
    assert pool.index(31) < pool.index(1000)


def test_length_boundaries_generate_strings_around_limit() -> None:
    node = function(
        "def v(username: str) -> bool:\n    if not username:\n        raise ValueError\n"
        "    if len(username) > 20:\n        raise ValueError\n    return username in ('admin', 'root')\n",
        "v",
    ).node
    hints = discover_hints([node], {"username"})
    assert hints.lengths == {"username": {20}}
    assert hints.strings == {"username": ["admin", "root"]}
    pool = value_pool(STR, "username", hints)
    lengths = {len(value) for value in pool}
    assert {19, 20, 21} <= lengths
    for edge_case in ["", " ", "\n", "é", "\x00", "admin"]:
        assert edge_case in pool
    assert any(len(value) >= 1000 for value in pool)


def test_other_type_pools() -> None:
    hints = discover_hints([], set())
    assert value_pool(TypeSpec("bool"), "flag", hints) == [False, True]
    assert value_pool(TypeSpec("optional", (INT,)), "x", hints)[0] is None
    lists = value_pool(TypeSpec("list", (INT,)), "items", hints)
    assert [] in lists and [0] in lists and [0, 0] in lists
    floats = value_pool(TypeSpec("float"), "x", hints)
    assert any(isinstance(value, float) and math.isnan(value) for value in floats)
    assert math.inf in floats


def _changed(base: str, patch: str, name: str) -> ChangedFunction:
    return ChangedFunction(
        file="mod.py",
        base_file="mod.py",
        qualified_name=name,
        status="modified",
        diff="",
        patch=collect_functions(patch).get(name),
        base=collect_functions(base).get(name),
    )


def test_build_call_plan_and_deterministic_sweep_order() -> None:
    source = "def withdraw(balance: int, amount: int) -> int:\n    if amount <= 0:\n        raise ValueError\n    return balance - amount\n"
    plan = build_call_plan(_changed(source, source, "withdraw"))
    assert isinstance(plan, CallPlan)
    assert [slot.name for slot in plan.slots] == ["balance", "amount"]
    sweep = list(plan.sweep(6))
    assert sweep[0] == {"balance": 0, "amount": 0}
    assert len(sweep) == 6
    everything = list(plan.sweep(10_000))
    assert len(everything) == len(plan.slots[0].pool) * len(plan.slots[1].pool)
    assert len({tuple(args.items()) for args in everything}) == len(everything)


@pytest.mark.parametrize(
    ("base", "patch", "reason"),
    [
        ("def f(x): return x\n", "def f(x): return x\n", "no type annotation"),
        ("def f(g: Gateway) -> int: return 1\n", "def f(g: Gateway) -> int: return 2\n", "cannot construct"),
        ("x = 1\n", "def f(x: int) -> int: return x\n", "new function"),
        ("def f(x: int) -> int: return x\n", "def f(y: int) -> int: return y\n", "signature changed"),
        (
            "class A:\n    @property\n    def p(self) -> int: return 1\n",
            "class A:\n    @property\n    def p(self) -> int: return 2\n",
            "property",
        ),
    ],
)
def test_build_call_plan_skip_reasons(base: str, patch: str, reason: str) -> None:
    name = "A.p" if "class A" in patch else "f"
    result = build_call_plan(_changed(base, patch, name))
    assert isinstance(result, str)
    assert reason in result


def test_unannotated_parameters_inferred_from_defaults_and_usage() -> None:
    source = "def f(limit=10, name=None, *, text):\n    return text.strip()[:limit]\n"
    plan = build_call_plan(_changed(source, source, "f"))
    assert isinstance(plan, CallPlan)
    assert {slot.name: slot.spec.kind for slot in plan.slots} == {"limit": "int", "text": "str"}
    assert plan.fixed_defaults == ["name"]


def test_seed_examples_from_existing_tests() -> None:
    source = "def withdraw(balance: int, amount: int) -> int:\n    return balance - amount\n"
    seeds = [([100, 30], {}), ([], {"balance": 5, "amount": 7}), (["x", 1], {})]
    plan = build_call_plan(_changed(source, source, "withdraw"), seeds)
    assert isinstance(plan, CallPlan)
    assert seed_examples(plan, seeds) == [{"balance": 100, "amount": 30}, {"balance": 5, "amount": 7}]
    assert 100 in plan.slots[0].pool
