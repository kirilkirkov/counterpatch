"""Deterministic adversarial candidate generation for changed functions.

Given a changed function, this module decides whether CounterPatch can call it
automatically, and if so builds a :class:`CallPlan`: one :class:`ArgSlot` per
parameter, each with an ordered pool of adversarial values (simplest and most
boundary-relevant first) and a Hypothesis strategy biased towards that pool.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from hypothesis import strategies as st

from counterpatch.models import ChangedFunction, FunctionInfo, Param

LARGE_INTS = [2**31 - 1, -(2**31), 2**63, -(2**63)]
_STRING_METHODS = {
    "strip",
    "lstrip",
    "rstrip",
    "lower",
    "upper",
    "casefold",
    "startswith",
    "endswith",
    "split",
    "replace",
    "isdigit",
    "isalpha",
    "isalnum",
    "isspace",
    "encode",
    "format",
    "title",
    "splitlines",
}
_LIST_METHODS = {"append", "extend", "pop", "insert", "sort", "reverse"}
_MAX_POOL = 24


@dataclass(frozen=True)
class TypeSpec:
    """A small, closed description of the argument types CounterPatch can construct."""

    kind: str  # int float str bool bytes none any list tuple dict set optional literal union
    items: tuple[TypeSpec, ...] = ()
    values: tuple[Any, ...] = ()
    variadic: bool = False

    def describe(self) -> str:
        if self.kind in {"list", "set", "optional"} and self.items:
            return f"{self.kind}[{self.items[0].describe()}]"
        if self.kind == "dict" and len(self.items) == 2:
            return f"dict[{self.items[0].describe()}, {self.items[1].describe()}]"
        return self.kind


INT, FLOAT, STR, BOOL, BYTES, NONE, ANY = (TypeSpec(kind) for kind in ("int", "float", "str", "bool", "bytes", "none", "any"))
_SIMPLE_NAMES = {"int": INT, "float": FLOAT, "str": STR, "bool": BOOL, "bytes": BYTES, "None": NONE, "Any": ANY, "object": ANY}
_SEQUENCE_NAMES = {"list", "List", "Sequence", "Iterable", "Collection", "MutableSequence"}
_SET_NAMES = {"set", "Set", "frozenset", "FrozenSet", "AbstractSet", "MutableSet"}
_DICT_NAMES = {"dict", "Dict", "Mapping", "MutableMapping"}


def parse_annotation(node: ast.expr | None) -> TypeSpec | None:
    """Translate an annotation AST into a :class:`TypeSpec`, or ``None`` if unsupported."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if node.value is None:
            return NONE
        if isinstance(node.value, str):
            try:
                return parse_annotation(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return None
        return None
    name = _dotted_tail(node)
    if name is not None:
        if name in _SIMPLE_NAMES:
            return _SIMPLE_NAMES[name]
        if name in _SEQUENCE_NAMES:
            return TypeSpec("list", (ANY,))
        if name in _SET_NAMES:
            return TypeSpec("set", (ANY,))
        if name in _DICT_NAMES:
            return TypeSpec("dict", (STR, ANY))
        if name in {"tuple", "Tuple"}:
            return TypeSpec("tuple", (ANY,), variadic=True)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union([parse_annotation(node.left), parse_annotation(node.right)])
    if isinstance(node, ast.Subscript):
        base = _dotted_tail(node.value)
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base == "Optional":
            inner = parse_annotation(args[0])
            return TypeSpec("optional", (inner,)) if inner else None
        if base == "Union":
            return _union([parse_annotation(arg) for arg in args])
        if base == "Literal":
            values = []
            for arg in args:
                try:
                    values.append(ast.literal_eval(arg))
                except ValueError:
                    return None
            return TypeSpec("literal", values=tuple(values))
        if base in _SEQUENCE_NAMES:
            inner = parse_annotation(args[0])
            return TypeSpec("list", (inner,)) if inner else None
        if base in _SET_NAMES:
            inner = parse_annotation(args[0])
            return TypeSpec("set", (inner,)) if inner and inner.kind in {"int", "str", "bool", "bytes", "float"} else None
        if base in _DICT_NAMES and len(args) == 2:
            key, value = parse_annotation(args[0]), parse_annotation(args[1])
            if key in (STR, INT) and value:
                return TypeSpec("dict", (key, value))
            return None
        if base in {"tuple", "Tuple"}:
            if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:
                inner = parse_annotation(args[0])
                return TypeSpec("tuple", (inner,), variadic=True) if inner else None
            items = [parse_annotation(arg) for arg in args]
            return TypeSpec("tuple", tuple(items)) if all(items) else None  # type: ignore[arg-type]
    return None


def _dotted_tail(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"typing", "collections", "abc", "builtins", "t"}
    ):
        return node.attr
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        return node.attr if ast.unparse(node.value) == "collections.abc" else None
    return None


def _union(members: list[TypeSpec | None]) -> TypeSpec | None:
    if any(member is None for member in members):
        return None
    flat: list[TypeSpec] = []
    for member in members:
        assert member is not None
        flat.extend(member.items if member.kind == "union" else [member])
    non_none = [member for member in flat if member != NONE]
    if len(non_none) < len(flat):
        inner = non_none[0] if len(non_none) == 1 else TypeSpec("union", tuple(non_none))
        return TypeSpec("optional", (inner,))
    return TypeSpec("union", tuple(flat))


def infer_from_default(node: ast.expr | None) -> TypeSpec | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    for python_type, spec in ((bool, BOOL), (int, INT), (float, FLOAT), (str, STR), (bytes, BYTES)):
        if isinstance(value, python_type):
            return spec
    if isinstance(value, list):
        return TypeSpec("list", (ANY,))
    return None


def infer_from_usage(function: ast.AST, name: str) -> TypeSpec | None:
    """Guess a primitive type for an unannotated parameter from how it is used."""
    for node in ast.walk(function):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for left, right in zip(operands, operands[1:]):
                for side, other in ((left, right), (right, left)):
                    if isinstance(side, ast.Name) and side.id == name:
                        number = _number(other)
                        if number is not None:
                            return FLOAT if isinstance(number, float) else INT
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == name:
                if node.func.attr in _STRING_METHODS:
                    return STR
                if node.func.attr in _LIST_METHODS:
                    return TypeSpec("list", (ANY,))
    return None


@dataclass
class Hints:
    """Boundary information extracted from the base and patched function bodies."""

    numbers: dict[str, set[float]] = field(default_factory=dict)
    lengths: dict[str, set[int]] = field(default_factory=dict)
    strings: dict[str, list[str]] = field(default_factory=dict)
    constants: set[float] = field(default_factory=set)
    seeds: dict[str, list[Any]] = field(default_factory=dict)

    def add_number(self, name: str, value: float) -> None:
        self.numbers.setdefault(name, set()).add(value)

    def add_length(self, name: str, value: int) -> None:
        if 0 <= value <= 10_000:
            self.lengths.setdefault(name, set()).add(value)

    def add_string(self, name: str, value: str) -> None:
        bucket = self.strings.setdefault(name, [])
        if value not in bucket and len(value) <= 200:
            bucket.append(value)

    def add_seed(self, name: str, value: Any) -> None:
        bucket = self.seeds.setdefault(name, [])
        if not any(_same_value(value, existing) for existing in bucket):
            bucket.append(value)


def _number(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        return -inner if inner is not None else None
    return None


def _param_reference(node: ast.expr, params: set[str]) -> tuple[str, str] | None:
    """Return ``(param, "value"|"len")`` if ``node`` is ``param`` or ``len(param)``."""
    if isinstance(node, ast.Name) and node.id in params:
        return node.id, "value"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in params
    ):
        return node.args[0].id, "len"
    return None


def discover_hints(functions: list[ast.AST], params: set[str], constants: dict[str, float] | None = None) -> Hints:
    """Collect literal boundaries that parameters are compared against.

    ``if len(username) > 32`` yields a length boundary of 32 for ``username``;
    ``if amount <= 0`` yields a numeric boundary of 0 for ``amount``; string
    comparisons and ``in`` checks against literal collections yield string candidates.
    """
    hints = Hints()
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node, ast.Compare):
                _hints_from_compare(node, params, hints, constants or {})
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                reference = _param_reference(node.value, params)
                bound = _number(node.slice.upper) if node.slice.upper is not None else None
                if reference and isinstance(bound, int):
                    hints.add_length(reference[0], bound)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"startswith", "endswith"}:
                reference = _param_reference(node.func.value, params)
                if reference and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    hints.add_string(reference[0], node.args[0].value)
            number = _number(node) if isinstance(node, ast.expr) else None
            if number is not None and len(hints.constants) < 16:
                hints.constants.add(number)
    return hints


def _hints_from_compare(node: ast.Compare, params: set[str], hints: Hints, constants: dict[str, float]) -> None:
    operands = [node.left, *node.comparators]
    for operator, (left, right) in zip(node.ops, zip(operands, operands[1:])):
        for side, other in ((left, right), (right, left)):
            reference = _param_reference(side, params)
            if reference is None:
                continue
            name, mode = reference
            if isinstance(operator, (ast.In, ast.NotIn)) and isinstance(other, (ast.Tuple, ast.List, ast.Set)):
                for element in other.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        hints.add_string(name, element.value)
                    elif (number := _number(element)) is not None:
                        hints.add_number(name, number)
                continue
            number = _number(other)
            if number is None and isinstance(other, ast.Name) and other.id in constants:
                number = constants[other.id]
            if number is not None:
                if mode == "len" and isinstance(number, int):
                    hints.add_length(name, number)
                elif mode == "value":
                    hints.add_number(name, number)
            elif isinstance(other, ast.Constant) and isinstance(other.value, str) and mode == "value":
                hints.add_string(name, other.value)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and repr(left) == repr(right)


def _dedupe(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(_same_value(value, existing) for existing in unique):
            unique.append(value)
    return unique


def value_pool(spec: TypeSpec, name: str, hints: Hints) -> list[Any]:
    """Ordered adversarial values for one parameter: simplest first, then boundaries."""
    generated = _dedupe(_pool(spec, name, hints))[:_MAX_POOL]
    seeds = [seed for seed in hints.seeds.get(name, []) if conforms(seed, spec)]
    return _dedupe(generated + seeds[:4])


def _pool(spec: TypeSpec, name: str, hints: Hints) -> list[Any]:
    kind = spec.kind
    if kind == "int":
        boundaries = sorted(hints.numbers.get(name, set()))
        around = [value for bound in boundaries for value in (math.floor(bound) - 1, math.floor(bound), math.ceil(bound) + 1)]
        constants = [int(value) for value in sorted(hints.constants) if abs(value) < 10**12]
        return _dedupe([0, 1, -1, *around, *constants, 2, -2, 10, 100, 1000, *LARGE_INTS])
    if kind == "float":
        boundaries = sorted(hints.numbers.get(name, set()))
        around = [float(value) for bound in boundaries for value in (bound - 1, bound - 0.5, bound, bound + 0.5, bound + 1)]
        return _dedupe([0.0, 1.0, -1.0, 0.5, -0.5, *around, -0.0, 1e-9, -1e-9, 1e9, -1e9, 1e308, math.inf, -math.inf, math.nan])
    if kind == "str":
        lengths = sorted(hints.lengths.get(name, set()))
        around = ["a" * size for bound in lengths for size in (bound - 1, bound, bound + 1) if size >= 0]
        return _dedupe(
            ["", "a", *around, *hints.strings.get(name, []), " ", "\n", "A", "0", "-1", " a ", "é", "日本語", "😀", "\x00", "a" * 1000]
        )
    if kind == "bool":
        return [False, True]
    if kind == "bytes":
        return [b"", b"a", b"\x00", b"\xff\xfe"]
    if kind == "none":
        return [None]
    if kind == "any":
        return [None, 0, "", [], 1, -1, "a", True, 0.5]
    if kind == "literal":
        return list(spec.values)
    if kind == "optional":
        return _dedupe([None, *_pool(spec.items[0], name, hints)])
    if kind == "union":
        pools = [_pool(item, name, hints) for item in spec.items]
        mixed = [value for group in zip(*pools) for value in group] if pools else []
        return _dedupe(mixed + [value for pool in pools for value in pool])
    if kind in {"list", "tuple", "set"} and (spec.variadic or kind != "tuple"):
        elements = _pool(spec.items[0], name + "[]", hints)[:4] if spec.items else [None]
        first = elements[0]
        second = elements[1] if len(elements) > 1 else first
        lengths = sorted(hints.lengths.get(name, set()))
        sized = [[first] * size for bound in lengths for size in (bound - 1, bound, bound + 1) if 0 <= size <= 1000]
        lists = [[], [first], [second], [first, first], [second, first], [first, second], *sized, [first] * 100]
        if kind == "tuple":
            return _dedupe([tuple(values) for values in lists])
        if kind == "set":
            return _dedupe([set(values) for values in lists if _hashable(values)])
        return _dedupe(lists)
    if kind == "tuple":
        pools = [_pool(item, name, hints)[:3] for item in spec.items]
        base = [pool[0] for pool in pools]
        variants = [tuple(base)]
        for index, pool in enumerate(pools):
            for value in pool[1:]:
                variant = list(base)
                variant[index] = value
                variants.append(tuple(variant))
        return _dedupe(variants)
    if kind == "dict":
        keys = _pool(spec.items[0], name + "{}", hints)[:3]
        values = _pool(spec.items[1], name + "{}", hints)[:3]
        return _dedupe([{}, {keys[0]: values[0]}, {keys[0]: values[-1]}, {keys[-1]: values[0], keys[0]: values[-1]}])
    return []


def _hashable(values: list[Any]) -> bool:
    try:
        set(values)
    except TypeError:
        return False
    return True


def conforms(value: Any, spec: TypeSpec) -> bool:
    """Loose structural check used to filter seeds from tests and AI suggestions."""
    kind = spec.kind
    if kind == "any":
        return True
    if kind == "optional":
        return value is None or conforms(value, spec.items[0])
    if kind == "union":
        return any(conforms(value, item) for item in spec.items)
    if kind == "literal":
        return any(_same_value(value, option) for option in spec.values)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    simple = {"str": str, "bool": bool, "bytes": bytes, "list": list, "tuple": tuple, "set": set, "dict": dict}
    if kind == "none":
        return value is None
    return isinstance(value, simple.get(kind, object))


def strategy_for(spec: TypeSpec, pool: list[Any], hints: Hints, name: str) -> st.SearchStrategy[Any]:
    """Hypothesis strategy that mixes the adversarial pool with general exploration.

    ``sampled_from(pool)`` comes first so Hypothesis shrinks towards the simplest pool
    values (``0``, ``""``, ``[]``...), which keeps minimized counterexamples readable.
    """
    general = _general_strategy(spec, hints, name)
    if pool:
        return st.one_of(st.sampled_from(pool), general)
    return general


def _general_strategy(spec: TypeSpec, hints: Hints, name: str) -> st.SearchStrategy[Any]:
    kind = spec.kind
    max_length = max(hints.lengths.get(name, {0}) | {16}) + 2
    if kind == "int":
        return st.integers()
    if kind == "float":
        return st.floats(allow_nan=True, allow_infinity=True)
    if kind == "str":
        return st.text(max_size=max_length)
    if kind == "bool":
        return st.booleans()
    if kind == "bytes":
        return st.binary(max_size=max_length)
    if kind == "none":
        return st.none()
    if kind == "any":
        return st.one_of(st.none(), st.integers(), st.text(max_size=8), st.booleans())
    if kind == "literal":
        return st.sampled_from(list(spec.values))
    if kind == "optional":
        return st.one_of(st.none(), _general_strategy(spec.items[0], hints, name))
    if kind == "union":
        return st.one_of(*(_general_strategy(item, hints, name) for item in spec.items))
    if kind == "list":
        return st.lists(_general_strategy(spec.items[0], hints, name + "[]"), max_size=max_length)
    if kind == "set":
        return st.sets(_general_strategy(spec.items[0], hints, name + "[]"), max_size=max_length)
    if kind == "tuple" and spec.variadic:
        return st.lists(_general_strategy(spec.items[0], hints, name + "[]"), max_size=max_length).map(tuple)
    if kind == "tuple":
        return st.tuples(*(_general_strategy(item, hints, name) for item in spec.items))
    if kind == "dict":
        return st.dictionaries(
            _general_strategy(spec.items[0], hints, name + "{}"), _general_strategy(spec.items[1], hints, name + "{}"), max_size=4
        )
    return st.nothing()


@dataclass
class ArgSlot:
    """One parameter CounterPatch will vary."""

    name: str
    spec: TypeSpec
    pool: list[Any]
    strategy: st.SearchStrategy[Any] = field(repr=False)
    positional: bool  # must be passed positionally (positional-only parameter)
    leading_positional: bool  # can be written positionally in a readable reproduction


@dataclass
class CallPlan:
    """How to call a changed function with generated arguments on both revisions."""

    changed: ChangedFunction
    slots: list[ArgSlot]
    hints: Hints
    fixed_defaults: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return self.changed.qualified_name

    @property
    def method_kind(self) -> str:
        assert self.changed.patch is not None
        return self.changed.patch.method_kind

    def args_strategy(self) -> st.SearchStrategy[dict[str, Any]]:
        return st.fixed_dictionaries({slot.name: slot.strategy for slot in self.slots})

    def sweep(self, limit: int) -> Iterator[dict[str, Any]]:
        """Deterministic cartesian sweep over the pools, simplest combinations first.

        Combinations are enumerated by increasing sum of pool indices, so every
        single-parameter boundary value is paired with simple values for the others
        before exotic combinations are tried.
        """
        if not self.slots:
            yield {}
            return
        sizes = [len(slot.pool) for slot in self.slots]
        produced = 0
        for total in range(sum(size - 1 for size in sizes) + 1):
            for indices in _compositions(total, sizes):
                yield {slot.name: slot.pool[index] for slot, index in zip(self.slots, indices)}
                produced += 1
                if produced >= limit:
                    return


def _compositions(total: int, sizes: list[int]) -> Iterator[tuple[int, ...]]:
    if len(sizes) == 1:
        if total < sizes[0]:
            yield (total,)
        return
    for first in range(min(total, sizes[0] - 1) + 1):
        for rest in _compositions(total - first, sizes[1:]):
            yield (first, *rest)


def _callable_params(info: FunctionInfo) -> list[Param]:
    params = list(info.params)
    if info.method_kind in {"instance", "class"} and params and params[0].kind in {"positional_only", "positional_or_keyword"}:
        params = params[1:]
    return params


def build_call_plan(changed: ChangedFunction, seeds: list[tuple[list[Any], dict[str, Any]]] | None = None) -> CallPlan | str:
    """Return a :class:`CallPlan`, or a human-readable reason the function is skipped."""
    patch, base = changed.patch, changed.base
    if changed.status == "added" or base is None:
        return "new function: there is no base behavior to compare against"
    if changed.status == "deleted" or patch is None:
        return "function was deleted by the patch"
    if patch.is_property or base.is_property:
        return "property accessors are not exercised automatically in v0.1"
    if patch.name.startswith("__") and patch.name.endswith("__"):
        return "dunder methods are not exercised automatically in v0.1"
    if patch.method_kind != base.method_kind:
        return f"method kind changed ({base.method_kind} -> {patch.method_kind})"
    if "<locals>" in patch.qualified_name:
        return "nested functions are not exercised automatically"

    patch_params = _callable_params(patch)
    base_params = {param.name: param for param in _callable_params(base)}
    param_names = {param.name for param in patch_params}
    hints = discover_hints([patch.node], param_names, patch.constants)
    base_hints = discover_hints([base.node], param_names, base.constants)
    for name, values in base_hints.numbers.items():
        hints.numbers.setdefault(name, set()).update(values)
    for name, sizes in base_hints.lengths.items():
        hints.lengths.setdefault(name, set()).update(sizes)
    for name, strings in base_hints.strings.items():
        for value in strings:
            hints.add_string(name, value)
    hints.constants |= base_hints.constants
    _add_seed_hints(hints, patch_params, seeds or [])

    slots: list[ArgSlot] = []
    fixed_defaults: list[str] = []
    leading = True
    for param in patch_params:
        if param.kind in {"var_positional", "var_keyword"}:
            continue
        base_param = base_params.get(param.name)
        if base_param is None and param.has_default:
            fixed_defaults.append(param.name)  # new optional parameter: keep its default
            leading = False
            continue
        spec = parse_annotation(param.annotation_node)
        if spec is None and param.annotation is None:
            spec = infer_from_default(param.default_node) or infer_from_usage(patch.node, param.name)
        if spec is None:
            if param.has_default:
                fixed_defaults.append(param.name)
                leading = False
                continue
            if param.annotation:
                return f"parameter `{param.name}: {param.annotation}` requires a value CounterPatch cannot construct automatically"
            return f"parameter `{param.name}` has no type annotation, default value, or inferable usage"
        if base_param is None or base_param.kind in {"var_positional", "var_keyword"}:
            return f"signature changed: parameter `{param.name}` does not exist in the base revision"
        if (param.kind == "positional_only") != (base_param.kind == "positional_only"):
            return f"signature changed: `{param.name}` is positional-only in only one revision"
        pool = value_pool(spec, param.name, hints)
        if not pool:
            return f"parameter `{param.name}` has an unsupported type"
        slots.append(
            ArgSlot(
                name=param.name,
                spec=spec,
                pool=pool,
                strategy=strategy_for(spec, pool, hints, param.name),
                positional=param.kind == "positional_only",
                leading_positional=leading and param.kind in {"positional_only", "positional_or_keyword"},
            )
        )
        if param.kind == "keyword_only":
            leading = False

    covered = {slot.name for slot in slots} | set(fixed_defaults)
    missing = [
        name
        for name, param in base_params.items()
        if not param.has_default and param.kind not in {"var_positional", "var_keyword"} and name not in covered
    ]
    if missing:
        return f"signature changed: base revision requires {', '.join(missing)}"
    positions = [param.name for param in patch_params if param.kind == "positional_only"]
    base_positions = [param.name for param in _callable_params(base) if param.kind == "positional_only"]
    if positions != base_positions[: len(positions)]:
        return "signature changed: positional-only parameters differ"
    return CallPlan(changed=changed, slots=slots, hints=hints, fixed_defaults=fixed_defaults)


def _add_seed_hints(hints: Hints, params: list[Param], seeds: list[tuple[list[Any], dict[str, Any]]]) -> None:
    positional = [param.name for param in params if param.kind in {"positional_only", "positional_or_keyword"}]
    for args, kwargs in seeds[:50]:
        for name, value in zip(positional, args):
            hints.add_seed(name, value)
        for name, value in kwargs.items():
            hints.add_seed(name, value)


def seed_examples(plan: CallPlan, seeds: list[tuple[list[Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Full argument dictionaries taken verbatim from existing tests (when complete)."""
    assert plan.changed.patch is not None
    positional = [
        param.name for param in _callable_params(plan.changed.patch) if param.kind in {"positional_only", "positional_or_keyword"}
    ]
    examples: list[dict[str, Any]] = []
    slot_by_name = {slot.name: slot for slot in plan.slots}
    for args, kwargs in seeds[:20]:
        candidate = dict(zip(positional, args))
        candidate.update({key: value for key, value in kwargs.items() if key in slot_by_name})
        if set(candidate) == set(slot_by_name) and all(conforms(candidate[name], slot_by_name[name].spec) for name in candidate):
            examples.append(candidate)
    return examples
