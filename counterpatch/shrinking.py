"""Search for behavioral differences and minimize them into readable counterexamples.

1. A deterministic sweep over the adversarial pools (simplest combinations first).
2. A Hypothesis search (``hypothesis.find``) for a regression, biased towards the
   pools, which also explores values the pools do not contain.
3. Minimization of the most severe finding: Hypothesis' shrinker first, then a small
   greedy pass that tries obviously simpler values (``0``, ``""``, ``[]``...).

Hypothesis runs derandomized and without an example database so results are
reproducible in CI.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, find, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.errors import HypothesisException

from counterpatch.candidates import conforms
from counterpatch.differential import DifferentialExecutor, args_key, is_possibly_intended
from counterpatch.models import CounterPatchError, Example, Severity

MAX_TIMEOUTS_PER_FUNCTION = 3
HYPOTHESIS_HOME = Path(tempfile.gettempdir()) / "counterpatch-hypothesis"
_SEVERITY_RANK = {Severity.REGRESSION: 0, Severity.DIVERGENCE: 1, Severity.IMPROVEMENT: 2, Severity.NONE: 3}
_TYPE_RANK = {type(None): 0, bool: 1, int: 2, float: 3, str: 4, bytes: 5, tuple: 6, list: 6, set: 7, frozenset: 7, dict: 8}


def simplicity(value: Any) -> tuple[Any, ...]:
    """Sort key: smaller means simpler for a human reader."""
    rank = _TYPE_RANK.get(type(value), 9)
    if isinstance(value, bool) or value is None:
        return (rank, int(bool(value)))
    if isinstance(value, int):
        return (rank, abs(value), value < 0)
    if isinstance(value, float):
        finite = math.isfinite(value)
        return (rank, 0 if finite else 1, abs(value) if finite else 0.0, value < 0 if finite else False)
    if isinstance(value, (str, bytes)):
        non_ascii = sum(1 for character in value if (ord(character) if isinstance(character, str) else character) > 127)
        return (rank, len(value), non_ascii, str(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=simplicity) if isinstance(value, (set, frozenset)) else list(value)
        return (rank, len(items), tuple(simplicity(item) for item in items))
    if isinstance(value, dict):
        return (rank, len(value), tuple(sorted((simplicity(key), simplicity(item)) for key, item in value.items())))
    return (rank, repr(value))


def args_simplicity(args: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(simplicity(value) for value in args.values())


def simpler_values(value: Any) -> list[Any]:
    """Candidate replacements that are simpler than ``value``."""
    candidates: list[Any] = []
    if isinstance(value, bool):
        candidates = [False]
    elif isinstance(value, int):
        candidates = [0, 1, -1, value // 2, value - 1 if value > 0 else value + 1, -value]
    elif isinstance(value, float):
        candidates = [0.0, 1.0, -1.0, value / 2 if math.isfinite(value) else 0.0]
        if math.isfinite(value):
            candidates.append(float(int(value)))
    elif isinstance(value, (str, bytes)):
        half = len(value) // 2
        empty = value[:0]
        filler = "a" if isinstance(value, str) else b"a"
        candidates = [empty, value[:half], value[half:], value[1:], value[:-1], filler * len(value)]
    elif isinstance(value, (list, tuple)):
        half = len(value) // 2
        candidates = [value[:0], value[:half], value[half:]]
        candidates += [value[:index] + value[index + 1 :] for index in range(min(len(value), 8))]
    elif isinstance(value, dict):
        candidates = [{}] + [{k: v for k, v in value.items() if k != key} for key in list(value)[:8]]
    elif isinstance(value, set):
        candidates = [set()] + [value - {item} for item in list(value)[:8]]
    if not isinstance(value, (bool, type(None))):
        candidates.append(None)
    key = simplicity(value)
    return [candidate for candidate in candidates if simplicity(candidate) < key]


@dataclass
class Exploration:
    best: Example | None = None
    source: str = "deterministic"
    shrunk_from: dict[str, Any] | None = None
    others: list[Example] = field(default_factory=list)
    improvements: int = 0
    stopped_early: bool = False


def _find(executor: DifferentialExecutor, predicate: Callable[[Example], bool], max_examples: int) -> Example | None:
    # Hypothesis caches data under ``$CWD/.hypothesis`` by default; keep the developer's repository clean.
    set_hypothesis_home_dir(HYPOTHESIS_HOME)
    failure: list[BaseException] = []

    def condition(args: dict[str, Any]) -> bool:
        if failure or not _affordable(executor, args):
            return False
        try:
            return predicate(executor.evaluate(args))
        except CounterPatchError as error:
            failure.append(error)
            return False

    try:
        found = find(
            executor.plan.args_strategy(),
            condition,
            settings=settings(
                max_examples=max_examples,
                derandomize=True,
                database=None,
                deadline=None,
                suppress_health_check=list(HealthCheck),
            ),
        )
    except HypothesisException:
        found = None
    if failure:
        raise failure[0]
    return executor.evaluate(found) if found is not None else None


def explore(
    executor: DifferentialExecutor,
    sweep: list[tuple[dict[str, Any], str]],
    hypothesis_examples: int,
    shrink_examples: int = 150,
) -> Exploration:
    """Run the sweep and Hypothesis search, then minimize the most severe finding.

    ``sweep`` holds ``(args, label)`` pairs; the label (``deterministic``, ``existing-test``,
    ``ai-input``) records where a candidate came from.
    """
    sources: dict[tuple[Any, ...], str] = {}
    for args, label in sweep:
        if executor.timeouts >= MAX_TIMEOUTS_PER_FUNCTION:
            break
        example = executor.evaluate(args)
        sources.setdefault(args_key(example.args), label)

    hangs = executor.timeouts >= MAX_TIMEOUTS_PER_FUNCTION
    if not _has(executor, Severity.REGRESSION) and hypothesis_examples > 0 and not hangs:
        found = _find(executor, lambda example: example.verdict.severity is Severity.REGRESSION, hypothesis_examples)
        if found is not None:
            sources[args_key(found.args)] = "hypothesis"

    examples = list(executor.cache.values())
    result = Exploration(
        improvements=sum(1 for example in examples if example.verdict.severity is Severity.IMPROVEMENT),
        stopped_early=hangs,
    )
    interesting = [example for example in examples if example.verdict.severity in {Severity.REGRESSION, Severity.DIVERGENCE}]
    if not interesting:
        return result

    # Most severe first; among divergences, unexplained ones before "possibly intended" ones,
    # so an intended change can never hide an unrelated behavior change.
    target = min(interesting, key=lambda example: (*_rank(example), args_simplicity(example.args)))
    result.source = sources.get(args_key(target.args), "hypothesis")

    def matches(example: Example) -> bool:
        return _rank(example) == _rank(target) and example.verdict.kind == target.verdict.kind

    minimized = target
    # Every hanging candidate costs a full call timeout, so hangs get only a short greedy pass.
    slow = hangs or target.verdict.kind == "new-hang"
    hypothesis_minimum = _find(executor, matches, shrink_examples) if shrink_examples > 0 and not slow else None
    if hypothesis_minimum is not None and args_simplicity(hypothesis_minimum.args) < args_simplicity(minimized.args):
        minimized = hypothesis_minimum
    minimized = greedy_shrink(executor, minimized, matches, max_steps=10 if slow else 200)
    if minimized is not target:
        result.shrunk_from = target.args
    result.best = minimized
    result.others = _other_examples(executor, minimized)
    return result


def greedy_shrink(executor: DifferentialExecutor, example: Example, matches: Callable[[Example], bool], max_steps: int = 200) -> Example:
    """Repeatedly replace one argument with a simpler value while the verdict holds."""
    specs = {slot.name: slot.spec for slot in executor.plan.slots}
    current, steps, improved = example, 0, True
    while improved and steps < max_steps:
        improved = False
        for name, value in current.args.items():
            for candidate in simpler_values(value):
                args = {**current.args, name: candidate}
                if not conforms(candidate, specs[name]) or not _affordable(executor, args):
                    continue
                steps += 1
                attempt = executor.evaluate(args)
                if matches(attempt):
                    current, improved = attempt, True
                    break
                if steps >= max_steps:
                    return current
            if improved:
                break
    return current


def _rank(example: Example) -> tuple[int, bool]:
    return _SEVERITY_RANK[example.verdict.severity], is_possibly_intended(example.verdict)


def _other_examples(executor: DifferentialExecutor, best: Example, limit: int = 2) -> list[Example]:
    """A few additional, differently-behaving inputs with the same classification."""
    seen = {(best.base.describe(), best.patch.describe())}
    others: list[Example] = []
    candidates = sorted(
        (example for example in executor.cache.values() if _rank(example) == _rank(best)),
        key=lambda example: args_simplicity(example.args),
    )
    for example in candidates:
        signature = (example.base.describe(), example.patch.describe())
        if signature in seen:
            continue
        seen.add(signature)
        others.append(example)
        if len(others) >= limit:
            break
    return others


def _affordable(executor: DifferentialExecutor, args: dict[str, Any]) -> bool:
    """After repeated timeouts, only already-evaluated candidates may be (re)used."""
    return executor.timeouts < MAX_TIMEOUTS_PER_FUNCTION or args_key(args) in executor.cache


def _has(executor: DifferentialExecutor, severity: Severity) -> bool:
    return any(example.verdict.severity is severity for example in executor.cache.values())
