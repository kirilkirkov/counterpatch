from __future__ import annotations

import sys
from pathlib import Path

from counterpatch.analyzer import collect_functions
from counterpatch.candidates import CallPlan, build_call_plan
from counterpatch.differential import DifferentialExecutor, FunctionContext
from counterpatch.intent import Intent
from counterpatch.models import ChangedFunction, Severity
from counterpatch.runner import Worker
from counterpatch.shrinking import explore, simpler_values, simplicity
from counterpatch.utils import Deadline


def test_simplicity_prefers_small_readable_values() -> None:
    assert simplicity(0) < simplicity(1) < simplicity(-1) < simplicity(918273)
    assert simplicity("") < simplicity("a") < simplicity("é") < simplicity("aaaa")
    assert simplicity([]) < simplicity([0]) < simplicity([0, 0])
    assert simplicity(None) < simplicity(False) < simplicity(0)


def test_simpler_values_are_strictly_simpler() -> None:
    for value in [918273, -9281, "hello world", [3, 1, 2], {"a": 1}, 2.5]:
        assert all(simplicity(candidate) < simplicity(value) for candidate in simpler_values(value))
    assert 0 in simpler_values(-9281)


def _executor(tmp_path: Path, base: str, patch: str, name: str) -> tuple[DifferentialExecutor, list[Worker]]:
    base_dir, patch_dir = tmp_path / "base", tmp_path / "patch"
    base_dir.mkdir()
    patch_dir.mkdir()
    (base_dir / "mod.py").write_text(base)
    (patch_dir / "mod.py").write_text(patch)
    changed = ChangedFunction(
        file="mod.py",
        base_file="mod.py",
        qualified_name=name,
        status="modified",
        diff="",
        patch=collect_functions(patch)[name],
        base=collect_functions(base)[name],
    )
    plan = build_call_plan(changed)
    assert isinstance(plan, CallPlan)
    workers = [Worker(sys.executable, base_dir, "base"), Worker(sys.executable, patch_dir, "patch")]
    context = FunctionContext.for_plan(plan, Intent.from_text(None))
    return DifferentialExecutor(plan, workers[0], workers[1], context, 5, Deadline(120)), workers


def test_hypothesis_finds_and_minimizes_counterexample_outside_pools(tmp_path: Path) -> None:
    base = "def scale(x: int) -> int:\n    return x * 2\n"
    patch = "def scale(x: int) -> int:\n    if x > 5000 and x % 7 == 3:\n        return -x\n    return x * 2\n"
    executor, workers = _executor(tmp_path, base, patch, "scale")
    try:
        result = explore(executor, [], hypothesis_examples=300)
    finally:
        for worker in workers:
            worker.close()
    assert result.best is not None
    assert result.best.verdict.kind == "return-value-changed"
    assert result.best.args == {"x": 5001}
    assert result.source == "hypothesis"


def test_greedy_shrinking_reduces_large_counterexample(tmp_path: Path) -> None:
    base = "def withdraw(balance: int, amount: int) -> int:\n    if amount <= 0:\n        raise ValueError('positive')\n    return balance - amount\n"
    patch = "def withdraw(balance: int, amount: int) -> int:\n    return balance - amount\n"
    executor, workers = _executor(tmp_path, base, patch, "withdraw")
    try:
        result = explore(executor, [({"balance": 918273, "amount": -9281}, "deterministic")], hypothesis_examples=0, shrink_examples=0)
    finally:
        for worker in workers:
            worker.close()
    assert result.best is not None
    assert result.best.verdict.severity is Severity.REGRESSION
    assert result.best.args == {"balance": 0, "amount": 0}
    assert result.shrunk_from == {"balance": 918273, "amount": -9281}
