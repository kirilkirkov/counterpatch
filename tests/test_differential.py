from __future__ import annotations

from pathlib import Path

import pytest

from counterpatch.analyzer import collect_functions, raised_types
from counterpatch.differential import FunctionContext, classify, classify_test_outcomes
from counterpatch.intent import DiffNumbers, Intent, load_task
from counterpatch.models import ConfigError, HarnessError, Observation, Severity


def returned(value: str) -> Observation:
    return Observation(status="returned", return_value_repr=value, return_value_key=value)


def raised(exception: str, message: str = "") -> Observation:
    return Observation(status="raised", exception_type=exception, exception_message=message)


def context(
    task: str | None = None,
    base_raises: set[str] | None = None,
    diff: str = "",
    names: set[str] | None = None,
) -> FunctionContext:
    return FunctionContext(
        base_raises={"ValueError"} if base_raises is None else base_raises,
        patch_raises={"ValueError"},
        intent=Intent.from_text(task),
        diff_numbers=DiffNumbers.from_diff(diff),
        names=names or {"x"},
    )


@pytest.mark.parametrize(
    ("base", "patch", "severity", "kind"),
    [
        (returned("1"), returned("1"), Severity.NONE, "same"),
        (raised("ValueError", "a"), raised("ValueError", "b"), Severity.NONE, "same"),
        (returned("1"), returned("2"), Severity.DIVERGENCE, "return-value-changed"),
        (raised("ValueError"), returned("1"), Severity.REGRESSION, "accepts-rejected-input"),
        (raised("TypeError"), returned("1"), Severity.IMPROVEMENT, "base-crash-fixed"),
        (returned("1"), raised("ZeroDivisionError"), Severity.REGRESSION, "new-crash"),
        (returned("1"), raised("ValueError"), Severity.DIVERGENCE, "rejects-accepted-input"),
        (raised("ValueError"), raised("KeyError"), Severity.REGRESSION, "validation-became-crash"),
        (raised("ValueError"), raised("mod.CustomError"), Severity.DIVERGENCE, "exception-type-changed"),
        (returned("1"), Observation(status="timeout"), Severity.REGRESSION, "new-hang"),
        (returned("1"), Observation(status="crashed"), Severity.REGRESSION, "new-process-crash"),
        (Observation(status="timeout"), returned("1"), Severity.IMPROVEMENT, "base-abnormal"),
        (Observation(status="timeout"), Observation(status="timeout"), Severity.NONE, "same"),
    ],
)
def test_classification_table(base: Observation, patch: Observation, severity: Severity, kind: str) -> None:
    verdict = classify(base, patch, context(), {"x": 5})
    assert (verdict.severity, verdict.kind) == (severity, kind)


def test_incidental_exception_caught_by_patch_is_only_a_divergence() -> None:
    """``int(s)`` raising ValueError is not validation; catching it is not a regression."""
    verdict = classify(raised("ValueError"), returned("0"), context(base_raises=set()), {"x": ""})
    assert (verdict.severity, verdict.kind) == (Severity.DIVERGENCE, "accepts-failing-input")
    crash = classify(raised("ValueError"), raised("KeyError"), context(base_raises=set()), {"x": 1})
    assert crash.severity is Severity.DIVERGENCE


def test_validation_in_same_module_helper_counts_as_deliberate() -> None:
    source = (
        "def _check(amount):\n    if amount <= 0:\n        raise ValueError('positive')\n\n"
        "def withdraw(balance: int, amount: int) -> int:\n    _check(amount)\n    return balance - amount\n\n"
        "class Account:\n    def _guard(self, x):\n        raise PermissionError\n\n"
        "    def pay(self, x: int) -> int:\n        self._guard(x)\n        return x\n"
    )
    functions = collect_functions(source)
    assert functions["withdraw"].raises == {"ValueError"}
    assert functions["Account.pay"].raises == {"PermissionError"}


def test_float_noise_and_truncation_use_comparison_keys() -> None:
    base = Observation(status="returned", return_value_repr="28.333333333333332", return_value_key="k1")
    patch = Observation(status="returned", return_value_repr="28.333333333333336", return_value_key="k1")
    assert classify(base, patch, context("Refactor"), {"x": 1}).severity is Severity.NONE
    long_base = Observation(status="returned", return_value_repr="x" * 10, return_value_key="a")
    long_patch = Observation(status="returned", return_value_repr="x" * 10, return_value_key="b")
    assert classify(long_base, long_patch, context(), {"x": 1}).kind == "return-value-changed"


def test_mutation_and_output_differences() -> None:
    base = Observation(status="returned", return_value_key="None", args_after_key="a")
    patch = Observation(status="returned", return_value_key="None", args_after_key="b")
    assert classify(base, patch, context(), {}).kind == "argument-mutation-changed"
    base, patch = Observation(status="returned", stdout="a"), Observation(status="returned", stdout="b")
    assert classify(base, patch, context(), {}).kind == "output-changed"


def test_harness_errors_are_not_findings_and_keep_their_phase() -> None:
    failure = Observation(status="error", phase="import", detail="ImportError")
    with pytest.raises(HarnessError, match="patched revision") as info:
        classify(returned("1"), failure, context(), {})
    assert info.value.failures == {"patch": failure}


LIMIT_DIFF = "@@ -2 +2 @@\n-    if len(username) > 20:\n+    if len(username) > 32:\n"


def test_task_related_divergence_is_downgraded() -> None:
    ctx = context("Allow usernames up to 32 characters", diff=LIMIT_DIFF, names={"username", "validate"})
    at_new_limit = classify(raised("ValueError"), returned("True"), ctx, {"username": "a" * 32})
    assert at_new_limit.severity is Severity.DIVERGENCE
    assert at_new_limit.intent_note and "32" in at_new_limit.intent_note
    between_limits = classify(raised("ValueError"), returned("True"), ctx, {"username": "a" * 25})
    assert between_limits.severity is Severity.DIVERGENCE
    unrelated_empty = classify(raised("ValueError"), returned("True"), ctx, {"username": ""})
    assert unrelated_empty.severity is Severity.REGRESSION


def test_crashes_and_hangs_are_never_downgraded_by_the_task() -> None:
    ctx = context("Allow usernames up to 32 characters", diff=LIMIT_DIFF, names={"username"})
    crash = classify(returned("True"), raised("IndexError"), ctx, {"username": "a" * 32})
    assert (crash.severity, crash.intent_note) == (Severity.REGRESSION, None)
    hang = classify(returned("True"), Observation(status="timeout"), ctx, {"username": "a" * 32})
    assert hang.severity is Severity.REGRESSION


def test_task_numbers_only_count_when_the_diff_touches_them() -> None:
    ctx = context("Refactor withdraw, fixes ticket 1", names={"balance", "amount", "withdraw"})
    verdict = classify(raised("ValueError"), returned("0"), ctx, {"balance": 0, "amount": 0})
    assert verdict.severity is Severity.REGRESSION


def test_keywords_only_count_when_the_task_names_the_parameter_or_function() -> None:
    names = {"amount", "withdraw"}
    about_amount = context("Allow negative amounts", names=names)
    assert classify(raised("ValueError"), returned("1"), about_amount, {"amount": -1}).severity is Severity.DIVERGENCE
    unrelated = context("Clean up negative test fixtures", names=names)
    assert classify(raised("ValueError"), returned("1"), unrelated, {"amount": -1}).severity is Severity.REGRESSION


def test_behavior_preserving_task_upgrades_divergence() -> None:
    verdict = classify(returned("1"), returned("2"), context("Refactor withdrawal validation"), {"x": 3})
    assert verdict.severity is Severity.REGRESSION
    assert "behavior-preserving" in (verdict.intent_note or "")
    assert classify(returned("1"), returned("2"), context(), {"x": 3}).severity is Severity.DIVERGENCE


@pytest.mark.parametrize(
    ("base", "patch", "flag", "severity", "kind"),
    [
        ("passed", "passed", False, Severity.NONE, "pass-pass"),
        ("passed", "failed", False, Severity.REGRESSION, "generated-test-regression"),
        ("passed", "failed", True, Severity.DIVERGENCE, "generated-test-diverges"),
        ("failed", "failed", False, Severity.NONE, "fail-fail"),
        ("failed", "passed", False, Severity.IMPROVEMENT, "fail-pass"),
        ("error", "error", False, Severity.NONE, "invalid"),
        ("passed", "error", False, Severity.DIVERGENCE, "generated-test-errors-on-patch"),
    ],
)
def test_generated_test_outcomes(base: str, patch: str, flag: bool, severity: Severity, kind: str) -> None:
    verdict = classify_test_outcomes(base, patch, flag)
    assert (verdict.severity, verdict.kind) == (severity, kind)


def test_raised_types_and_diff_numbers() -> None:
    import ast

    tree = ast.parse("def f(x):\n    if x:\n        raise ValueError('a')\n    raise errors.Custom\n")
    assert raised_types(tree) == {"ValueError", "Custom"}
    numbers = DiffNumbers.from_diff("@@ -1 +1 @@\n-    if len(name) > 20:  # max 99\n+    if len(name) > 32:\n")
    assert (numbers.removed, numbers.added) == ({20.0}, {32.0})


def test_load_task_from_cli_and_file(tmp_path: Path) -> None:
    assert load_task(None, None) is None
    assert load_task("  Allow longer names ", None) == "Allow longer names"
    issue = tmp_path / "issue.md"
    issue.write_text("# Issue\nAllow usernames up to 32 characters.\n")
    assert load_task(None, issue) == "# Issue\nAllow usernames up to 32 characters."
    assert load_task("Title", issue).startswith("Title\n\n# Issue")
    with pytest.raises(ConfigError, match="does not exist"):
        load_task(None, tmp_path / "missing.md")
