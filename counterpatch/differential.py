"""Base-vs-patch execution and behavioral classification.

Classification rules (before task adjustments):

===========================================  ==============================  ===========
base                                         patch                           severity
===========================================  ==============================  ===========
same behavior                                same behavior                   none
returned X                                   returned Y != X                 divergence
returned                                     raised a crash-type error       regression
returned                                     raised any other error          divergence
raised a deliberate rejection                returned                        regression
raised another non-crash error               returned                        divergence
raised a crash-type error                    returned                        improvement
raised a deliberate rejection                raised a crash-type error       regression
raised A                                     raised B (other cases)          divergence
completed                                    timed out / killed interpreter  regression
timed out / killed interpreter               completed                       improvement
===========================================  ==============================  ===========

A *deliberate rejection* is an exception type the base function raises explicitly
(``raise X``), directly or through a same-module helper it calls. An exception that
merely escapes from other code (``int("x")`` raising ``ValueError``) is not treated
as validation: the patch catching it is reported as a divergence, not a regression.
*Crash-type* errors are generic failures (``TypeError``, ``IndexError``,
``ZeroDivisionError``...) that the function does not raise explicitly.

Values are compared on noise-tolerant keys: floats to 12 significant digits, sets and
dicts order-insensitively, memory addresses stripped. Exception *messages* are ignored.

Task adjustments: a divergence or rejection-related regression whose input relates to
the task description is downgraded to a divergence ("possibly intended"); crashes and
hangs are never downgraded. A behavior-preserving task (refactor, cleanup) upgrades
unexplained divergences to regressions. Improvements are counted, not reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from counterpatch.candidates import CallPlan
from counterpatch.intent import DiffNumbers, Intent
from counterpatch.models import CheckTimeout, Example, HarnessError, Observation, Severity, Verdict
from counterpatch.runner import Worker
from counterpatch.utils import Deadline, module_for_path

CRASH_TYPES = frozenset(
    {
        "TypeError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "ZeroDivisionError",
        "RecursionError",
        "UnboundLocalError",
        "NameError",
        "OverflowError",
        "AssertionError",
        "UnicodeEncodeError",
        "UnicodeDecodeError",
        "NotImplementedError",
        "StopIteration",
        "MemoryError",
        "SystemError",
    }
)
# Kinds that are never "the intended way" to implement a requested change.
NEVER_INTENDED = frozenset({"new-crash", "new-hang", "new-process-crash", "validation-became-crash"})
HANG_CONFIRMATION_FACTOR = 4


@dataclass
class FunctionContext:
    base_raises: set[str]
    patch_raises: set[str]
    intent: Intent
    diff_numbers: DiffNumbers
    names: set[str]

    @classmethod
    def for_plan(cls, plan: CallPlan, intent: Intent) -> FunctionContext:
        assert plan.changed.base is not None and plan.changed.patch is not None
        return cls(
            base_raises=plan.changed.base.raises,
            patch_raises=plan.changed.patch.raises,
            intent=intent,
            diff_numbers=DiffNumbers.from_diff(plan.changed.diff),
            names={slot.name for slot in plan.slots} | {plan.changed.patch.name},
        )


def _is_crash(observation: Observation, explicit: set[str]) -> bool:
    name = observation.exception_short_name or ""
    return name in CRASH_TYPES and name not in explicit


def _is_deliberate(observation: Observation, explicit: set[str]) -> bool:
    return (observation.exception_short_name or "") in explicit


def same_behavior(base: Observation, patch: Observation) -> bool:
    return (
        base.status == patch.status
        and base.return_value_key == patch.return_value_key
        and base.exception_type == patch.exception_type
        and base.args_after_key == patch.args_after_key
        and base.state_after_key == patch.state_after_key
        and base.stdout == patch.stdout
    )


def classify(base: Observation, patch: Observation, context: FunctionContext, args: dict[str, Any]) -> Verdict:
    """Classify one base/patch observation pair; see the module docstring for the rules."""
    failures = {side: observation for side, observation in (("base", base), ("patch", patch)) if observation.status == "error"}
    if failures:
        details = "; ".join(f"{'patched' if side == 'patch' else 'base'} revision: {obs.detail}" for side, obs in failures.items())
        raise HarnessError(f"could not execute ({details})", failures)
    verdict = _raw_verdict(base, patch, context)
    if verdict.severity in {Severity.NONE, Severity.IMPROVEMENT}:
        return verdict
    related = None if verdict.kind in NEVER_INTENDED else context.intent.relatedness(args, context.diff_numbers, context.names)
    if related:
        return Verdict(Severity.DIVERGENCE, verdict.kind, verdict.summary, f"Possibly intended: {related}.")
    if context.intent.preserves_behavior and verdict.severity is Severity.DIVERGENCE:
        return Verdict(
            Severity.REGRESSION,
            verdict.kind,
            verdict.summary,
            "The task describes a behavior-preserving change, so any behavior change is unexpected.",
        )
    return verdict


def is_possibly_intended(verdict: Verdict) -> bool:
    return bool(verdict.intent_note and verdict.intent_note.startswith("Possibly intended"))


def _raw_verdict(base: Observation, patch: Observation, context: FunctionContext) -> Verdict:
    abnormal = {"timeout", "crashed"}
    if same_behavior(base, patch):
        return Verdict(Severity.NONE, "same", "Both revisions behave the same")
    if base.status in abnormal and patch.status in abnormal:
        return Verdict(Severity.NONE, "both-abnormal", "Neither revision completes")
    if base.status in abnormal:
        return Verdict(Severity.IMPROVEMENT, "base-abnormal", "Base did not complete; patch does")
    if patch.status == "timeout":
        return Verdict(Severity.REGRESSION, "new-hang", "The patched revision did not finish; the base revision did")
    if patch.status == "crashed":
        return Verdict(Severity.REGRESSION, "new-process-crash", "The patched revision kills the interpreter")

    if base.status == "raised" and patch.status == "raised":
        if base.exception_short_name == patch.exception_short_name:
            return Verdict(Severity.NONE, "same-exception", "Same exception type")
        if _is_deliberate(base, context.base_raises) and _is_crash(patch, context.patch_raises):
            return Verdict(Severity.REGRESSION, "validation-became-crash", "A deliberate rejection became an unhandled crash")
        return Verdict(Severity.DIVERGENCE, "exception-type-changed", "The exception type changed")
    if base.status == "raised" and patch.status == "returned":
        if _is_deliberate(base, context.base_raises):
            return Verdict(Severity.REGRESSION, "accepts-rejected-input", "Input the base revision deliberately rejects is now accepted")
        if _is_crash(base, context.base_raises):
            return Verdict(Severity.IMPROVEMENT, "base-crash-fixed", "Base crashed; patch returns a value")
        return Verdict(Severity.DIVERGENCE, "accepts-failing-input", "Input that raised an error in the base revision now returns a value")
    if base.status == "returned" and patch.status == "raised":
        if _is_crash(patch, context.patch_raises):
            return Verdict(Severity.REGRESSION, "new-crash", "Previously accepted input now crashes")
        return Verdict(Severity.DIVERGENCE, "rejects-accepted-input", "Previously accepted input is now rejected")
    if base.return_value_key != patch.return_value_key:
        return Verdict(Severity.DIVERGENCE, "return-value-changed", "The return value changed")
    if base.args_after_key != patch.args_after_key:
        return Verdict(Severity.DIVERGENCE, "argument-mutation-changed", "How the arguments are mutated changed")
    if base.state_after_key != patch.state_after_key:
        return Verdict(Severity.DIVERGENCE, "object-state-changed", "The object's state after the call changed")
    return Verdict(Severity.DIVERGENCE, "output-changed", "Printed output changed")


def classify_test_outcomes(base_outcome: str, patch_outcome: str, targets_intended_change: bool) -> Verdict:
    """Classify a generated pytest test run against both revisions.

    A test failing on the patch is only a regression signal when the same test passes
    on the base revision; a failing assertion alone proves nothing.
    """
    if base_outcome == "passed" and patch_outcome == "passed":
        return Verdict(Severity.NONE, "pass-pass", "No counterexample")
    if base_outcome == "passed" and patch_outcome == "failed":
        if targets_intended_change:
            return Verdict(
                Severity.DIVERGENCE,
                "generated-test-diverges",
                "Generated test passes on base and fails on the patch",
                "Possibly intended: the generator marked this test as targeting the requested change.",
            )
        return Verdict(Severity.REGRESSION, "generated-test-regression", "Generated test passes on base and fails on the patch")
    if base_outcome == "passed" and patch_outcome == "error":
        return Verdict(Severity.DIVERGENCE, "generated-test-errors-on-patch", "Generated test passes on base and errors on the patch")
    if base_outcome in {"failed", "error"} and patch_outcome == "passed":
        return Verdict(Severity.IMPROVEMENT, "fail-pass", "Possible improvement or intentional behavior change")
    if base_outcome == "failed" and patch_outcome == "failed":
        return Verdict(Severity.NONE, "fail-fail", "Non-discriminating: the test fails on both revisions")
    return Verdict(Severity.NONE, "invalid", "Invalid candidate: the test could not run cleanly on both revisions")


def args_key(args: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple((name, type(value).__name__, repr(value)) for name, value in args.items())


def _budget_exhausted(deadline: Deadline) -> CheckTimeout:
    return CheckTimeout(f"CounterPatch exceeded its --timeout budget of {deadline.seconds:g}s")


class DifferentialExecutor:
    """Runs one changed function's candidates on both revisions, with caching."""

    def __init__(
        self,
        plan: CallPlan,
        base_worker: Worker,
        patch_worker: Worker,
        context: FunctionContext,
        call_timeout: float,
        deadline: Deadline,
    ) -> None:
        self.plan = plan
        self.workers = {"base": base_worker, "patch": patch_worker}
        self.context = context
        self.call_timeout = call_timeout
        self.deadline = deadline
        self.cache: dict[tuple[tuple[str, str, str], ...], Example] = {}
        self.executions = 0
        self.timeouts = 0

    def request(self, args: dict[str, Any], side: str, workspace: Path) -> dict[str, Any]:
        file = self.plan.changed.file if side == "patch" else (self.plan.changed.base_file or self.plan.changed.file)
        root, module = module_for_path(workspace, file)
        positional = {slot.name for slot in self.plan.slots if slot.positional}
        return {
            "module": module,
            "file": str((workspace / file).resolve()),
            "qualname": self.plan.qualified_name,
            "kind": self.plan.method_kind,
            "roots": [str((workspace / root).resolve()), str(workspace.resolve())],
            "args": [[name, repr(value), name in positional] for name, value in args.items()],
        }

    def observe(self, args: dict[str, Any], side: str, worker: Worker | None = None, timeout: float | None = None) -> Observation:
        """Run one call on one revision.

        The per-call limit is shortened to what is left of the ``--timeout`` budget; a
        timeout caused by the budget (not the per-call limit) aborts the check instead of
        being recorded as a hang.
        """
        wanted = timeout or self.call_timeout
        available = self.deadline.remaining()
        if available <= 0.05:
            raise _budget_exhausted(self.deadline)
        effective = min(wanted, available)
        worker = worker or self.workers[side]
        self.executions += 1
        observation = worker.call(self.request(args, side, worker.workspace), effective, import_timeout=min(120.0, available))
        if observation.status == "timeout" and effective < wanted:
            raise _budget_exhausted(self.deadline)
        if observation.status == "error" and self.deadline.remaining() <= 0.05:
            raise _budget_exhausted(self.deadline)
        return observation

    def evaluate(self, args: dict[str, Any]) -> Example:
        key = args_key(args)
        if key not in self.cache:
            base = self.observe(args, "base")
            patch = self.observe(args, "patch")
            self.timeouts += int("timeout" in {base.status, patch.status})
            self.cache[key] = Example(dict(args), base, patch, classify(base, patch, self.context, args))
        return self.cache[key]

    @property
    def candidates(self) -> int:
        return len(self.cache)


def confirm(executor: DifferentialExecutor, example: Example, python: str, base_workspace: Path, patch_workspace: Path) -> str | None:
    """Re-run ``example`` twice per revision in fresh workers.

    Returns ``None`` when the verdict reproduces deterministically, otherwise a reason.
    This guards against findings caused by state leaking between candidates in the
    long-lived workers, against non-deterministic functions, and (for hangs) against
    code that is merely slow: the patched side gets a longer timeout, and a call that
    finishes within it is not reported as a hang.
    """
    hang = example.verdict.kind == "new-hang"
    patch_timeout = executor.call_timeout * HANG_CONFIRMATION_FACTOR if hang else None
    observations: dict[str, list[Observation]] = {"base": [], "patch": []}
    for side, workspace in (("base", base_workspace), ("patch", patch_workspace)):
        for _ in range(2):
            with Worker(python, workspace, f"{side}-confirm") as worker:
                timeout = patch_timeout if side == "patch" else None
                observations[side].append(executor.observe(example.args, side, worker, timeout))
    base_runs, patch_runs = observations["base"], observations["patch"]
    if not same_behavior(*base_runs) or not same_behavior(*patch_runs):
        return "behavior is not deterministic across fresh processes"
    if hang and patch_runs[0].status != "timeout":
        return f"the patched revision is slower but finishes within {patch_timeout:g}s, so it is not reported as a hang"
    verdict = classify(base_runs[0], patch_runs[0], executor.context, example.args)
    if (verdict.severity, verdict.kind) != (example.verdict.severity, example.verdict.kind):
        return "the difference did not reproduce in a fresh process (it may depend on state from earlier calls)"
    if hang:
        example.patch = patch_runs[0]
    return None
