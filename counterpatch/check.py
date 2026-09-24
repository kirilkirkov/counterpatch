"""Orchestrates a full ``counterpatch check`` run.

git diff -> changed functions -> candidates -> base & patch execution -> compare
-> shrink -> confirm -> reproduction -> verify reproduction.
"""

from __future__ import annotations

import math
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from counterpatch import ai
from counterpatch.ai import DEFAULT_MODEL
from counterpatch.analyzer import analyze_changes, find_related_tests, literal_calls_in_tests
from counterpatch.candidates import CallPlan, build_call_plan, seed_examples
from counterpatch.differential import DifferentialExecutor, FunctionContext, classify_test_outcomes, confirm
from counterpatch.git import (
    base_worktree,
    default_base,
    diff_python_files,
    has_uncommitted_changes,
    repo_root,
    resolve_base,
    short_sha,
)
from counterpatch.intent import Intent
from counterpatch.models import (
    ChangedFunction,
    CheckResult,
    CheckTimeout,
    ConfigError,
    FileDiff,
    Finding,
    FunctionReport,
    HarnessError,
    Observation,
    Severity,
)
from counterpatch.reproduction import (
    GENERATED_DIR,
    REPRODUCTIONS_DIR,
    call_expression,
    ensure_output_dir,
    with_import_shim,
    write_reproduction,
)
from counterpatch.runner import Worker, run_pytest
from counterpatch.shrinking import explore
from counterpatch.utils import Deadline, module_for_path, safe_join, sanitize_filename

Logger = Callable[[str], None]
_PYTEST_TIMEOUT = 120.0


@dataclass
class CheckOptions:
    repo: Path = Path(".")
    base: str | None = None
    task: str | None = None
    ai: bool = False
    ai_model: str = DEFAULT_MODEL
    max_tests: int = 100
    timeout: float = 600.0
    call_timeout: float = 5.0
    python: str = sys.executable
    fail_on_divergence: bool = False


def _quiet(_: str) -> None:
    return None


class _Run:
    """State for one check run."""

    def __init__(self, options: CheckOptions, log: Logger, ai_client: Any | None) -> None:
        self.options = options
        self.log = log
        self.ai_client = ai_client
        self.deadline = Deadline(options.timeout)
        self.repo = repo_root(options.repo.resolve())
        self.intent = Intent.from_text(options.task)
        self.test_cache: dict[str, list[tuple[str, str]]] = {}

    def related_tests(self, changed: ChangedFunction) -> list[tuple[str, str]]:
        _, module = module_for_path(self.repo, changed.file)
        if module not in self.test_cache:
            files = find_related_tests(self.repo, module, changed.file)
            self.test_cache[module] = [
                (path.relative_to(self.repo).as_posix(), path.read_text(encoding="utf-8", errors="replace")) for path in files
            ]
        return self.test_cache[module]

    def execute(self) -> CheckResult:
        options, repo = self.options, self.repo
        if options.max_tests < 1:
            raise ConfigError("--max-tests must be at least 1")
        if options.ai and self.ai_client is None:
            ai.require_api_key()
        base_ref = options.base or default_base(repo)
        base_sha = resolve_base(repo, base_ref)
        result = CheckResult(
            base_ref=base_ref,
            base_sha=short_sha(repo, base_sha),
            patch_label="working tree (HEAD + uncommitted changes)" if has_uncommitted_changes(repo) else "HEAD",
            task=options.task,
            fail_on_divergence=options.fail_on_divergence,
        )
        result.ai.enabled = options.ai
        result.ai.model = options.ai_model if options.ai else None

        analysis = analyze_changes(repo, base_sha, diff_python_files(repo, base_sha))
        result.changed_files = [file_diff.path for file_diff in analysis.python_files]
        result.changed_test_files = analysis.test_files
        result.parse_errors = analysis.parse_errors
        result.errors.extend(f"could not analyze {error}" for error in analysis.parse_errors)
        self.log(f"Analyzing {len(analysis.changed_functions)} changed function(s) against {base_ref} ({result.base_sha})")

        plans: list[tuple[CallPlan, FunctionReport, list[tuple[list[Any], dict[str, Any]]]]] = []
        for changed in analysis.changed_functions:
            report = FunctionReport(file=changed.file, function=changed.qualified_name, signature=changed.signature, status="skipped")
            result.functions.append(report)
            seeds: list[tuple[list[Any], dict[str, Any]]] = []
            if changed.status == "modified":
                for path, text in self.related_tests(changed):
                    calls = literal_calls_in_tests(text, changed.qualified_name.rsplit(".", 1)[-1])
                    seeds.extend(calls)
                    if path not in result.related_tests:
                        result.related_tests.append(path)
            plan = build_call_plan(changed, seeds)
            if isinstance(plan, str):
                report.skip_reason = plan
                self.log(f"Skipping {changed.qualified_name}: {plan}")
                continue
            report.status = "tested"
            plans.append((plan, report, seeds))

        modified = [changed for changed in analysis.changed_functions if changed.status == "modified"]
        if not plans and not analysis.import_checks and not (options.ai and modified):
            result.duration_seconds = self.deadline.seconds - self.deadline.remaining()
            return result

        suggestions = self.ai_suggestions(result, modified, plans) if options.ai else ai.AISuggestions()
        started = time.monotonic()
        with base_worktree(repo, base_sha) as base_workspace:
            self.log(f"Base revision checked out at {base_workspace}")
            try:
                with Worker(options.python, base_workspace, "base") as base_worker, Worker(options.python, repo, "patch") as patch_worker:
                    for file_diff in analysis.import_checks:
                        self.check_import(file_diff, base_worker, patch_worker, result)
                    for plan, report, seeds in plans:
                        self.deadline.check()
                        ai_inputs = [item.args for item in suggestions.inputs if item.function == plan.qualified_name]
                        self.test_function(plan, report, seeds, ai_inputs, base_worker, patch_worker, base_workspace, result)
                if suggestions.tests:
                    self.run_ai_tests(suggestions, analysis.changed_functions, base_workspace, result)
            except CheckTimeout as error:
                result.timed_out = True
                result.errors.append(str(error))
        self.log(f"Execution finished in {time.monotonic() - started:.1f}s")
        result.duration_seconds = self.deadline.seconds - self.deadline.remaining()
        return result

    def ai_suggestions(
        self, result: CheckResult, modified: list[ChangedFunction], plans: list[tuple[CallPlan, FunctionReport, Any]]
    ) -> ai.AISuggestions:
        plan_by_name = {plan.qualified_name: plan for plan, _, _ in plans}
        contexts: list[ai.FunctionPromptContext] = []
        modules: set[str] = set()
        test_context: list[tuple[str, str]] = []
        for changed in modified[:10]:
            _, module = module_for_path(self.repo, changed.file)
            modules.add(module)
            plan = plan_by_name.get(changed.qualified_name)
            parameters = None
            if plan is not None:
                parameters = ", ".join(f"{slot.name}: {slot.spec.describe()}" for slot in plan.slots)
            imported = changed.qualified_name.split(".")[0]
            call_hint = call_expression(plan, {slot.name: slot.pool[0] for slot in plan.slots}) if plan else changed.qualified_name
            contexts.append(ai.FunctionPromptContext(changed, f"from {module} import {imported}", call_hint, parameters))
            for item in self.related_tests(changed):
                if item not in test_context:
                    test_context.append(item)
        prompt = ai.build_prompt(contexts, self.options.task, test_context)
        self.log(f"Requesting adversarial candidates from {self.options.ai_model}")
        try:
            data = ai.request_suggestions(prompt, self.options.ai_model, self.ai_client)
        except ai.AIError as error:
            result.ai.error = str(error)
            result.errors.append(str(error))
            return ai.AISuggestions()
        suggestions = ai.parse_suggestions(data, plan_by_name, modules)
        result.ai.inputs_accepted = len(suggestions.inputs)
        result.ai.tests_accepted = len(suggestions.tests)
        result.ai.rejected = suggestions.rejected
        for reason in suggestions.rejected:
            self.log(f"AI output {reason}")
        return suggestions

    def test_function(
        self,
        plan: CallPlan,
        report: FunctionReport,
        seeds: list[tuple[list[Any], dict[str, Any]]],
        ai_inputs: list[dict[str, Any]],
        base_worker: Worker,
        patch_worker: Worker,
        base_workspace: Path,
        result: CheckResult,
    ) -> None:
        options = self.options
        context = FunctionContext.for_plan(plan, self.intent)
        executor = DifferentialExecutor(plan, base_worker, patch_worker, context, options.call_timeout, self.deadline)
        sweep_limit = max(1, (options.max_tests * 3) // 5)
        sweep: list[tuple[dict[str, Any], str]] = [(args, "existing-test") for args in seed_examples(plan, seeds)]
        sweep += [(args, "ai-input") for args in ai_inputs]
        sweep += [(args, "deterministic") for args in plan.sweep(sweep_limit)]
        hypothesis_budget = max(0, options.max_tests - sweep_limit)
        self.log(f"Testing {plan.qualified_name}: {len(sweep)} sweep candidates, Hypothesis budget {hypothesis_budget}")

        try:
            exploration = explore(executor, sweep, hypothesis_budget)
        except HarnessError as error:
            self.harness_failure(plan, report, error, result)
            return
        finally:
            report.candidates = executor.candidates
            report.executions = executor.executions
        report.improvements = exploration.improvements
        if exploration.stopped_early:
            report.notes.append(
                f"Exploration of {plan.qualified_name} stopped early after {executor.timeouts} timeouts; only "
                f"{executor.candidates} candidates were tried, so coverage is partial."
            )
        best = exploration.best
        if best is None:
            return

        try:
            problem = confirm(executor, best, options.python, base_workspace, self.repo)
        except HarnessError as error:
            problem = str(error)
        if problem:
            report.notes.append(f"Ignored an unconfirmed difference for {call_expression(plan, best.args)}: {problem}")
            self.log(report.notes[-1])
            return

        finding = Finding(
            file=plan.changed.file,
            function=plan.qualified_name,
            signature=plan.changed.signature,
            severity=best.verdict.severity,
            kind=best.verdict.kind,
            summary=best.verdict.summary,
            source=exploration.source,
            args=best.args,
            call=call_expression(plan, best.args),
            base=best.base,
            patch=best.patch,
            intent_note=best.verdict.intent_note,
            line=plan.changed.start_line,
            shrunk_from=exploration.shrunk_from,
            other_examples=exploration.others,
        )
        report.findings.append(finding)
        if finding.intent_note and finding.intent_note.startswith("Possibly intended"):
            return
        self.save_reproduction(plan, finding, base_workspace, result)

    def check_import(self, file_diff: FileDiff, base_worker: Worker, patch_worker: Worker, result: CheckResult) -> None:
        """A patch that breaks importing a module must never pass as "no regression found".

        Runs for modules whose module-level code changed (imports, constants...), which the
        function-level analysis would otherwise not execute at all.
        """
        timeout = min(120.0, self.deadline.remaining())

        def attempt(worker: Worker, path: str) -> Observation:
            root, module = module_for_path(worker.workspace, path)
            roots = [str((worker.workspace / root).resolve()), str(worker.workspace.resolve())]
            return worker.prepare(module, str((worker.workspace / path).resolve()), roots, timeout)

        self.deadline.check()
        patched = attempt(patch_worker, file_diff.path)
        if patched.status == "ready" or patched.phase != "import":
            return
        self.deadline.check()  # an import cut short by the overall budget is not a broken import
        base = attempt(base_worker, file_diff.old_path) if file_diff.old_path else None
        if base is None:
            result.errors.append(f"the added module {file_diff.path} fails to import: {patched.detail}")
        elif base.status == "ready":
            result.errors.append(
                f"the patched revision of {file_diff.path} fails to import, but the base revision imports: {patched.detail}"
            )
        else:
            self.log(f"{file_diff.path} cannot be imported on either revision: {patched.detail}")

    def harness_failure(self, plan: CallPlan, report: FunctionReport, error: HarnessError, result: CheckResult) -> None:
        """Decide whether "could not execute" is a limitation (skip) or an error (exit 2).

        If the patched module cannot even be imported, reporting "no regression found"
        would be dangerously wrong, so it is recorded as an error. Construction or lookup
        problems are CounterPatch limitations and only skip the function.
        """
        report.status = "skipped"
        patch_failure = error.failures.get("patch")
        base_failure = error.failures.get("base")
        if patch_failure is not None and patch_failure.phase == "import":
            if base_failure is None:
                message = (
                    f"the patched revision of {plan.changed.file} fails to import, but the base revision imports: {patch_failure.detail}"
                )
            else:
                message = (
                    f"{plan.changed.file} cannot be imported on either revision (are its dependencies installed?): {patch_failure.detail}"
                )
            report.skip_reason = message
            result.errors.append(message)
        elif base_failure is not None and base_failure.phase == "import":
            report.skip_reason = (
                f"the base revision of the module cannot be imported, so there is nothing to compare against: {base_failure.detail}"
            )
        else:
            report.skip_reason = f"not automatically exercisable: {error}"
        self.log(f"Skipping {plan.qualified_name}: {report.skip_reason}")

    def save_reproduction(self, plan: CallPlan, finding: Finding, base_workspace: Path, result: CheckResult) -> None:
        import_root, module = module_for_path(self.repo, plan.changed.file)
        hang_seconds = max(2, math.ceil(self.options.call_timeout * 2))
        path = write_reproduction(self.repo, plan, finding, import_root, module, f"{result.base_ref} @ {result.base_sha}", hang_seconds)
        finding.reproduction = path.relative_to(self.repo).as_posix()
        finding.reproduction_verified, finding.reproduction_note = self.verify(path, base_workspace)

    def verify(self, path: Path, base_workspace: Path) -> tuple[bool, str]:
        """Run a reproduction on both revisions: it must pass on base and fail on the patch."""
        base_copy_dir = base_workspace / REPRODUCTIONS_DIR
        base_copy_dir.mkdir(parents=True, exist_ok=True)
        base_copy = base_copy_dir / path.name
        shutil.copyfile(path, base_copy)
        timeout = self.deadline.clamp(_PYTEST_TIMEOUT)
        try:
            patch_run = run_pytest(self.options.python, self.repo, [path], [], timeout)
            base_run = run_pytest(self.options.python, base_workspace, [base_copy], [], timeout)
        except HarnessError as error:
            return False, str(error)
        base_outcome, patch_outcome = base_run.outcomes[base_copy.name], patch_run.outcomes[path.name]
        if base_outcome == "passed" and patch_outcome == "failed":
            return True, "passes on base, fails on patch"
        return False, f"pytest outcome was {base_outcome} on base and {patch_outcome} on patch"

    def run_ai_tests(
        self, suggestions: ai.AISuggestions, changed: list[ChangedFunction], base_workspace: Path, result: CheckResult
    ) -> None:
        roots = sorted({module_for_path(self.repo, function.file)[0] for function in changed if function.status == "modified"})
        patch_dir = ensure_output_dir(self.repo, GENERATED_DIR)
        base_dir = base_workspace / GENERATED_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        written: list[tuple[ai.GeneratedTest, Path, Path]] = []
        try:
            for test in suggestions.tests:
                content = with_import_shim(test.code, roots)
                patch_path = safe_join(patch_dir, test.filename)
                base_path = safe_join(base_dir, test.filename)
                patch_path.write_text(content, encoding="utf-8")
                base_path.write_text(content, encoding="utf-8")
                written.append((test, patch_path, base_path))
            self.log(f"Running {len(written)} generated test(s) against both revisions")
            outcomes = self._run_pair([patch for _, patch, _ in written], [base for _, _, base in written], base_workspace)
            result.ai.tests_executed = len(written)
            for test, patch_path, base_path in written:
                base_outcome, patch_outcome = outcomes[0][base_path.name], outcomes[1][patch_path.name]
                verdict = classify_test_outcomes(base_outcome, patch_outcome, test.targets_intended_change)
                if verdict.kind == "fail-fail":
                    result.ai.tests_non_discriminating += 1
                elif verdict.kind == "invalid":
                    result.ai.tests_invalid += 1
                if verdict.severity not in {Severity.REGRESSION, Severity.DIVERGENCE}:
                    continue
                again = self._run_pair([patch_path], [base_path], base_workspace)
                if (again[0][base_path.name], again[1][patch_path.name]) != (base_outcome, patch_outcome):
                    self.log(f"Generated test {test.name} is flaky; ignoring it")
                    continue
                finding = Finding(
                    file=", ".join(sorted({function.file for function in changed if function.status == "modified"})),
                    function=test.name,
                    signature=None,
                    severity=verdict.severity,
                    kind=verdict.kind,
                    summary=verdict.summary,
                    source="ai-test",
                    intent_note=verdict.intent_note,
                    test_name=test.name,
                    rationale=test.rationale,
                )
                if verdict.severity is Severity.REGRESSION:
                    reproductions = ensure_output_dir(self.repo, REPRODUCTIONS_DIR)
                    destination = safe_join(reproductions, sanitize_filename(test.filename.removeprefix("test_cp_")))
                    shutil.copyfile(patch_path, destination)
                    finding.reproduction = destination.relative_to(self.repo).as_posix()
                    finding.reproduction_verified = True
                    finding.reproduction_note = "passes on base, fails on patch"
                result.ai_findings.append(finding)
        finally:
            for _, patch_path, _ in written:
                patch_path.unlink(missing_ok=True)

    def _run_pair(self, patch_files: list[Path], base_files: list[Path], base_workspace: Path) -> tuple[dict[str, str], dict[str, str]]:
        timeout = self.deadline.clamp(_PYTEST_TIMEOUT)
        base_run = run_pytest(self.options.python, base_workspace, base_files, [], timeout)
        patch_run = run_pytest(self.options.python, self.repo, patch_files, [], timeout)
        return base_run.outcomes, patch_run.outcomes


def run_check(options: CheckOptions, log: Logger = _quiet, ai_client: Any | None = None) -> CheckResult:
    """Run a full check. Raises :class:`CounterPatchError` subclasses for exit-code-2 conditions."""
    return _Run(options, log, ai_client).execute()
