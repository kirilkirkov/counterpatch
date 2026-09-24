"""Human-readable, Markdown, JSON and GitHub annotation output for a check."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any

from counterpatch import __version__
from counterpatch.models import CheckResult, Finding, Severity
from counterpatch.utils import truncate

REGRESSION_HEADLINE = "✗ Regression candidate found"
DIVERGENCE_HEADLINE = "⚠ Behavioral divergence found"
CLEAN_HEADLINE = "✓ No regression counterexample found."
ERROR_HEADLINE = "✗ CounterPatch could not complete the check"
DISCLAIMER = "This does not prove the patch is correct."
_SOURCES = {
    "deterministic": "deterministic boundary sweep",
    "hypothesis": "Hypothesis exploration",
    "existing-test": "inputs from existing tests",
    "ai-input": "AI-suggested input",
    "ai-test": "AI-generated test, executed on both revisions",
}


def _indent(text: str, spaces: int = 4) -> str:
    return "\n".join(" " * spaces + line if line else line for line in text.splitlines())


def render_text(result: CheckResult, verbose: bool = False) -> str:
    lines = [f"CounterPatch v{__version__}", ""]
    lines.append(f"Base: {result.base_ref} ({result.base_sha})")
    lines.append(f"Patch: {result.patch_label}")
    if result.task:
        lines.append(f"Task: {truncate(result.task.splitlines()[0], 100)}")
    lines.append("")
    tested = [report for report in result.functions if report.status == "tested"]
    skipped = [report for report in result.functions if report.status == "skipped"]
    lines.append(f"Changed Python files: {len(result.changed_files)}")
    lines.append(f"Changed functions: {len(result.functions)}")
    lines.append(f"Functions exercised: {len(tested)}")
    lines.append(f"Candidates explored: {sum(report.candidates for report in result.functions)}")
    lines.append(f"Differential executions: {sum(report.executions for report in result.functions)}")
    lines.append(f"Skipped functions: {len(skipped)}")
    if result.related_tests:
        lines.append(f"Existing tests consulted: {len(result.related_tests)}")
    if result.ai.enabled:
        ai_line = f"AI candidates: {result.ai.inputs_accepted} inputs, {result.ai.tests_accepted} tests ({result.ai.model})"
        if result.ai.rejected:
            ai_line += f", {len(result.ai.rejected)} rejected"
        lines.append(ai_line)
    lines.append("")

    for finding in result.findings:
        lines += _render_finding(finding, verbose)

    if skipped:
        lines.append("Not automatically exercisable:")
        for report in skipped:
            lines.append(f"  - {report.signature or report.function} ({report.file}): {report.skip_reason}")
        lines.append("")
    for report in result.functions:
        for note in report.notes:
            lines.append(f"Note: {note}")
    if result.parse_errors:
        lines += [f"Parse error: {error}" for error in result.parse_errors] + [""]
    if verbose:
        lines += _verbose_details(result)
    for error in result.errors:
        lines.append(f"Error: {error}")

    lines.append(_summary_line(result))
    if not result.regressions and not result.errors and not result.timed_out:
        if not tested and not result.ai_findings and not result.ai.tests_executed:
            lines.append("Nothing was executed, so nothing was tested.")
        lines.append(DISCLAIMER)
    return "\n".join(lines).rstrip() + "\n"


def _summary_line(result: CheckResult) -> str:
    regressions, divergences = len(result.regressions), len(result.divergences)
    if regressions:
        return f"{REGRESSION_HEADLINE}: {regressions} regression candidate(s), {divergences} divergence(s)."
    if result.timed_out:
        return f"{ERROR_HEADLINE}: the --timeout budget was exhausted before the check finished."
    if result.errors:
        return f"{ERROR_HEADLINE}: {len(result.errors)} error(s); the result is incomplete."
    if divergences:
        suffix = " (failing because of --fail-on-divergence)" if result.fail_on_divergence else ""
        return f"{DIVERGENCE_HEADLINE}: {divergences} divergence(s), no regression candidates{suffix}."
    return CLEAN_HEADLINE


def _render_finding(finding: Finding, verbose: bool) -> list[str]:
    headline = REGRESSION_HEADLINE if finding.severity is Severity.REGRESSION else DIVERGENCE_HEADLINE
    lines = [headline, ""]
    if finding.source == "ai-test":
        lines += ["Generated test:", _indent(finding.test_name or ""), ""]
        if finding.rationale:
            lines += ["Rationale (from the generator, not evidence):", _indent(finding.rationale), ""]
        lines += ["Result:", _indent(finding.summary + "."), ""]
    else:
        location = f"{finding.file}:{finding.line}" if finding.line else finding.file
        lines += ["File:", _indent(location), "", "Function:", _indent(finding.signature or finding.function), ""]
        lines.append("Minimal counterexample:")
        lines += [_indent(f"{name} = {value!r:.200}") for name, value in finding.args.items()] or [_indent("(no arguments)")]
        lines.append("")
        if finding.call:
            lines += ["Call:", _indent(finding.call), ""]
        assert finding.base is not None and finding.patch is not None
        lines += ["Base revision:", _indent(finding.base.describe()), "", "Patched revision:", _indent(finding.patch.describe()), ""]
        lines += ["Why:", _indent(finding.summary + ".")]
    if finding.intent_note:
        lines.append(_indent(finding.intent_note))
    lines.append("")
    if finding.other_examples:
        lines.append("Also differs for:")
        for example in finding.other_examples:
            args = ", ".join(f"{name}={value!r:.60}" for name, value in example.args.items())
            lines.append(_indent(f"{args}: base {example.base.describe()}; patch {example.patch.describe()}"))
        lines.append("")
    source = _SOURCES.get(finding.source, finding.source)
    if finding.shrunk_from is not None:
        original = ", ".join(f"{name}={value!r:.40}" for name, value in finding.shrunk_from.items())
        source += f"; minimized from {original}"
    lines += [f"Found by: {source}", ""]
    if finding.reproduction:
        status = ""
        if finding.reproduction_verified:
            status = f"  (verified: {finding.reproduction_note})"
        elif finding.reproduction_verified is False:
            status = f"  (not verified: {finding.reproduction_note})"
        lines += ["Reproduction:", _indent(finding.reproduction + status), "", "Run:", _indent(f"pytest {finding.reproduction} -q"), ""]
    if verbose and finding.patch and finding.patch.detail:
        lines += ["Patched revision traceback (tail):", _indent(finding.patch.detail.strip()), ""]
    return lines


def _verbose_details(result: CheckResult) -> list[str]:
    lines = ["Details:"]
    for report in result.functions:
        lines.append(
            f"  {report.function} [{report.status}] candidates={report.candidates} executions={report.executions} "
            f"improvements={report.improvements}"
        )
    if result.changed_test_files:
        lines.append(f"  Changed test files (not exercised): {', '.join(result.changed_test_files)}")
    if result.ai.enabled:
        lines.append(
            f"  AI tests: executed={result.ai.tests_executed} non-discriminating={result.ai.tests_non_discriminating} "
            f"invalid={result.ai.tests_invalid}"
        )
        lines += [f"  AI {reason}" for reason in result.ai.rejected]
    lines.append(f"  Duration: {result.duration_seconds:.1f}s")
    lines.append("")
    return lines


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else repr(value)
    return repr(value)


def render_json(result: CheckResult) -> str:
    data = _jsonable(result)
    data["version"] = __version__
    data["exit_code"] = result.exit_code
    data["findings"] = [_jsonable(finding) for finding in result.findings]
    return json.dumps(data, indent=2, ensure_ascii=False)


def render_markdown(result: CheckResult) -> str:
    """Short summary for ``$GITHUB_STEP_SUMMARY``."""
    lines = [f"## CounterPatch v{__version__}", "", f"**{_summary_line(result)}**", ""]
    lines.append(
        f"Base `{result.base_ref}` ({result.base_sha}) · {len(result.functions)} changed function(s) · "
        f"{sum(report.candidates for report in result.functions)} candidates explored"
    )
    lines.append("")
    for finding in result.findings:
        label = "Regression candidate" if finding.severity is Severity.REGRESSION else "Behavioral divergence"
        lines.append(f"### {label}: `{finding.function}`")
        if finding.call and finding.base and finding.patch:
            lines += [
                "",
                "```python",
                finding.call,
                "```",
                "",
                f"- Base: {finding.base.describe()}",
                f"- Patch: {finding.patch.describe()}",
            ]
        lines.append(f"- {finding.summary}.")
        if finding.intent_note:
            lines.append(f"- {finding.intent_note}")
        if finding.reproduction:
            lines.append(f"- Reproduction: `{finding.reproduction}` (download it from the job workspace or re-run CounterPatch locally)")
        lines.append("")
    if not result.regressions:
        lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines) + "\n"


def github_annotations(result: CheckResult) -> list[str]:
    """Workflow commands that surface findings as PR annotations."""
    annotations = []
    for finding in result.findings:
        if finding.source == "ai-test" or not finding.base or not finding.patch:
            continue
        level = "error" if finding.severity is Severity.REGRESSION else "warning"
        message = f"{finding.summary}: {finding.call} -> base {finding.base.describe()}; patch {finding.patch.describe()}"
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        annotations.append(f"::{level} file={finding.file},line={finding.line},title=CounterPatch::{truncate(message, 900)}")
    return annotations
