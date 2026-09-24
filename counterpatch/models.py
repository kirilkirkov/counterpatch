"""Typed data structures shared across CounterPatch modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CounterPatchError(Exception):
    """Base class for errors that map to exit code 2."""


class ConfigError(CounterPatchError):
    """Invalid CLI options, missing files or missing credentials."""


class CheckTimeout(CounterPatchError):
    """The overall time budget for a check was exhausted."""


class HarnessError(CounterPatchError):
    """CounterPatch could not execute a candidate (as opposed to the candidate failing).

    ``failures`` maps a revision (``base``/``patch``) to the observation describing why
    it could not be prepared, so callers can tell "the patch no longer imports" from
    "this function needs arguments CounterPatch cannot build".
    """

    def __init__(self, message: str, failures: dict[str, Observation] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or {}

    def phase(self, side: str) -> str | None:
        failure = self.failures.get(side)
        return failure.phase if failure else None


class FileStatus(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class Hunk:
    """One ``@@`` hunk of a zero-context unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)

    @property
    def header(self) -> str:
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"

    def new_lines(self) -> set[int]:
        return set(range(self.new_start, self.new_start + self.new_count))

    def old_lines(self) -> set[int]:
        return set(range(self.old_start, self.old_start + self.old_count))


@dataclass
class FileDiff:
    """Changes to a single file between the base revision and the patch."""

    old_path: str | None
    new_path: str | None
    status: FileStatus
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    def changed_new_lines(self) -> set[int]:
        lines: set[int] = set()
        for hunk in self.hunks:
            lines |= hunk.new_lines()
        return lines

    def changed_old_lines(self) -> set[int]:
        lines: set[int] = set()
        for hunk in self.hunks:
            lines |= hunk.old_lines()
        return lines


@dataclass(frozen=True)
class Param:
    """A function parameter as declared in source."""

    name: str
    kind: str  # positional_only | positional_or_keyword | keyword_only | var_positional | var_keyword
    annotation: str | None
    default: str | None
    has_default: bool
    annotation_node: ast.expr | None = field(default=None, compare=False, repr=False, hash=False)
    default_node: ast.expr | None = field(default=None, compare=False, repr=False, hash=False)


@dataclass
class FunctionInfo:
    """A top-level function or class method discovered with ``ast``."""

    qualified_name: str
    class_name: str | None
    method_kind: str  # function | instance | static | class
    is_async: bool
    is_property: bool
    params: list[Param]
    start_line: int
    end_line: int
    signature: str
    source: str
    node: ast.FunctionDef | ast.AsyncFunctionDef = field(compare=False, repr=False)
    raises: set[str] = field(default_factory=set, compare=False)  # explicit raises, incl. direct same-module helpers
    constants: dict[str, float] = field(default_factory=dict, compare=False)  # numeric module-level constants

    @property
    def name(self) -> str:
        return self.qualified_name.rsplit(".", 1)[-1]


@dataclass
class ChangedFunction:
    """A function touched by the diff, with both revisions when available."""

    file: str
    base_file: str | None
    qualified_name: str
    status: str  # modified | added | deleted
    diff: str
    patch: FunctionInfo | None
    base: FunctionInfo | None

    @property
    def signature(self) -> str | None:
        info = self.patch or self.base
        return info.signature if info else None

    @property
    def start_line(self) -> int:
        info = self.patch or self.base
        return info.start_line if info else 0


@dataclass
class AnalysisResult:
    """Everything the analyzer learned from the diff."""

    python_files: list[FileDiff]
    changed_functions: list[ChangedFunction]
    test_files: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    import_checks: list[FileDiff] = field(default_factory=list)  # modules with module-level changes


@dataclass
class Observation:
    """What happened when a candidate was executed on one revision."""

    status: str  # returned | raised | timeout | crashed | error
    return_value_repr: str | None = None
    return_value_key: str | None = None
    return_value_source: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    args_after: dict[str, str] = field(default_factory=dict)
    args_after_source: dict[str, str] = field(default_factory=dict)
    args_after_key: str | None = None
    state_after: str | None = None
    state_after_source: str | None = None
    state_after_key: str | None = None
    detail: str | None = None
    phase: str | None = None  # for status == "error": import | resolve | construct
    timeout_seconds: float | None = None

    @property
    def returned(self) -> bool:
        return self.status == "returned"

    @property
    def exception_short_name(self) -> str | None:
        if self.exception_type is None:
            return None
        return self.exception_type.rsplit(".", 1)[-1]

    def describe(self) -> str:
        """Human readable one-liner used in reports."""
        if self.status == "returned":
            return f"returns {self.return_value_repr}"
        if self.status == "raised":
            message = self.exception_message or ""
            return f"raises {self.exception_type}({message!r})" if message else f"raises {self.exception_type}"
        if self.status == "timeout":
            limit = f" within {self.timeout_seconds:g}s" if self.timeout_seconds else " before the per-call timeout"
            return f"did not finish{limit}"
        if self.status == "crashed":
            return f"crashed the interpreter ({self.detail or 'abnormal exit'})"
        return f"could not be executed ({self.detail or 'unknown error'})"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


class Severity(StrEnum):
    NONE = "none"
    IMPROVEMENT = "improvement"
    DIVERGENCE = "divergence"
    REGRESSION = "regression"


@dataclass(frozen=True)
class Verdict:
    """Classification of a base-vs-patch observation pair."""

    severity: Severity
    kind: str
    summary: str
    intent_note: str | None = None


@dataclass
class Example:
    """A candidate input together with what both revisions did with it."""

    args: dict[str, Any]
    base: Observation
    patch: Observation
    verdict: Verdict


@dataclass
class Finding:
    """A reportable behavioral difference, already minimized when possible."""

    file: str
    function: str
    signature: str | None
    severity: Severity
    kind: str
    summary: str
    source: str  # deterministic | hypothesis | ai-input | ai-test
    args: dict[str, Any] = field(default_factory=dict)
    call: str | None = None
    base: Observation | None = None
    patch: Observation | None = None
    intent_note: str | None = None
    line: int = 0
    shrunk_from: dict[str, Any] | None = None
    other_examples: list[Example] = field(default_factory=list)
    reproduction: str | None = None
    reproduction_verified: bool | None = None
    reproduction_note: str | None = None
    test_name: str | None = None
    rationale: str | None = None


@dataclass
class FunctionReport:
    """Per-function result of a check."""

    file: str
    function: str
    signature: str | None
    status: str  # tested | skipped
    skip_reason: str | None = None
    candidates: int = 0
    executions: int = 0
    improvements: int = 0
    non_discriminating: int = 0
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AIReport:
    enabled: bool = False
    model: str | None = None
    inputs_accepted: int = 0
    tests_accepted: int = 0
    rejected: list[str] = field(default_factory=list)
    tests_executed: int = 0
    tests_non_discriminating: int = 0
    tests_invalid: int = 0
    error: str | None = None


@dataclass
class CheckResult:
    """Everything needed to render a report and pick an exit code."""

    base_ref: str
    base_sha: str
    patch_label: str
    task: str | None
    changed_files: list[str] = field(default_factory=list)
    changed_test_files: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    functions: list[FunctionReport] = field(default_factory=list)
    ai_findings: list[Finding] = field(default_factory=list)
    ai: AIReport = field(default_factory=AIReport)
    related_tests: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timed_out: bool = False
    fail_on_divergence: bool = False
    duration_seconds: float = 0.0

    @property
    def findings(self) -> list[Finding]:
        found = [finding for report in self.functions for finding in report.findings]
        found.extend(self.ai_findings)
        order = {Severity.REGRESSION: 0, Severity.DIVERGENCE: 1}
        return sorted(found, key=lambda finding: order.get(finding.severity, 2))

    @property
    def regressions(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity is Severity.REGRESSION]

    @property
    def divergences(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity is Severity.DIVERGENCE]

    @property
    def exit_code(self) -> int:
        if self.regressions:
            return 1
        if self.timed_out or self.errors:
            return 2
        if self.fail_on_divergence and self.divergences:
            return 1
        return 0
