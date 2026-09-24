"""Optional Anthropic-powered adversarial candidate generation.

The model is a hypothesis generator only. It proposes (a) literal inputs for changed
functions and (b) small pytest tests. Everything it returns is treated as untrusted:
inputs are parsed with ``ast.literal_eval``; test code is statically validated,
written only into ``.counterpatch/generated/`` under sanitized names, and executed in
subprocesses against both revisions. Differential execution decides what counts.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any

from counterpatch.candidates import CallPlan, conforms
from counterpatch.models import ChangedFunction, ConfigError, CounterPatchError
from counterpatch.utils import sanitize_filename, truncate

DEFAULT_MODEL = "claude-opus-5"
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}
MAX_INPUTS_PER_FUNCTION = 15
MAX_TESTS = 8
MAX_TEST_CODE_CHARS = 8000
_MAX_SOURCE_CHARS = 6000
_MAX_TEST_CONTEXT_CHARS = 4000

SAFE_TEST_IMPORTS = frozenset(
    {
        "pytest",
        "math",
        "cmath",
        "hypothesis",
        "typing",
        "decimal",
        "fractions",
        "re",
        "string",
        "itertools",
        "collections",
        "functools",
        "dataclasses",
        "enum",
        "datetime",
        "copy",
        "operator",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "breakpoint",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "exit",
        "quit",
        "memoryview",
    }
)
_ALLOWED_DUNDERS = frozenset({"__name__", "__doc__"})

SYSTEM_PROMPT = """You generate adversarial test candidates for CounterPatch, a differential regression-testing tool.

You are NOT asked whether a patch is correct, and your opinion is not evidence. Every candidate you produce is executed against BOTH the base revision and the patched revision of the code; only a concrete behavioral difference counts as a signal.

Your job: generate concrete, executable adversarial inputs and small pytest tests that could falsify the intended behavior if the patch introduced a regression. Target:
- boundaries and off-by-one errors (values just below, at and above every limit in the code)
- missing or weakened validation (inputs the base rejects)
- empty values, None, whitespace, negative, zero and extreme values
- incorrect assumptions introduced by the change
- interactions between branches that the diff touched
- state transitions, when the function is a method

Rules:
- Inputs: give each argument value as a Python literal (ast.literal_eval-compatible). Use only the listed parameter names.
- Tests: plain pytest functions named test_*; import the code exactly as shown; only import pytest, hypothesis, and standard-library modules for math, strings, collections, typing, decimal, fractions, datetime, itertools, functools, re, enum, dataclasses. No file, network, process, environment or OS access. No eval/exec/open/getattr.
- Tests must assert the behavior the task (and the base revision) implies, so that a correct patch passes them.
- Set targets_intended_change=true for candidates that exercise the behavior the task explicitly asks to change.
- Prefer a few sharp candidates over many redundant ones."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "arguments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "rationale": {"type": "string"},
                    "targets_intended_change": {"type": "boolean"},
                },
                "required": ["function", "arguments", "rationale", "targets_intended_change"],
                "additionalProperties": False,
            },
        },
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "code": {"type": "string"},
                    "targets_intended_change": {"type": "boolean"},
                },
                "required": ["name", "rationale", "code", "targets_intended_change"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["inputs", "tests"],
    "additionalProperties": False,
}


class AIError(CounterPatchError):
    """The Anthropic API call failed or returned unusable output."""


@dataclass
class GeneratedInput:
    function: str
    args: dict[str, Any]
    rationale: str
    targets_intended_change: bool


@dataclass
class GeneratedTest:
    name: str
    rationale: str
    code: str
    targets_intended_change: bool
    filename: str = ""


@dataclass
class AISuggestions:
    inputs: list[GeneratedInput] = field(default_factory=list)
    tests: list[GeneratedTest] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


@dataclass
class FunctionPromptContext:
    changed: ChangedFunction
    import_line: str
    call_hint: str
    parameters: str | None  # None when CounterPatch cannot call the function itself


def require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError("--ai requires the ANTHROPIC_API_KEY environment variable. Run without --ai for deterministic mode.")


def build_prompt(functions: list[FunctionPromptContext], task: str | None, test_context: list[tuple[str, str]]) -> str:
    """Only relevant context: task, changed functions (both revisions), diff, nearby tests."""
    sections = [f"Task description (context, may be incomplete):\n{task or '(none provided)'}"]
    for context in functions:
        changed = context.changed
        assert changed.base is not None and changed.patch is not None
        if context.parameters is None:
            params = "none: CounterPatch cannot call this function directly, so only tests (not inputs) can target it"
        else:
            params = context.parameters or "(no arguments)"
        sections.append(
            "\n".join(
                [
                    f"## Function `{changed.qualified_name}` in {changed.file}",
                    f"Import: {context.import_line}",
                    f"Call as: {context.call_hint}",
                    f"Parameters CounterPatch can vary: {params}",
                    "Base revision source:",
                    "```python",
                    truncate(changed.base.source, _MAX_SOURCE_CHARS),
                    "```",
                    "Patched revision source:",
                    "```python",
                    truncate(changed.patch.source, _MAX_SOURCE_CHARS),
                    "```",
                    "Diff:",
                    "```diff",
                    truncate(changed.diff, _MAX_SOURCE_CHARS),
                    "```",
                ]
            )
        )
    for path, text in test_context[:3]:
        sections.append(
            f"Existing test file {path} (for conventions and established behavior):\n```python\n{truncate(text, _MAX_TEST_CONTEXT_CHARS)}\n```"
        )
    sections.append(
        f"Return up to {MAX_INPUTS_PER_FUNCTION} inputs per function and up to {MAX_TESTS} tests in total. "
        "Use the exact function names shown above in `function`."
    )
    return "\n\n".join(sections)


def request_suggestions(prompt: str, model: str, client: Any | None = None) -> dict[str, Any]:
    """Call the Messages API with a JSON-schema constrained output."""
    if client is None:
        try:
            import anthropic
        except ImportError as error:
            raise ConfigError("--ai requires the Anthropic SDK: pip install 'counterpatch[ai]'") from error
        client = anthropic.Anthropic(timeout=300.0, max_retries=2)
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }
    if model in _FALLBACK_MODELS:
        request |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        response = client.beta.messages.create(**request)
    except Exception as error:  # noqa: BLE001 - SDK errors, network errors; mapped to exit code 2
        raise AIError(f"Anthropic API request failed: {type(error).__name__}: {truncate(str(error), 300)}") from error
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        raise AIError("The model declined to generate candidates (stop_reason=refusal)")
    if stop_reason == "max_tokens":
        raise AIError("The model response was truncated (stop_reason=max_tokens)")
    text = next((block.text for block in response.content if getattr(block, "type", None) == "text"), None)
    if text is None:
        raise AIError("The model response contained no text block")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise AIError(f"The model returned invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise AIError("The model returned JSON that is not an object")
    return data


def parse_suggestions(data: dict[str, Any], plans: dict[str, CallPlan], allowed_modules: set[str]) -> AISuggestions:
    """Validate untrusted model output; anything invalid is dropped with a reason."""
    suggestions = AISuggestions()
    per_function: dict[str, int] = {}
    for item in data.get("inputs", []) if isinstance(data.get("inputs"), list) else []:
        try:
            generated = _parse_input(item, plans)
        except ValueError as error:
            suggestions.rejected.append(f"input rejected: {error}")
            continue
        per_function[generated.function] = per_function.get(generated.function, 0) + 1
        if per_function[generated.function] <= MAX_INPUTS_PER_FUNCTION:
            suggestions.inputs.append(generated)

    used_names: set[str] = set()
    for item in data.get("tests", []) if isinstance(data.get("tests"), list) else []:
        if len(suggestions.tests) >= MAX_TESTS:
            break
        if not isinstance(item, dict) or not isinstance(item.get("code"), str):
            suggestions.rejected.append("test rejected: malformed entry")
            continue
        name = str(item.get("name", "generated"))
        problem = validate_test_code(item["code"], allowed_modules)
        if problem:
            suggestions.rejected.append(f"test {truncate(name, 60)!r} rejected: {problem}")
            continue
        filename = sanitize_filename(f"cp_ai_{name}")
        while filename in used_names:
            filename = filename[:-3] + "_x.py"
        used_names.add(filename)
        suggestions.tests.append(
            GeneratedTest(
                name=truncate(name, 80),
                rationale=truncate(str(item.get("rationale", "")), 300),
                code=item["code"],
                targets_intended_change=bool(item.get("targets_intended_change", False)),
                filename=filename,
            )
        )
    return suggestions


def _parse_input(item: Any, plans: dict[str, CallPlan]) -> GeneratedInput:
    if not isinstance(item, dict):
        raise ValueError("malformed entry")
    function = str(item.get("function", ""))
    plan = plans.get(function)
    if plan is None:
        raise ValueError(f"unknown function {truncate(function, 60)!r}")
    slots = {slot.name: slot for slot in plan.slots}
    args: dict[str, Any] = {}
    for argument in item.get("arguments", []) or []:
        name = str(argument.get("name", "")) if isinstance(argument, dict) else ""
        if name not in slots:
            raise ValueError(f"unknown parameter {truncate(name, 40)!r} for {function}")
        source = str(argument.get("value", ""))
        if len(source) > 5000:
            raise ValueError(f"value for {name} is too large")
        try:
            value = ast.literal_eval(source)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as error:
            raise ValueError(f"value for {name} is not a Python literal") from error
        if not conforms(value, slots[name].spec):
            raise ValueError(f"value for {name} does not match {slots[name].spec.describe()}")
        args[name] = value
    for name, slot in slots.items():
        args.setdefault(name, slot.pool[0])
    ordered = {slot.name: args[slot.name] for slot in plan.slots}
    return GeneratedInput(
        function, ordered, truncate(str(item.get("rationale", "")), 300), bool(item.get("targets_intended_change", False))
    )


def validate_test_code(code: str, allowed_modules: set[str]) -> str | None:
    """Return a reason to reject ``code``, or ``None`` if it passes static checks.

    These checks reduce accidental damage from generated code; they are not a
    security boundary. Generated tests still execute arbitrary Python.
    """
    if len(code) > MAX_TEST_CODE_CHARS:
        return "code is too long"
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return f"syntax error: {error.msg}"
    allowed_roots = SAFE_TEST_IMPORTS | {module.split(".")[0] for module in allowed_modules}
    has_test = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_test |= node.name.startswith("test")
        elif isinstance(node, ast.ClassDef):
            has_test |= node.name.startswith("Test")
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except ValueError:
                return "top-level assignments must be literals"
        elif not isinstance(node, (ast.Import, ast.ImportFrom)) and not _is_docstring(node):
            return f"unsupported top-level statement ({type(node).__name__})"
    if not has_test:
        return "no test function found"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_roots:
                    return f"import of {alias.name!r} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in allowed_roots:
                return f"import from {node.module!r} is not allowed"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            return f"call to {node.func.id}() is not allowed"
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_CALLS | {"__builtins__"}:
            return f"use of {node.id!r} is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in _ALLOWED_DUNDERS:
            return f"access to {node.attr!r} is not allowed"
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
