"""Render minimized counterexamples as standalone pytest files.

A reproduction encodes the *base* revision's behavior, so it passes on the base
revision and fails on the patch. It only depends on pytest and the project itself.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from counterpatch.candidates import CallPlan
from counterpatch.models import Finding, Observation
from counterpatch.utils import safe_join, sanitize_filename

COUNTERPATCH_DIR = ".counterpatch"
REPRODUCTIONS_DIR = f"{COUNTERPATCH_DIR}/reproductions"
GENERATED_DIR = f"{COUNTERPATCH_DIR}/generated"
_HANG_KINDS = {"new-hang"}


def ensure_output_dir(repo: Path, relative: str) -> Path:
    directory = repo / relative
    directory.mkdir(parents=True, exist_ok=True)
    ignore = repo / COUNTERPATCH_DIR / ".gitignore"
    if not ignore.exists():
        ignore.write_text("# CounterPatch scratch output. Reproductions may be committed.\ngenerated/\nreport.json\n", encoding="utf-8")
    return directory


def value_source(value: Any) -> str:
    """Python source for ``value`` that also works for ``inf``/``nan`` and nested containers."""
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    if isinstance(value, list):
        return "[" + ", ".join(value_source(item) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(value_source(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, set):
        return "{" + ", ".join(value_source(item) for item in value) + "}" if value else "set()"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{value_source(key)}: {value_source(item)}" for key, item in value.items()) + "}"
    return repr(value)


def import_shim(import_root: str) -> str:
    target = "Path(__file__).resolve().parents[2]" + ("" if import_root == "." else f" / {json.dumps(import_root)}")
    return f"sys.path.insert(0, str({target}))"


def receiver_expression(plan: CallPlan) -> tuple[str, str]:
    """Return ``(name to import, expression to call)`` for the changed function."""
    parts = plan.qualified_name.split(".")
    if plan.method_kind == "function":
        return parts[0], parts[0]
    owner = ".".join(parts[:-1])
    if plan.method_kind == "instance":
        return parts[0], f"{owner}().{parts[-1]}"
    return parts[0], plan.qualified_name


def call_arguments(plan: CallPlan, args: dict[str, Any], names: dict[str, str] | None = None) -> str:
    """Argument list, positional where readable, keyword otherwise."""
    rendered: list[str] = []
    for slot in plan.slots:
        if slot.name not in args:
            continue
        source = (names or {}).get(slot.name) or value_source(args[slot.name])
        rendered.append(source if slot.leading_positional or slot.positional else f"{slot.name}={source}")
    return ", ".join(rendered)


def call_expression(plan: CallPlan, args: dict[str, Any]) -> str:
    return f"{receiver_expression(plan)[1]}({call_arguments(plan, args)})"


def test_function_name(plan: CallPlan, kind: str) -> str:
    stem = sanitize_filename(f"{plan.qualified_name}_{kind}", prefix="test_", suffix="")
    return stem if stem.isidentifier() else "test_counterexample"


def render_reproduction(plan: CallPlan, finding: Finding, import_root: str, module: str, base_label: str, hang_seconds: int = 10) -> str:
    assert finding.base is not None and finding.patch is not None
    imported, receiver = receiver_expression(plan)
    base, patch = finding.base, finding.patch
    imports = {"import sys", "from pathlib import Path"}
    third_party: set[str] = set()
    project_imports = [f"from {module} import {imported}"]
    fixtures = ""
    body: list[str] = []

    mutable_names = {name for name in base.args_after_source if name in finding.args}
    check_mutation = finding.kind == "argument-mutation-changed" and bool(mutable_names)
    names: dict[str, str] = {}
    if check_mutation:
        for name in sorted(mutable_names):
            body.append(f"{name} = {value_source(finding.args[name])}")
            names[name] = name
    check_state = finding.kind == "object-state-changed" and base.state_after_source is not None and plan.method_kind == "instance"
    if check_state:
        owner = ".".join(plan.qualified_name.split(".")[:-1])
        body.append(f"obj = {owner}()")
        call = f"obj.{plan.qualified_name.split('.')[-1]}({call_arguments(plan, finding.args, names)})"
    else:
        call = f"{receiver}({call_arguments(plan, finding.args, names)})"

    if base.status == "raised":
        third_party.add("import pytest")
        exception = base.exception_type or "Exception"
        if "." not in exception:
            body += [f"with pytest.raises({exception}):", f"    {call}"]
        elif exception.rsplit(".", 1)[0] == module and "." not in exception.rsplit(".", 1)[1]:
            name = exception.rsplit(".", 1)[1]
            project_imports.append(f"from {module} import {name}")
            body += [f"with pytest.raises({name}):", f"    {call}"]
        else:
            body += [
                "with pytest.raises(Exception) as excinfo:",
                f"    {call}",
                f"assert type(excinfo.value).__name__ == {base.exception_short_name!r}",
            ]
    else:
        body.append(f"result = {call}")
        assertion = _return_assertion(base)
        if finding.kind == "output-changed":
            fixtures = "capsys"
            body.append(f"assert capsys.readouterr().out == {base.stdout!r}")
        elif check_mutation:
            body += [f"assert {name} == {base.args_after_source[name]}" for name in sorted(mutable_names)]
        elif check_state:
            body.append(f"assert vars(obj) == {base.state_after_source}")
        if assertion and finding.kind not in {"output-changed", "argument-mutation-changed", "object-state-changed"}:
            if "math." in assertion:
                imports.add("import math")
            body.append(assertion)
        elif not assertion and finding.kind not in {"output-changed", "argument-mutation-changed", "object-state-changed"}:
            body.append(f"# The base revision returned {base.return_value_repr}, which cannot be written as a literal.")

    helpers: list[str] = []
    if finding.kind in _HANG_KINDS:
        imports.add("import signal")
        helpers = [
            "def _fail_on_timeout(signum: int, frame: object) -> None:",
            f'    raise TimeoutError("no result within {hang_seconds}s")',
            "",
            "",
        ]
        body = [
            "signal.signal(signal.SIGALRM, _fail_on_timeout)",
            f"signal.alarm({hang_seconds})",
            "try:",
            *[f"    {line}" for line in body],
            "finally:",
            "    signal.alarm(0)",
        ]

    docstring = [
        f'"""CounterPatch reproduction for {module}.{plan.qualified_name}.',
        "",
        f"{_label(finding)}: {finding.summary}.",
        f"Base revision ({base_label}): {base.describe()}",
        f"Patched revision: {patch.describe()}",
    ]
    if finding.intent_note:
        docstring.append(finding.intent_note)
    docstring += [
        "",
        "This test encodes the base revision's behavior: it passes on the base",
        "revision and fails on the patch. If the new behavior is intended, delete it.",
        '"""',
    ]
    lines = [
        *docstring,
        "",
        *sorted(imports, key=lambda line: (line.startswith("from"), line)),
        *(["", *sorted(third_party)] if third_party else []),
        "",
        import_shim(import_root),
        "",
        *project_imports,
        "",
        "",
        *helpers,
        f"def {test_function_name(plan, finding.kind)}({fixtures}) -> None:",
        *[f"    {line}" for line in body],
        "",
    ]
    return "\n".join(lines).replace("\n\n\n\n", "\n\n\n")


def _return_assertion(base: Observation) -> str | None:
    if base.return_value_source is not None:
        return f"assert result == {base.return_value_source}"
    special = {"nan": "assert math.isnan(result)", "inf": 'assert result == float("inf")', "-inf": 'assert result == float("-inf")'}
    if base.return_value_repr in special:
        return special[base.return_value_repr]
    if base.return_value_repr and "0x…" not in base.return_value_repr and not re.search(r"[{(]", base.return_value_repr):
        return f"assert repr(result) == {base.return_value_repr!r}"
    return None


def _label(finding: Finding) -> str:
    return "Regression candidate" if finding.severity.value == "regression" else "Behavioral divergence"


def write_reproduction(
    repo: Path, plan: CallPlan, finding: Finding, import_root: str, module: str, base_label: str, hang_seconds: int = 10
) -> Path:
    directory = ensure_output_dir(repo, REPRODUCTIONS_DIR)
    path = safe_join(directory, sanitize_filename(f"{plan.qualified_name}_{finding.kind}"))
    path.write_text(render_reproduction(plan, finding, import_root, module, base_label, hang_seconds), encoding="utf-8")
    return path


def with_import_shim(code: str, import_roots: list[str]) -> str:
    """Prefix generated test code with ``sys.path`` setup so it runs from ``.counterpatch/*/``."""
    shims = "\n".join(import_shim(root) for root in dict.fromkeys(import_roots))
    return f"import sys\nfrom pathlib import Path\n\n{shims}\n\n{code.strip()}\n"
