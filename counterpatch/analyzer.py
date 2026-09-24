"""Map a diff onto the Python functions and methods it touches."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from counterpatch.git import show_file
from counterpatch.models import (
    AnalysisResult,
    ChangedFunction,
    FileDiff,
    FileStatus,
    FunctionInfo,
    Param,
)

_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")
_TEST_DIR = re.compile(r"(^|/)tests?/")
_IGNORED_DIRS = {".git", ".venv", "venv", "env", "node_modules", "build", "dist", ".tox", ".nox", ".counterpatch", "site-packages"}
_MAX_TEST_FILES_SCANNED = 2000
_MAX_RELATED_TEST_FILES = 5


def is_test_path(path: str) -> bool:
    return bool(_TEST_FILE.search(path) or _TEST_DIR.search(path)) or path.startswith(".counterpatch/")


def collect_functions(source: str) -> dict[str, FunctionInfo]:
    """Return top-level functions and (nested) class methods keyed by qualified name.

    Functions nested inside other functions are not collected separately: a change to
    them is a change to their enclosing function. Raises ``SyntaxError``.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    found: dict[str, FunctionInfo] = {}

    def visit(body: list[ast.stmt], class_path: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, [*class_path, node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = _function_info(node, class_path, lines)
                found[info.qualified_name] = info

    visit(tree.body, [])
    _attach_raises(found)
    constants = _numeric_constants(tree)
    for info in found.values():
        info.constants = constants
    return found


def _numeric_constants(tree: ast.Module) -> dict[str, float]:
    """``NAME = 32`` / ``NAME: int = -1`` at module level, used as boundary hints."""
    constants: dict[str, float] = {}
    for statement in tree.body:
        targets = (
            statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        )
        value = getattr(statement, "value", None)
        if value is None or not all(isinstance(target, ast.Name) for target in targets):
            continue
        try:
            number = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            constants.update({target.id: number for target in targets})  # type: ignore[union-attr]
    return constants


def raised_types(function: ast.AST) -> set[str]:
    """Names of exception types explicitly raised (``raise X`` / ``raise X(...)``) in a body."""
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _attach_raises(functions: dict[str, FunctionInfo]) -> None:
    """Record deliberately raised exception types, following one level of same-module helpers.

    ``validate(amount)`` or ``self._check(x)`` defined in the same module/class count, so
    validation extracted into a helper is still recognized as deliberate rejection.
    """
    own = {name: raised_types(info.node) for name, info in functions.items()}
    for name, info in functions.items():
        raises = set(own[name])
        for node in ast.walk(info.node):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            if isinstance(callee, ast.Name):
                raises |= own.get(callee.id, set())
            elif (
                isinstance(callee, ast.Attribute)
                and isinstance(callee.value, ast.Name)
                and callee.value.id in {"self", "cls"}
                and info.class_name
            ):
                raises |= own.get(f"{info.class_name}.{callee.attr}", set())
        info.raises = raises


def _function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, class_path: list[str], lines: list[str]) -> FunctionInfo:
    decorators = [_decorator_name(decorator) for decorator in node.decorator_list]
    is_property = any(name in {"property", "cached_property"} or name.endswith((".setter", ".getter", ".deleter")) for name in decorators)
    if not class_path:
        method_kind = "function"
    elif "staticmethod" in decorators:
        method_kind = "static"
    elif "classmethod" in decorators:
        method_kind = "class"
    else:
        method_kind = "instance"

    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    end = node.end_lineno or node.lineno
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return FunctionInfo(
        qualified_name=".".join([*class_path, node.name]),
        class_name=".".join(class_path) or None,
        method_kind=method_kind,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        is_property=is_property,
        params=_params(node.args),
        start_line=start,
        end_line=end,
        signature=f"{prefix}{node.name}({ast.unparse(node.args)}){returns}",
        source="\n".join(lines[start - 1 : end]),
        node=node,
    )


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return ast.unparse(target).removeprefix("functools.").removeprefix("builtins.")


def _params(arguments: ast.arguments) -> list[Param]:
    params: list[Param] = []
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = "positional_only" if index < len(arguments.posonlyargs) else "positional_or_keyword"
        params.append(_param(arg, kind, default))
    if arguments.vararg:
        params.append(_param(arguments.vararg, "var_positional", None))
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        params.append(_param(arg, "keyword_only", default))
    if arguments.kwarg:
        params.append(_param(arguments.kwarg, "var_keyword", None))
    return params


def _param(arg: ast.arg, kind: str, default: ast.expr | None) -> Param:
    return Param(
        name=arg.arg,
        kind=kind,
        annotation=ast.unparse(arg.annotation) if arg.annotation is not None else None,
        default=ast.unparse(default) if default is not None else None,
        has_default=default is not None,
        annotation_node=arg.annotation,
        default_node=default,
    )


def analyze_changes(repo: Path, base_sha: str, diffs: list[FileDiff]) -> AnalysisResult:
    """Find changed functions for every changed non-test Python file."""
    result = AnalysisResult(python_files=[], changed_functions=[])
    for file_diff in diffs:
        if is_test_path(file_diff.path):
            result.test_files.append(file_diff.path)
            continue
        result.python_files.append(file_diff)
        try:
            functions, module_level_changed = _changed_functions_in_file(repo, base_sha, file_diff)
            result.changed_functions.extend(functions)
            if module_level_changed and _import_checkable(file_diff):
                result.import_checks.append(file_diff)
        except SyntaxError as error:
            result.parse_errors.append(f"{file_diff.path}: could not parse ({error.msg} at line {error.lineno})")
    return result


_UNSAFE_TO_IMPORT = {"setup.py", "__main__.py", "manage.py", "noxfile.py", "fabfile.py", "tasks.py", "conf.py"}


def _import_checkable(file_diff: FileDiff) -> bool:
    """Modules that look like libraries (not scripts) and still exist in the patch."""
    return file_diff.status is not FileStatus.DELETED and Path(file_diff.path).name not in _UNSAFE_TO_IMPORT


def _changed_functions_in_file(repo: Path, base_sha: str, file_diff: FileDiff) -> tuple[list[ChangedFunction], bool]:
    """Changed functions, and whether module-level code (outside functions) changed."""
    base_functions: dict[str, FunctionInfo] = {}
    patch_functions: dict[str, FunctionInfo] = {}
    base_tree: ast.Module | None = None
    patch_tree: ast.Module | None = None

    if file_diff.status is not FileStatus.ADDED and file_diff.old_path:
        base_source = show_file(repo, base_sha, file_diff.old_path)
        if base_source is not None:
            base_functions = collect_functions(base_source)
            base_tree = ast.parse(base_source)
    if file_diff.status is not FileStatus.DELETED and file_diff.new_path:
        patch_path = repo / file_diff.new_path
        if patch_path.exists():
            patch_source = patch_path.read_text(encoding="utf-8", errors="replace")
            patch_functions = collect_functions(patch_source)
            patch_tree = ast.parse(patch_source)

    new_lines = file_diff.changed_new_lines()
    old_lines = file_diff.changed_old_lines()
    touched: list[str] = []
    for name, info in patch_functions.items():
        if file_diff.status is FileStatus.ADDED or _overlaps(info, new_lines):
            touched.append(name)
    for name, info in base_functions.items():
        if (file_diff.status is FileStatus.DELETED or _overlaps(info, old_lines)) and name not in touched:
            touched.append(name)

    names, attributes, module_level_changed = _module_level_changes(patch_tree, new_lines)
    base_names, base_attributes, base_module_level_changed = _module_level_changes(base_tree, old_lines)
    names |= base_names
    attributes |= base_attributes
    for name, info in patch_functions.items():
        if name not in touched and _uses_changed_names(info, names, attributes):
            touched.append(name)

    changed: list[ChangedFunction] = []
    for name in touched:
        patch_info = patch_functions.get(name)
        base_info = base_functions.get(name)
        if patch_info and base_info:
            status = "modified"
        elif patch_info:
            status = "added"
        else:
            status = "deleted"
        changed.append(
            ChangedFunction(
                file=file_diff.new_path or file_diff.path,
                base_file=file_diff.old_path,
                qualified_name=name,
                status=status,
                diff=_function_diff(file_diff, base_info, patch_info),
                patch=patch_info,
                base=base_info,
            )
        )
    return changed, module_level_changed or base_module_level_changed


def _statement_names(statement: ast.stmt) -> set[str]:
    """Names bound by a module- or class-level statement."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in statement.names}
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {statement.name}
    names: set[str] = set()
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    for target in targets:
        names |= {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}
    return names


def _module_level_changes(tree: ast.Module | None, lines: set[int]) -> tuple[set[str], set[tuple[str, str]], bool]:
    """Names and class attributes bound by changed statements outside any function."""
    names: set[str] = set()
    attributes: set[tuple[str, str]] = set()
    changed = False
    if tree is None:
        return names, attributes, changed

    def overlaps(node: ast.stmt) -> bool:
        return any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in lines)

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and overlaps(member):
                    changed = True
                    attributes |= {(statement.name, name) for name in _statement_names(member)}
            continue
        if overlaps(statement):
            changed = True
            names |= _statement_names(statement)
    return names, attributes, changed


def _uses_changed_names(info: FunctionInfo, names: set[str], attributes: set[tuple[str, str]]) -> bool:
    """Does the function read a changed module-level name or a changed attribute of its class?"""
    class_name = (info.class_name or "").split(".")[0]
    for node in ast.walk(info.node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
            return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"self", "cls", class_name}
            and (class_name, node.attr) in attributes
        ):
            return True
    return False


def _overlaps(info: FunctionInfo, lines: set[int]) -> bool:
    return any(info.start_line <= line <= info.end_line for line in lines)


def _function_diff(file_diff: FileDiff, base: FunctionInfo | None, patch: FunctionInfo | None) -> str:
    """Hunks of ``file_diff`` that fall inside the function in either revision."""
    selected: list[str] = []
    for hunk in file_diff.hunks:
        in_patch = patch is not None and bool(hunk.new_lines() & set(range(patch.start_line, patch.end_line + 1)))
        in_base = base is not None and bool(hunk.old_lines() & set(range(base.start_line, base.end_line + 1)))
        if in_patch or in_base:
            selected.append(hunk.header)
            selected.extend(hunk.lines)
    if not selected:
        # Touched through a changed module-level name: the relevant change is outside the body.
        for hunk in file_diff.hunks:
            selected.append(hunk.header)
            selected.extend(hunk.lines)
    return "\n".join(selected)


def find_related_tests(repo: Path, module: str, source_path: str) -> list[Path]:
    """Return a few existing test files that import ``module`` or are named after it.

    This is deliberately bounded: it scans at most a couple of thousand files and
    returns at most five.
    """
    leaf = module.rsplit(".", 1)[-1]
    stem = Path(source_path).stem
    patterns = [
        re.compile(rf"^\s*(from|import)\s+{re.escape(module)}\b", re.MULTILINE),
        re.compile(rf"^\s*from\s+[\w.]+\s+import\s+[^\n]*\b{re.escape(leaf)}\b", re.MULTILINE),
    ]
    related: list[Path] = []
    scanned = 0
    for path in sorted(repo.rglob("*.py")):
        if any(part in _IGNORED_DIRS for part in path.relative_to(repo).parts):
            continue
        name = path.name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue
        scanned += 1
        if scanned > _MAX_TEST_FILES_SCANNED:
            break
        if name in {f"test_{stem}.py", f"{stem}_test.py"}:
            related.append(path)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            related.append(path)
        if len(related) >= _MAX_RELATED_TEST_FILES:
            break
    return related[:_MAX_RELATED_TEST_FILES]


def literal_calls_in_tests(test_source: str, function_name: str) -> list[tuple[list[object], dict[str, object]]]:
    """Literal argument lists used when existing tests call ``function_name``.

    Existing tests encode inputs a human already thought were interesting, so they make
    good seeds for adversarial exploration.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    calls: list[tuple[list[object], dict[str, object]]] = []
    nodes = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Call)), key=lambda node: (node.lineno, node.col_offset))
    for node in nodes:
        callee = node.func
        name = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else None
        if name != function_name:
            continue
        try:
            positional = [ast.literal_eval(arg) for arg in node.args]
            keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except (ValueError, TypeError, SyntaxError, RecursionError):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
            continue
        calls.append((positional, keywords))
    return calls
