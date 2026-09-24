"""Standalone probe worker executed in a subprocess, once per revision.

It must only depend on the standard library because it runs under the target
project's interpreter. Protocol: one JSON request per line on stdin, one JSON
response per line on a private copy of the original stdout file descriptor. The
real fd 1 is redirected to stderr so stray writes by target code cannot corrupt
the protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import traceback

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_MAX_REPR = 2000
_EVAL_NAMES = {"__builtins__": {}, "inf": math.inf, "nan": math.nan, "set": set, "frozenset": frozenset}
_modules: dict[str, object] = {}


def canonical(value: object, depth: int = 0, approximate: bool = False) -> str:
    """Order-insensitive, address-free representation.

    With ``approximate=True`` floats are rounded to 12 significant digits, so results
    that differ only by floating-point noise (e.g. summation order) compare equal.
    """
    if depth > 6:
        return "…"
    if isinstance(value, dict):
        items = sorted(
            f"{canonical(key, depth + 1, approximate)}: {canonical(item, depth + 1, approximate)}" for key, item in value.items()
        )
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (set, frozenset)):
        items = sorted(canonical(item, depth + 1, approximate) for item in value)
        return f"{type(value).__name__}({{{', '.join(items)}}})"
    if isinstance(value, list):
        return "[" + ", ".join(canonical(item, depth + 1, approximate) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(canonical(item, depth + 1, approximate) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    value_type = type(value)
    if approximate and value_type is float and math.isfinite(value):  # type: ignore[arg-type]
        return f"float({value:.12g})"
    if value_type.__repr__ is object.__repr__ and hasattr(value, "__dict__"):
        return f"{value_type.__qualname__}({canonical(vars(value), depth + 1, approximate)})"
    try:
        text = repr(value)
    except Exception as error:  # noqa: BLE001 - target code may define broken __repr__
        text = f"<unrepresentable {value_type.__qualname__}: {type(error).__name__}>"
    return _ADDRESS.sub(" at 0x…", text)


def comparison_key(value: object) -> str:
    """Full-length digest of the approximate canonical form (display reprs are truncated)."""
    return hashlib.sha256(canonical(value, approximate=True).encode("utf-8", "surrogatepass")).hexdigest()[:24]


def literal_source(value: object) -> str | None:
    """``repr(value)`` if it round-trips through ``ast.literal_eval``, else ``None``."""
    if not _is_literal(value, 0):
        return None
    text = repr(value)
    return text if len(text) <= _MAX_REPR else None


def _is_literal(value: object, depth: int) -> bool:
    if depth > 8:
        return False
    if value is None or type(value) in (bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (list, tuple, set):
        return all(_is_literal(item, depth + 1) for item in value)  # type: ignore[union-attr]
    if type(value) is dict:
        return all(_is_literal(key, depth + 1) and _is_literal(item, depth + 1) for key, item in value.items())  # type: ignore[union-attr]
    return False


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_REPR else text[: _MAX_REPR - 1] + "…"


def _exception_name(error: BaseException) -> str:
    module = type(error).__module__
    name = type(error).__qualname__
    return name if module == "builtins" else f"{module}.{name}"


class SetupError(Exception):
    """CounterPatch could not prepare the call (not a behavior of the target code)."""

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


def _add_roots(request: dict) -> None:
    for root in reversed(request.get("roots", [])):
        if root not in sys.path:
            sys.path.insert(0, root)


def _load_module(name: str, expected_file: str) -> object:
    """Import ``name`` and verify it is exactly ``expected_file``.

    The check also runs on cache hits: two files that both import as ``utils`` (from
    different roots) must never be confused with each other.
    """
    module = _modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except BaseException as error:  # noqa: BLE001 - any import-time failure, including SystemExit
            raise SetupError("import", f"importing {name} failed: {type(error).__name__}: {error}") from error
        _modules[name] = module
    location = os.path.realpath(getattr(module, "__file__", "") or "")
    if location != os.path.realpath(expected_file):
        raise SetupError("resolve", f"module {name!r} resolved to {location or 'a non-file module'}, expected {expected_file}")
    return module


def _resolve(request: dict) -> tuple[object, object | None]:
    module = _load_module(request["module"], request["file"])
    parts = request["qualname"].split(".")
    owner: object = module
    try:
        for part in parts[:-1]:
            owner = getattr(owner, part)
        target = getattr(owner, parts[-1])
    except AttributeError as error:
        raise SetupError("resolve", f"{request['qualname']} not found: {error}") from error
    instance = None
    if request["kind"] == "instance":
        try:
            instance = owner()  # type: ignore[operator]
        except Exception as error:  # noqa: BLE001
            raise SetupError(
                "construct", f"could not construct {'.'.join(parts[:-1])}() without arguments: {type(error).__name__}: {error}"
            ) from error
        return getattr(instance, parts[-1]), instance
    return target, None


def prepare(request: dict) -> dict:
    """Import the target module; run with its own (generous) timeout, before any timed call."""
    _add_roots(request)
    try:
        _load_module(request["module"], request["file"])
    except SetupError as error:
        return {"status": "error", "phase": error.phase, "detail": str(error)}
    return {"status": "ready"}


def handle(request: dict) -> dict:
    _add_roots(request)
    try:
        function, instance = _resolve(request)
        positional, keywords = [], {}
        for name, source, is_positional in request["args"]:
            value = eval(source, dict(_EVAL_NAMES))  # noqa: S307 - sources are CounterPatch-generated literals
            if is_positional:
                positional.append(value)
            else:
                keywords[name] = value
    except SetupError as error:
        return {"status": "error", "phase": error.phase, "detail": str(error)}
    except BaseException as error:  # noqa: BLE001
        return {"status": "error", "phase": "resolve", "detail": f"{type(error).__name__}: {error}"}

    random.seed(0)
    captured_out, captured_err = io.StringIO(), io.StringIO()
    response: dict = {}
    with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
        try:
            result = function(*positional, **keywords)  # type: ignore[operator]
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
            elif inspect.isgenerator(result):
                result = list(itertools.islice(result, 1000))
            response = {
                "status": "returned",
                "return_value_repr": _clip(canonical(result)),
                "return_value_key": comparison_key(result),
                "return_value_source": literal_source(result),
            }
        except BaseException as error:  # noqa: BLE001 - SystemExit and friends are behavior too
            frames = traceback.format_exception(type(error), error, error.__traceback__)
            response = {
                "status": "raised",
                "exception_type": _exception_name(error),
                "exception_message": _clip(str(error)),
                "detail": _clip("".join(frames[-3:])),
            }
    response["stdout"] = _clip(captured_out.getvalue())
    response["stderr"] = _clip(captured_err.getvalue())
    values = dict(zip([name for name, _, positional_flag in request["args"] if positional_flag], positional))
    values.update(keywords)
    mutable = {name: value for name, value in values.items() if isinstance(value, (list, dict, set))}
    response["args_after"] = {name: _clip(canonical(value)) for name, value in mutable.items()}
    response["args_after_key"] = comparison_key(mutable)
    response["args_after_source"] = {name: source for name, value in mutable.items() if (source := literal_source(value))}
    if instance is not None and hasattr(instance, "__dict__"):
        response["state_after"] = _clip(canonical(vars(instance)))
        response["state_after_key"] = comparison_key(vars(instance))
        response["state_after_source"] = literal_source(vars(instance))
    return response


def main() -> None:
    own_directory = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or ".") != own_directory]
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    requests = os.fdopen(os.dup(0), "r", encoding="utf-8")
    os.dup2(2, 1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115 - lives for the whole process
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), encoding="utf-8", line_buffering=True)
    sys.dont_write_bytecode = True
    os.chdir(sys.argv[1])
    for line in requests:
        if not line.strip():
            continue
        request = json.loads(line)
        response = prepare(request) if request.get("op") == "prepare" else handle(request)
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
