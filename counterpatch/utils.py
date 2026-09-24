"""Small helpers: safe paths, subprocess environments, deadlines."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from counterpatch.models import CheckTimeout, ConfigError

_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-z0-9_]+")
_SECRET_ENV_NAME = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY)", re.IGNORECASE)
_MAX_FILENAME_STEM = 80


def sanitize_filename(name: str, prefix: str = "test_", suffix: str = ".py") -> str:
    """Turn arbitrary text (possibly LLM output) into a safe, flat Python test filename.

    The result only contains ``[a-z0-9_]``, never contains path separators or ``..``,
    always starts with ``prefix`` and ends with ``suffix``.
    """
    stem = re.sub("_+", "_", _UNSAFE_FILENAME_CHARS.sub("_", name.lower())).strip("_")
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    stem = stem.strip("_") or "case"
    stem = (prefix + stem)[:_MAX_FILENAME_STEM].rstrip("_")
    return stem + suffix


def safe_join(base_dir: Path, filename: str) -> Path:
    """Join ``filename`` onto ``base_dir`` and refuse anything that escapes it."""
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."} or "\x00" in filename:
        raise ConfigError(f"Refusing unsafe output filename: {filename!r}")
    base = base_dir.resolve()
    candidate = (base / filename).resolve()
    if candidate.parent != base:
        raise ConfigError(f"Refusing to write outside {base}: {filename!r}")
    return candidate


def subprocess_env(extra_pythonpath: list[Path] | None = None) -> dict[str, str]:
    """Environment for running target code.

    Secret-looking variables (API keys, tokens, passwords) are removed so that code
    under test cannot trivially read CounterPatch's own credentials. This is defense in
    depth, not a sandbox.
    """
    env = {key: value for key, value in os.environ.items() if not _SECRET_ENV_NAME.search(key) and not key.startswith("ACTIONS_")}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONSTARTUP", None)
    if extra_pythonpath:
        paths = [str(path) for path in extra_pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def module_for_path(root: Path, relative_path: str) -> tuple[str, str]:
    """Return ``(import_root, dotted_module)`` for a ``.py`` path relative to ``root``.

    The import root is the first ancestor directory that is not a regular package
    (has no ``__init__.py``), which covers flat layouts, ``src/`` layouts and nested
    packages. ``import_root`` is relative to ``root`` (``"."`` for ``root`` itself).
    """
    path = Path(relative_path)
    parts = [path.stem] if path.stem != "__init__" else []
    directory = path.parent
    while directory != Path(".") and (root / directory / "__init__.py").exists():
        parts.insert(0, directory.name)
        directory = directory.parent
    return (directory.as_posix() if directory != Path(".") else "."), ".".join(parts)


class Deadline:
    """Overall wall-clock budget for a check."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.expires_at = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def check(self) -> None:
        if self.expired():
            raise CheckTimeout(f"CounterPatch exceeded its --timeout budget of {self.seconds:g}s")

    def clamp(self, seconds: float) -> float:
        """Return ``seconds`` limited to what is left of the budget (at least a small epsilon)."""
        return max(0.05, min(seconds, self.remaining()))
