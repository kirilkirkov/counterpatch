"""Command-line interface."""

from __future__ import annotations

import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer

from counterpatch import __version__
from counterpatch.ai import DEFAULT_MODEL
from counterpatch.check import CheckOptions, run_check
from counterpatch.git import repo_root
from counterpatch.intent import load_task
from counterpatch.models import CheckResult, CounterPatchError
from counterpatch.reporter import github_annotations, render_json, render_markdown, render_text
from counterpatch.reproduction import COUNTERPATCH_DIR, ensure_output_dir

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CounterPatch attacks your patch before users do: it executes adversarial inputs against the base "
    "and patched revisions and reports small, reproducible behavioral regressions.",
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"counterpatch {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit."),
    ] = None,
) -> None:
    """Adversarial regression testing for code patches."""


@app.command(epilog="Exit codes: 0 = no regression counterexample found, 1 = regression candidate found, 2 = error.")
def check(
    base: Annotated[
        str | None,
        typer.Option(help="Base revision to compare against (branch, tag or SHA). Defaults to origin/main, main, origin/master or master."),
    ] = None,
    task: Annotated[str | None, typer.Option(help="Description of the intended change (e.g. the PR title/body).")] = None,
    task_file: Annotated[Path | None, typer.Option("--task-file", help="Read the intended-change description from a file.")] = None,
    ai: Annotated[
        bool, typer.Option("--ai", help="Also ask the Anthropic API for adversarial candidates (needs ANTHROPIC_API_KEY).")
    ] = False,
    ai_model: Annotated[
        str, typer.Option("--ai-model", envvar="COUNTERPATCH_AI_MODEL", help="Anthropic model used with --ai.")
    ] = DEFAULT_MODEL,
    max_tests: Annotated[
        int, typer.Option("--max-tests", min=1, help="Candidate budget per changed function (deterministic sweep + Hypothesis).")
    ] = 100,
    timeout: Annotated[float, typer.Option(min=1, help="Overall time budget in seconds; exceeding it exits with code 2.")] = 600.0,
    call_timeout: Annotated[
        float, typer.Option("--call-timeout", min=0.1, help="Seconds a single call may run before it counts as a hang.")
    ] = 5.0,
    python: Annotated[
        str | None, typer.Option(help="Python interpreter used to run the target code. Defaults to the one running CounterPatch.")
    ] = None,
    fail_on_divergence: Annotated[
        bool, typer.Option("--fail-on-divergence", help="Also exit 1 for behavioral divergences, not only regression candidates.")
    ] = False,
    repo: Annotated[Path, typer.Option(help="Path inside the Git repository to check.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print progress and extra details.")] = False,
) -> None:
    """Find behavioral regressions between the base revision and the current patch."""

    def log(message: str) -> None:
        if verbose:
            typer.secho(f"· {message}", fg=typer.colors.BRIGHT_BLACK, err=True)

    signal.signal(signal.SIGTERM, _terminate)
    try:
        options = CheckOptions(
            repo=repo,
            base=base,
            task=load_task(task, task_file),
            ai=ai,
            ai_model=ai_model,
            max_tests=max_tests,
            timeout=timeout,
            call_timeout=call_timeout,
            python=python or sys.executable,
            fail_on_divergence=fail_on_divergence,
        )
        result = run_check(options, log=log)
    except CounterPatchError as error:
        typer.secho(f"counterpatch: error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except KeyboardInterrupt:
        typer.secho("counterpatch: interrupted", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except Exception as error:  # noqa: BLE001 - any bug must still map to exit code 2
        typer.secho(f"counterpatch: internal error: {type(error).__name__}: {error}", fg=typer.colors.RED, err=True)
        if verbose:
            traceback.print_exc()
        raise typer.Exit(EXIT_ERROR) from None

    _emit(result, verbose)
    _write_outputs(result, repo)
    raise typer.Exit(result.exit_code)


def _terminate(signum: int, frame: object) -> None:
    """Turn SIGTERM (e.g. a cancelled CI job) into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt


def _emit(result: CheckResult, verbose: bool) -> None:
    colors = {"✗": typer.colors.RED, "⚠": typer.colors.YELLOW, "✓": typer.colors.GREEN}
    for line in render_text(result, verbose).splitlines():
        color = colors.get(line[:1])
        typer.secho(line, fg=color, bold=bool(color))
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for annotation in github_annotations(result):
            typer.echo(annotation)


def _write_outputs(result: CheckResult, repo: Path) -> None:
    try:
        root = repo_root(repo.resolve())
        ensure_output_dir(root, COUNTERPATCH_DIR)
        (root / COUNTERPATCH_DIR / "report.json").write_text(render_json(result), encoding="utf-8")
    except (CounterPatchError, OSError):
        pass
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(render_markdown(result))
        except OSError:
            pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
