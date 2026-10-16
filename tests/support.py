"""Shared CLI runner for the test suite.

Most tests only care about what `analyze.py` prints, so `run_cli` calls `music_guide.cli.main` in this process. A
fresh interpreter costs 0.2 to 1.1 s of startup and import work per call, which dominated the suite. Tests whose
subject is the process boundary (isolated mode, missing modules, the interpreter version check, argparse exit
behaviour at the real entry point, the installed copy, the AAF child) call `run_subprocess` instead.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "music-guide"
SCRIPT = SKILL / "scripts" / "analyze.py"
sys.path.insert(0, str(SKILL / "scripts"))

from music_guide.cli import main  # noqa: E402


def run_subprocess(*args, cwd: Path, script: Path | None = None, env: dict | None = None,
                   timeout: float | None = None, python: str | None = None, flags: tuple[str, ...] = ()
                   ) -> subprocess.CompletedProcess:
    """Run `analyze.py` in a real interpreter, with the process boundary intact."""
    return subprocess.run([python or sys.executable, *flags, str(script or SCRIPT), *map(str, args)], cwd=cwd,
                          text=True, capture_output=True, env=env, timeout=timeout)


def run_cli(*args, cwd: Path, script: Path | None = None, env: dict | None = None,
            timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run the CLI in this process and return what the process would have returned.

    The return value mirrors `subprocess.run(..., text=True, capture_output=True)`. Warnings are errors, as they are
    under `PYTHONWARNINGS=error`. An exception that escapes `main` becomes a traceback on stderr and exit status 1,
    which is what the interpreter does, so a crash fails the same assertions it failed before. A custom `script` or
    `env` names a different process, so those calls fall through to a real subprocess.
    """
    if script is not None or env is not None:
        return run_subprocess(*args, cwd=cwd, script=script, env=env, timeout=timeout)
    out = io.StringIO()
    previous_cwd, previous_argv = os.getcwd(), sys.argv
    returncode = 0
    os.chdir(cwd)
    sys.argv = ["analyze.py"]
    try:
        with _capture_stderr() as captured, warnings.catch_warnings(), contextlib.redirect_stdout(out):
            warnings.simplefilter("error")
            try:
                returncode = main([str(arg) for arg in args])
            except SystemExit as stop:
                returncode = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
            except Exception:
                traceback.print_exc()
                returncode = 1
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    return subprocess.CompletedProcess([str(SCRIPT), *map(str, args)], returncode, out.getvalue(), captured.text)


class _Captured:
    text = ""


@contextlib.contextmanager
def _capture_stderr():
    """Collect everything written to stderr, in order, whether Python code or a C library (libsndfile, mpg123) wrote it.

    Descriptor 2 points at a scratch file for the duration, which is what a pipe to a child process would give, and
    `sys.stderr` writes through to the same file so the two sources keep their real order.
    """
    captured = _Captured()
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile("w+b") as sink:
        os.dup2(sink.fileno(), 2)
        python_side = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), encoding="utf-8", write_through=True)
        try:
            with contextlib.redirect_stderr(python_side):
                yield captured
        finally:
            python_side.flush()
            os.dup2(saved, 2)
            os.close(saved)
            sink.seek(0)
            captured.text = sink.read().decode(errors="replace")


def assert_matches_subprocess(case, *args, cwd: Path) -> subprocess.CompletedProcess:
    """End-to-end smoke check: the real entry point and the in-process runner must agree on every output byte."""
    direct = run_subprocess(*args, cwd=cwd, timeout=120)
    inline = run_cli(*args, cwd=cwd)
    case.assertEqual((inline.returncode, inline.stdout, inline.stderr), (direct.returncode, direct.stdout, direct.stderr))
    return direct
