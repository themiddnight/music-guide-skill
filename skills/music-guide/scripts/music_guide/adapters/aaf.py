"""Binary AAF (Advanced Authoring Format) composition metadata, read by pyaaf2 in an isolated child process.

pyaaf2 sizes buffers from counts inside the file, so a corrupt index can demand gigabytes. The reading therefore runs in
`aaf_child`, a separate interpreter that this module stops when it exceeds a wall-clock or memory limit. The child also
bounds itself (an alarm and, on Linux, an address-space limit), so it ends even if this process is interrupted. This
module does not import pyaaf2: it is needed only when an AAF file is analyzed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..model import Detail, Diagnostic, Evidence

# Exit codes the child uses: the deadline passed (the alarm signal, or this code where there is no alarm) or the
# address-space limit was hit.
DEADLINE_EXIT = 3
MEMORY_EXIT = 4
CHILD_TIMEOUT_SECONDS = 60.0
CHILD_MEMORY_BYTES = 1536 * 1024 * 1024
CHILD_OUTPUT_BYTES = 32 * 1024 * 1024
CHILD_GRACE_SECONDS = 5.0
STDERR_BYTES = 64 * 1024
POLL_SECONDS = 0.1
MEMORY_POLL_SECONDS = 0.2
MAX_PATH_ENTRIES = 64
MAX_PATH_CHARS = 4096
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
# -I keeps the working directory, PYTHONPATH and user site out of the child. The parent's own import path is passed
# through stdin instead, so packages installed for this interpreter (including `pip install --user`) stay importable.
CHILD_CODE = (
    "import json, sys; config = json.loads(sys.stdin.buffer.read(1 << 20)); sys.path[:0] = config['paths']; "
    "from music_guide.adapters.aaf_child import main; raise SystemExit(main(config))")


def child_sys_path() -> list[str]:
    """The parent's import path without the working directory, empty entries and relative entries."""
    here = os.path.realpath(os.getcwd())
    chosen: list[str] = [str(SCRIPTS_DIR)]
    for entry in sys.path:
        if (isinstance(entry, str) and entry and len(entry) <= MAX_PATH_CHARS and os.path.isabs(entry)
                and os.path.realpath(entry) != here and entry not in chosen):
            chosen.append(entry)
    return chosen[:MAX_PATH_ENTRIES]


def _memory_poller() -> str | None:
    """The `ps` executable used to poll the child's resident memory, or None where there is none (Windows, minimal containers)."""
    return shutil.which("ps")


def _resident_bytes(ps: str, pid: int) -> int | None:
    """Resident memory of a process, or None when it cannot be read (the process is gone)."""
    try:
        output = subprocess.run([ps, "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return int(output.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class _Reader(threading.Thread):
    """Drains one child pipe without holding more than `limit` bytes.

    Past the limit it either stops (`drain` False, for output the caller rejects) or keeps reading and discards, so a
    chatty child never blocks on a full pipe.
    """

    def __init__(self, stream: Any, limit: int, drain: bool = False) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.limit = limit
        self.drain = drain
        self.data = bytearray()
        self.overflow = False

    def run(self) -> None:
        while chunk := self.stream.read(65536):
            if len(self.data) + len(chunk) > self.limit:
                self.overflow = True
                self.data += chunk[:max(0, self.limit - len(self.data))]
                if not self.drain:
                    return
                continue
            self.data += chunk


def _stopped(reason: str) -> ValueError:
    return ValueError(f"AAF parser stopped: reading the file {reason}; the file is corrupt or unsupported")


def _run_child(path: Path, source_id: str, detail: Detail, timeout: float, memory: int, ps: str | None) -> dict[str, Any]:
    config = json.dumps({"paths": child_sys_path(), "path": str(path), "sourceId": source_id, "detail": detail,
                         "deadline": timeout + CHILD_GRACE_SECONDS, "memory": memory}).encode()
    command = [sys.executable, "-I", "-c", CHILD_CODE]
    started = time.monotonic()
    out = err = None
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        try:
            try:
                child.stdin.write(config)
                child.stdin.close()
            except OSError:
                pass  # the child already ended; its exit status is reported below
            out, err = _Reader(child.stdout, CHILD_OUTPUT_BYTES), _Reader(child.stderr, STDERR_BYTES, drain=True)
            out.start()
            err.start()
            next_memory_check = started
            while True:
                try:
                    child.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if out.overflow:
                    raise _stopped(f"produced more than {CHILD_OUTPUT_BYTES // (1024 * 1024)} MiB of output")
                if now - started > timeout:
                    raise _stopped(f"took longer than {timeout:g} seconds")
                if ps is not None and now >= next_memory_check:
                    next_memory_check = now + MEMORY_POLL_SECONDS
                    resident = _resident_bytes(ps, child.pid)
                    if resident is not None and resident > memory:
                        raise _stopped(f"needed more than {memory // (1024 * 1024)} MiB of memory")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            # Leaving the Popen block closes the pipes, so the drain threads must finish reading first.
            for reader in (out, err):
                if reader is not None:
                    reader.join(5)
    code = child.returncode
    if code == MEMORY_EXIT:
        raise _stopped(f"needed more than {memory // (1024 * 1024)} MiB of memory")
    if code in (DEADLINE_EXIT, -signal.SIGALRM if hasattr(signal, "SIGALRM") else DEADLINE_EXIT):
        raise _stopped(f"took longer than {timeout:g} seconds")
    if code != 0 or out.overflow:
        tail = bytes(err.data).decode("utf-8", "replace").strip().splitlines()[-1:] or ["no message"]
        raise ValueError(f"AAF parser exited abnormally (status {code}): {tail[0][:200]}")
    try:
        reply = json.loads(bytes(out.data))
    except ValueError as error:
        raise ValueError("AAF parser returned unreadable output") from error
    if not isinstance(reply, dict):
        raise ValueError("AAF parser returned unreadable output")
    return reply


def aaf(path: Path, source_id: str, evidence: Evidence, detail: Detail, *,
        timeout_seconds: float | None = None, memory_bytes: int | None = None) -> None:
    ps = _memory_poller()
    reply = _run_child(path, source_id, detail, CHILD_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
                       CHILD_MEMORY_BYTES if memory_bytes is None else memory_bytes, ps)
    if not reply.get("ok"):
        raise ValueError(str(reply.get("error", "unknown AAF failure"))[:500])
    for item in reply["observations"]:
        evidence.add(source_id, item["property"], item["value"], item["unit"], item["scope"], item["timeDomain"],
                     item["locator"], item["method"], item["provenance"], item["confidence"],
                     item["confidenceReason"])
    for item in reply["diagnostics"]:
        evidence.diagnostics.append(Diagnostic(item["code"], item["severity"], item["message"], source_id))
    if reply.get("memoryGuard") == "none" and ps is None:
        evidence.diagnostics.append(Diagnostic(
            "aaf_memory_cap_unavailable", "info",
            "No memory limit could be applied to the AAF parser on this system (no address-space limit and no `ps` "
            "to poll), so only the time limit protected the read.", source_id))
