#!/usr/bin/env python3
"""Time the Music Guide CLI on synthetic files and report wall time and peak resident memory.

    benchmark.py [--six-hour] [--detail summary|full] [--timeout SECONDS]

The 4 minute stereo file always runs. The 6 hour mono file at 8 kHz takes minutes and writes about 350 MB to a
temporary directory, so it runs only with --six-hour. Each file is analyzed by a fresh interpreter, and the child's
own peak resident set comes from `wait4`, so one case never inherits another's peak. That per-child peak was verified
on macOS only, where `ru_maxrss` is in bytes. The Linux branch assumes kilobytes and has not been run.

A child that outlives --timeout (600 s by default) is killed with its process group and reported as a failure, and so
is the child of an interrupted run. The stereo file is truly stereo: the two channels carry the same kick and chords
with different phases and independent deterministic noise, so a decoder that folds a mono file to stereo cannot pass
for it.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "music-guide" / "scripts" / "analyze.py"
BEATS_PER_SECOND = 2.0
CHORD_SECONDS = 2.0
CHORDS = ((146.83, 164.81, 220.0), (174.61, 220.0, 261.63), (233.08, 293.66, 349.23), (220.0, 277.18, 329.63))
BYTES_PER_MB = 1 << 20
DEFAULT_TIMEOUT = 600.0
POLL_SECONDS = 0.05
NOISE_LEVEL = 0.03
CHANNEL_PHASE = 2.0


class Case(NamedTuple):
    label: str
    seconds: float
    rate: int
    channels: int
    max_seconds: float | None
    max_mb: float


class Measurement(NamedTuple):
    returncode: int
    seconds: float
    peak_mb: float
    stderr: str = ""
    timed_out: bool = False
    stdout: str = ""


CASES = (Case("4 minute stereo 44.1 kHz", 240.0, 44100, 2, 15.0, 400.0),
         Case("6 hour mono 8 kHz", 6 * 3600.0, 8000, 1, None, 1024.0))


def select_cases(*, six_hour: bool) -> tuple[Case, ...]:
    return CASES if six_hour else CASES[:1]


def block(start: int, frames: int, rate: int, channel: int = 0, channels: int = 1) -> np.ndarray:
    """A 120 BPM kick and a four-chord pad, computed from the absolute sample index so block size never matters.

    With more than one channel, each channel shifts the pad's phase and adds its own noise, so they decorrelate."""
    index = start + np.arange(frames)
    t = index / rate
    since_beat = t % (1 / BEATS_PER_SECOND)
    kick = np.sin(2 * np.pi * 60 * since_beat) * np.exp(-since_beat / 0.05)
    chord = (t // CHORD_SECONDS).astype(np.int64) % len(CHORDS)
    pad = np.zeros(frames)
    for number, pitches in enumerate(CHORDS):
        mask = chord == number
        for pitch in pitches:
            for harmonic in (1, 2, 3):
                pad[mask] += np.sin(2 * np.pi * pitch * harmonic * t[mask] + channel * CHANNEL_PHASE * harmonic) \
                    / harmonic
    signal_ = 0.5 * kick + 0.08 * pad
    if channels > 1:
        signal_ = signal_ + NOISE_LEVEL * (np.modf(np.sin(index * 12.9898 + channel * 78.233) * 43758.5453)[0])
    return signal_.astype(np.float32)


def synthesize(path: Path, *, seconds: float, rate: int, channels: int, block_seconds: float = 30.0) -> None:
    total, step = round(seconds * rate), max(1, round(block_seconds * rate))
    with sf.SoundFile(str(path), "w", samplerate=rate, channels=channels, subtype="PCM_16") as out:
        for start in range(0, total, step):
            frames = min(step, total - start)
            columns = [block(start, frames, rate, channel, channels) for channel in range(channels)]
            out.write(np.stack(columns, axis=1) if channels > 1 else columns[0])


def kill(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def measure(command: Sequence[str], *, timeout: float | None = DEFAULT_TIMEOUT) -> Measurement:
    """Run `command` in its own process group and report its wall time and peak RSS.

    After `timeout` seconds the group is killed and the result has `timed_out` set. An interrupt kills it too."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        started = time.perf_counter()
        child = subprocess.Popen(list(command), stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    break
                if timeout is not None and time.perf_counter() - started > timeout:
                    timed_out = True
                    kill(child)
                    _, status, usage = os.wait4(child.pid, 0)
                    break
                time.sleep(POLL_SECONDS)
        except BaseException:
            kill(child)
            try:
                child.returncode = os.waitstatus_to_exitcode(os.wait4(child.pid, 0)[1])
            except ChildProcessError:
                child.returncode = -signal.SIGKILL
            raise
        seconds = time.perf_counter() - started
        child.returncode = os.waitstatus_to_exitcode(status)
        err.seek(0)
        stderr = err.read().decode(errors="replace")
        out.seek(0)
        stdout = out.read().decode(errors="replace")
    peak_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    return Measurement(child.returncode, seconds, peak_bytes / BYTES_PER_MB, stderr, timed_out, stdout)


def verdict(case: Case, result: Measurement) -> str:
    if result.timed_out:
        return f"failed: timed out after {result.seconds:.0f} s and was killed"
    if result.returncode != 0:
        return f"failed: exit status {result.returncode}"
    over = []
    if case.max_seconds is not None and result.seconds > case.max_seconds:
        over.append(f"time {result.seconds:.1f} s over {case.max_seconds:g} s")
    if result.peak_mb > case.max_mb:
        over.append(f"memory {result.peak_mb:.0f} MB over {case.max_mb:g} MB")
    return "over budget: " + ", ".join(over) if over else "within budget"


def report_errors(output: str) -> list[str]:
    """Verify that a timed CLI run actually produced all supported analyzer families."""
    try:
        report = json.loads(output)
        properties = {item["property"] for item in report["observations"]}
        errors = [item["message"] for item in report["diagnostics"] if item["code"] == "estimate_failed"]
        required = {"audio.duration", "audio.integrated_loudness", "audio.beat_grid", "audio.chord_timeline",
                    "audio.rhythm", "audio.structure", "audio.energy_timeline", "audio.mood_estimate", "audio.melody"}
        errors.extend(f"missing {name}" for name in sorted(required - properties))
        return errors
    except (ValueError, TypeError, KeyError) as error:
        return [f"invalid evidence output: {error}"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--six-hour", action="store_true", help="also run the slow 6 hour mono file")
    parser.add_argument("--detail", choices=("summary", "full"), default="summary")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="seconds one analysis may run before it is killed and reported as failed (default 600)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="music-guide-bench-") as directory:
        for case in select_cases(six_hour=args.six_hour):
            wav = Path(directory) / "bench.wav"
            synthesize(wav, seconds=case.seconds, rate=case.rate, channels=case.channels)
            result = measure([sys.executable, str(SCRIPT), "--mixdown", str(wav), "--detail", args.detail],
                             timeout=args.timeout)
            wav.unlink()
            budget = (f"{case.max_seconds:g} s, " if case.max_seconds is not None else "no time budget, ") \
                + f"{case.max_mb:g} MB"
            print(f"{case.label}: {result.seconds:.1f} s wall, {result.peak_mb:.0f} MB peak RSS, "
                  f"{verdict(case, result)} (budget {budget}, detail {args.detail})")
            if result.returncode != 0:
                print(result.stderr.strip()[-2000:], file=sys.stderr)
            errors = report_errors(result.stdout) if result.returncode == 0 else []
            for error in errors:
                print(f"analysis error: {error}", file=sys.stderr)
            failures += verdict(case, result) != "within budget" or bool(errors)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
