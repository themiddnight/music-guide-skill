#!/usr/bin/env python3
"""Score a Music Guide JSON report against ground truth.

    evaluate.py REPORT.json TRUTH.json [--markdown]

The script reads only the CLI's JSON, never audio. It scores each truth key it finds and says `not reported` for an
estimate the report lacks, which counts as zero in the overall mean. Beat and downbeat scoring needs a report made with
`--detail full`, because the summary carries no beat times.

Truth keys, all optional:
    tempoBpm, meter ("4/4"), beats[], downbeats[],
    keys[{startSeconds, endSeconds, tonic, mode}],
    chords[{startSeconds, endSeconds, label}] with Harte-style labels such as D:sus2, A#:maj, A:7,
    firstChord, melody[{startSeconds, endSeconds, midi}], structure[{startSeconds, letter}].
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

BEAT_WINDOW = 0.07
TEMPO_TOLERANCE = 0.04
MELODY_TOLERANCE_MIDI = 0.5
MELODY_STEP = 0.01
BOUNDARY_WINDOW = 3.0
FIXED_UNIT_SECONDS = 0.5
OCTAVE_FACTORS = (1.0, 2.0, 0.5, 3.0, 1 / 3)

SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FAMILY = {"maj": "major", "maj7": "major", "7": "major", "min": "minor", "min7": "minor", "dim": "minor",
          "sus2": "suspended", "sus4": "suspended"}


class Metric(NamedTuple):
    name: str
    score: float | None
    detail: str


Interval = tuple[float, float, str, tuple[str, ...]]


def missing(name: str, why: str) -> Metric:
    return Metric(name, None, f"not reported: {why}")


def overall(metrics: Sequence[Metric]) -> float:
    return sum(m.score or 0.0 for m in metrics) / len(metrics) if metrics else 0.0


class Observations:
    """Report values by property. A mixdown estimate wins over a stem's, and a stem is used only when alone."""

    def __init__(self, report: dict):
        self.values: dict[str, object] = {}
        from_mixdown: set[str] = set()
        for observation in report.get("observations", []):
            prop, mixdown = observation.get("property"), observation.get("scope") == "mixdown"
            if prop not in self.values or (mixdown and prop not in from_mixdown):
                self.values[prop] = observation.get("value")
                if mixdown:
                    from_mixdown.add(prop)

    def get(self, prop: str) -> dict | None:
        value = self.values.get(prop)
        return value if isinstance(value, dict) else None


def normalise_tonic(tonic: str) -> str:
    return SHARPS[pitch_class(tonic)]


def pitch_class(name: str) -> int:
    value = PITCH_CLASS[name[0].upper()]
    for accidental in name[1:]:
        value += {"#": 1, "b": -1, "♯": 1, "♭": -1}[accidental]
    return value % 12


def parse_chord(label: str) -> tuple[str | None, str | None]:
    """Harte-style `root:quality[/bass]` as (root, quality). `N` is (None, None). A bare root is major."""
    label = label.split("/")[0].strip()
    if label in ("N", "", "none"):
        return None, None
    root, _, quality = label.partition(":")
    return normalise_tonic(root), quality or "maj"


def chord_family(label: str) -> str:
    root, quality = parse_chord(label)
    if root is None:
        return "none"
    if quality in FAMILY:
        return FAMILY[quality]
    return "minor" if quality.startswith("min") else "major" if quality.startswith("maj") or quality[0].isdigit() \
        else "other"


def sweep_overlap(truth: Sequence[Interval], estimate: Sequence[Interval],
                  agrees: Callable[[Interval, Interval], bool]) -> float:
    """Seconds of truth time during which `agrees(truth_interval, estimate_interval)` holds."""
    total, j = 0.0, 0
    for t in truth:
        while j < len(estimate) and estimate[j][1] <= t[0]:
            j += 1
        k = j
        while k < len(estimate) and estimate[k][0] < t[1]:
            overlap = min(t[1], estimate[k][1]) - max(t[0], estimate[k][0])
            if overlap > 0 and agrees(t, estimate[k]):
                total += overlap
            k += 1
    return total


def f_measure(truth: Sequence[float], estimate: Sequence[float], window: float) -> tuple[float, int]:
    """Greedy one-to-one match in time order. Returns (F, matched)."""
    if not truth and not estimate:
        return 1.0, 0
    estimate = sorted(estimate)
    used, matched = set(), 0
    for t in sorted(truth):
        i = bisect.bisect_left(estimate, t)
        candidates = [c for c in (i - 1, i) if 0 <= c < len(estimate) and c not in used
                      and abs(estimate[c] - t) <= window]
        if candidates:
            used.add(min(candidates, key=lambda c: abs(estimate[c] - t)))
            matched += 1
    return 2 * matched / (len(truth) + len(estimate)), matched


def tempo_metrics(obs: Observations, truth_bpm: float) -> list[Metric]:
    sources = {"beat_grid": ("audio.beat_grid", lambda v: v.get("bpm")),
               "tempo_estimate": ("audio.tempo_estimate",
                                  lambda v: (v.get("candidates") or [{}])[0].get("bpm"))}
    metrics = []
    for source, (prop, pick) in sources.items():
        strict, relaxed = f"tempo accuracy ({source})", f"tempo accuracy, octave-aware ({source})"
        value = obs.get(prop)
        bpm = pick(value) if value else None
        if bpm is None:
            metrics += [missing(strict, f"{prop} absent"), missing(relaxed, f"{prop} absent")]
            continue
        ratio = bpm / truth_bpm
        error = f"{bpm:.1f} BPM against {truth_bpm:g} ({(ratio - 1) * 100:+.1f}%)"

        def near(factor: float) -> bool:
            return abs(ratio / factor - 1) <= TEMPO_TOLERANCE

        metrics.append(Metric(strict, float(near(1.0)), error))
        factor = next((f for f in OCTAVE_FACTORS if near(f)), None)
        metrics.append(Metric(relaxed, float(factor is not None),
                              error + ("" if factor in (None, 1.0) else f", matches at x{factor:g}")))
    return metrics


def candidate_signature(candidate) -> str | None:
    if isinstance(candidate, dict):
        return candidate.get("signature") or candidate.get("meter")
    if isinstance(candidate, (list, tuple)) and candidate:
        return candidate[0]
    return candidate if isinstance(candidate, str) else None


def meter_metrics(obs: Observations, truth: dict) -> list[Metric]:
    value = obs.get("audio.meter_estimate")
    metrics = []
    if "meter" in truth:
        candidates = (value or {}).get("candidates") or []
        top = candidate_signature(candidates[0]) if candidates else None
        metrics.append(Metric("meter", float(top == truth["meter"]), f"{top} against {truth['meter']}")
                       if top else missing("meter", "audio.meter_estimate absent"))
    if "downbeats" in truth:
        name = "downbeat F-measure (70 ms)"
        times = (value or {}).get("downbeatTimes")
        if times is None:
            metrics.append(missing(name, "audio.meter_estimate has no downbeatTimes; rerun with --detail full"
                                   if value else "audio.meter_estimate absent"))
        else:
            metrics.append(beat_metric(name, truth["downbeats"], times))
    return metrics


def beat_metric(name: str, truth: Sequence[float], estimate: Sequence[float]) -> Metric:
    score, matched = f_measure(truth, estimate, BEAT_WINDOW)
    return Metric(name, score, f"{matched} of {len(truth)} truth, {len(estimate)} estimated")


def beat_metrics(obs: Observations, truth_beats: Sequence[float]) -> list[Metric]:
    name = "beat F-measure (70 ms)"
    grid = obs.get("audio.beat_grid")
    if grid is None:
        return [missing(name, "audio.beat_grid absent")]
    if grid.get("beatTimes") is None:
        return [missing(name, "audio.beat_grid has no beatTimes; rerun with --detail full")]
    return [beat_metric(name, truth_beats, grid["beatTimes"])]


def key_name(candidate: dict) -> str:
    return f"{normalise_tonic(candidate['tonic'])} {candidate['mode']}"


def key_metrics(obs: Observations, truth_keys: Sequence[dict]) -> list[Metric]:
    name = "key agreement"
    timeline, single = obs.get("audio.key_timeline"), obs.get("audio.key_estimate")
    if timeline and timeline.get("segments"):
        estimate = [(s["startSeconds"], s["endSeconds"], key_name(s["candidates"][0]), ())
                    for s in timeline["segments"] if s.get("candidates")]
    elif single and single.get("candidates"):
        estimate = [(float("-inf"), float("inf"), key_name(single["candidates"][0]), ())]
    else:
        return [missing(name, "neither audio.key_timeline nor audio.key_estimate reported")]
    reference = [(k["startSeconds"], k["endSeconds"], key_name(k), ()) for k in truth_keys]
    total = sum(end - start for start, end, _, _ in reference)
    agreed = sweep_overlap(reference, estimate, lambda t, e: t[2] == e[2])
    names = ", ".join(dict.fromkeys(e[2] for e in estimate))
    return [Metric(name, agreed / total if total else 0.0, f"{agreed:.1f} of {total:.1f} s agree; estimated {names}")]


def chord_intervals(obs: Observations) -> list[Interval] | None:
    """Estimated chords as (start, end, label, alternative labels).

    Full detail's `perUnit` list wins over the capped segments. Its entries are labels or `{startSeconds, chord}`
    objects. Without per-entry times the units start at `unitTimes`, else at the beat times for `basis` beats, else
    every 0.5 s.
    """
    value = obs.get("audio.chord_timeline")
    if value is None:
        return None
    per_unit = value.get("perUnit")
    if per_unit:
        starts = [u.get("startSeconds") if isinstance(u, dict) else None for u in per_unit]
        if any(s is None for s in starts):
            starts = value.get("unitTimes")
            if starts is None and value.get("basis") == "beats":
                starts = (obs.get("audio.beat_grid") or {}).get("beatTimes")
            if starts is None and value.get("basis") != "beats":
                starts = [i * FIXED_UNIT_SECONDS for i in range(len(per_unit))]
        if starts is not None and len(starts) >= len(per_unit):
            step = starts[len(per_unit) - 1] - starts[len(per_unit) - 2] if len(per_unit) > 1 else FIXED_UNIT_SECONDS
            ends = list(starts[1:len(per_unit)]) + [starts[len(per_unit) - 1] + step]
            labels = [u.get("chord", "N") if isinstance(u, dict) else u for u in per_unit]
            return [(s, e, str(c), ()) for s, e, c in zip(starts, ends, labels)]
    return [(s["startSeconds"], s["endSeconds"], s["chord"], tuple(map(str, s.get("alternatives", []))))
            for s in value.get("segments") or []]


def chord_metrics(obs: Observations, truth: dict) -> list[Metric]:
    names = ("chord label overlap", "chord label overlap, with alternatives", "chord root overlap",
             "chord major/minor overlap")
    estimate = chord_intervals(obs)
    if estimate is None:
        wanted = (names if "chords" in truth else ()) + (("first chord",) if "firstChord" in truth else ())
        return [missing(name, "audio.chord_timeline absent") for name in wanted]
    metrics = []
    if "chords" in truth:
        reference = [(c["startSeconds"], c["endSeconds"], c["label"], ()) for c in truth["chords"]]
        total = sum(end - start for start, end, _, _ in reference)
        rules = (lambda t, e: parse_chord(t[2]) == parse_chord(e[2]),
                 lambda t, e: any(parse_chord(t[2]) == parse_chord(label) for label in (e[2], *e[3])),
                 lambda t, e: parse_chord(t[2])[0] == parse_chord(e[2])[0],
                 lambda t, e: chord_family(t[2]) == chord_family(e[2]))
        for name, rule in zip(names, rules):
            seconds = sweep_overlap(reference, estimate, rule)
            metrics.append(Metric(name, seconds / total if total else 0.0, f"{seconds:.1f} of {total:.1f} s"))
    if "firstChord" in truth:
        first = next((label for _, _, label, _ in estimate if parse_chord(label)[0] is not None), None)
        metrics.append(Metric("first chord", float(first is not None and
                                                 parse_chord(first) == parse_chord(truth["firstChord"])),
                              f"{first} against {truth['firstChord']}"))
    return metrics


def melody_pieces(value: dict) -> tuple[list[float], list[float | None], float]:
    contour = sorted(value.get("contour") or [], key=lambda p: p["startSeconds"])
    starts = [p["startSeconds"] for p in contour]
    spacing = statistics.median(b - a for a, b in zip(starts, starts[1:])) if len(starts) > 1 else FIXED_UNIT_SECONDS
    end = value.get("coveredSeconds")
    return starts, [p.get("midi") for p in contour], (end if end is not None and starts and end > starts[-1]
                                                       else (starts[-1] + spacing if starts else 0.0))


def melody_metrics(obs: Observations, truth_notes: Sequence[dict]) -> list[Metric]:
    name = "melody raw pitch accuracy"
    value = obs.get("audio.melody")
    if value is None or not value.get("contour"):
        return [missing(name, "audio.melody absent" if value is None else "audio.melody has no contour")]
    starts, pitches, end = melody_pieces(value)
    voiced = correct = 0
    for note in truth_notes:
        frames = int(round((note["endSeconds"] - note["startSeconds"]) / MELODY_STEP))
        for k in range(frames):
            t = note["startSeconds"] + (k + 0.5) * MELODY_STEP
            voiced += 1
            i = bisect.bisect_right(starts, t) - 1
            estimated = pitches[i] if 0 <= i and t < end else None
            correct += estimated is not None and abs(estimated - note["midi"]) <= MELODY_TOLERANCE_MIDI
    return [Metric(name, correct / voiced if voiced else 0.0, f"{correct} of {voiced} voiced truth frames of 10 ms")]


def structure_metrics(obs: Observations, truth_segments: Sequence[dict]) -> list[Metric]:
    name = "structure boundary hit rate (3 s)"
    boundaries = [s["startSeconds"] for s in sorted(truth_segments, key=lambda s: s["startSeconds"])][1:]
    if not boundaries:
        return []
    value = obs.get("audio.structure")
    if value is None:
        return [missing(name, "audio.structure absent")]
    estimated = sorted(s["startSeconds"] for s in value.get("segments", []))[1:]
    _, matched = f_measure(boundaries, estimated, BOUNDARY_WINDOW)
    precision = f"{matched / len(estimated):.2f}" if estimated else "n/a"
    return [Metric(name, matched / len(boundaries),
                   f"{matched} of {len(boundaries)} truth boundaries; precision {precision}")]


def evaluate(report: dict, truth: dict) -> list[Metric]:
    obs = Observations(report)
    metrics: list[Metric] = []
    if "tempoBpm" in truth:
        metrics += tempo_metrics(obs, truth["tempoBpm"])
    if "beats" in truth:
        metrics += beat_metrics(obs, truth["beats"])
    metrics += meter_metrics(obs, truth)
    if "keys" in truth:
        metrics += key_metrics(obs, truth["keys"])
    if "chords" in truth or "firstChord" in truth:
        metrics += chord_metrics(obs, truth)
    if "melody" in truth:
        metrics += melody_metrics(obs, truth["melody"])
    if "structure" in truth:
        metrics += structure_metrics(obs, truth["structure"])
    return metrics


def render(metrics: Sequence[Metric], markdown: bool) -> str:
    def cell(m: Metric) -> str:
        return "n/a" if m.score is None else f"{m.score:.3f}"

    mean = f"{overall(metrics):.3f}"
    unreported = sum(m.score is None for m in metrics)
    note = f"{unreported} not reported, counted as 0" if unreported else "all reported"
    if markdown:
        rows = ["| metric | score | detail |", "|---|---|---|"]
        rows += [f"| {m.name} | {cell(m)} | {m.detail} |" for m in metrics]
        rows.append(f"| **overall** | **{mean}** | {note} |")
        return "\n".join(rows)
    width = max((len(m.name) for m in metrics), default=0)
    lines = [f"{m.name:<{width}}  {cell(m):>5}  {m.detail}" for m in metrics]
    lines.append(f"{'overall':<{width}}  {mean:>5}  {note}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a Music Guide report against ground truth.")
    parser.add_argument("report", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)
    print(render(evaluate(json.loads(args.report.read_text()), json.loads(args.truth.read_text())), args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
