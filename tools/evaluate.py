#!/usr/bin/env python3
"""Score a Music Guide JSON report against ground truth.

    evaluate.py REPORT.json TRUTH.json [--markdown]

The script reads only the CLI's JSON, never audio. It scores each truth key it finds and says `not reported` for an
estimate the report lacks, which counts as zero in its family's mean. Beat and downbeat scoring needs a report made
with `--detail full`, because the summary carries no beat times. A metric that cannot apply to this truth (a structure
truth with one segment has no boundaries) says `n/a` and is left out of every mean.

The last row is the mean of the per-family means over the families the truth covers: tempo, beats, meter (the meter
and the downbeats), key, chords (labels, root, major/minor, first chord), melody and structure. Each family counts
once, however many metrics it holds. Two runs are comparable only when they cover the same families, so a value from
a phase that scores more families than an earlier phase is not comparable with the earlier one.

Truth keys, all optional:
    tempoBpm, meter ("4/4"), beats[], downbeats[],
    keys[{startSeconds, endSeconds, tonic, mode}],
    chords[{startSeconds, endSeconds, label}] with Harte-style labels such as D:sus2, A#:maj, A:7,
    firstChord, melody[{startSeconds, endSeconds, midi}], structure[{startSeconds, letter}].

Chord labels are `root[:quality][/bass]`. A bare root is major, `N` is no chord, and the bass is ignored. A root outside
A-G with sharps or flats, or a quality outside the table below, raises ValueError naming the label. Qualities group
into three families for the major/minor metric:
    major: maj, maj6, 6, maj7, 7, 9, maj9, 11, 13, add9
    minor: min, min6, min7, min9, minmaj7, dim, dim7, hdim7
    other: sus2, sus4, aug, 5, 1
Suspended and augmented chords are `other` because they have no third that settles major or minor.

A report that carries an `estimate_truncated` diagnostic whose message names a property (`audio.melody`,
`audio.chord_timeline`) has that property's metric detail marked as truncated.
"""
from __future__ import annotations

import argparse
import bisect
import functools
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

BEAT_WINDOW = 0.07
TEMPO_TOLERANCE = 0.04
MELODY_TOLERANCE_MIDI = 0.5
MELODY_STEP = 0.01
BOUNDARY_WINDOW = 3.0
FIXED_UNIT_SECONDS = 0.5
WINDOW_EPSILON = 1e-9
OCTAVE_FACTORS = (1.0, 2.0, 0.5, 3.0, 1 / 3)

SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"#": 1, "b": -1, "♯": 1, "♭": -1}
ROOT = re.compile(r"[A-Ga-g][#b♯♭]*")
QUALITY_FAMILY = {
    **dict.fromkeys(("maj", "maj6", "6", "maj7", "7", "9", "maj9", "11", "13", "add9"), "major"),
    **dict.fromkeys(("min", "min6", "min7", "min9", "minmaj7", "dim", "dim7", "hdim7"), "minor"),
    **dict.fromkeys(("sus2", "sus4", "aug", "5", "1"), "other"),
}


class Metric(NamedTuple):
    name: str
    score: float | None
    detail: str
    family: str = ""
    applicable: bool = True


Interval = tuple[float, float, str, tuple[str, ...]]


def missing(name: str, why: str) -> Metric:
    return Metric(name, None, f"not reported: {why}")


def not_applicable(name: str, why: str) -> Metric:
    return Metric(name, None, f"n/a: {why}", applicable=False)


def stamp(family: str, metrics: Sequence[Metric]) -> list[Metric]:
    return [m._replace(family=family) for m in metrics]


def family_mean(metrics: Sequence[Metric]) -> tuple[float, int]:
    """The mean of the per-family means and the number of families. An unreported metric counts as 0."""
    by_family: dict[str, list[float]] = {}
    for metric in metrics:
        if metric.applicable:
            by_family.setdefault(metric.family, []).append(metric.score or 0.0)
    means = [sum(scores) / len(scores) for scores in by_family.values()]
    return (sum(means) / len(means) if means else 0.0), len(means)


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
        self.truncations = [d.get("message", "") for d in report.get("diagnostics", [])
                            if d.get("code") == "estimate_truncated"]

    def get(self, prop: str) -> dict | None:
        value = self.values.get(prop)
        return value if isinstance(value, dict) else None

    def truncated(self, prop: str) -> bool:
        return any(prop in message for message in self.truncations)


def normalise_tonic(tonic: str) -> str:
    return SHARPS[pitch_class(tonic)]


def pitch_class(name: str) -> int:
    if not ROOT.fullmatch(name):
        raise ValueError(f"invalid pitch name {name!r}: expected A-G with sharps or flats")
    return (PITCH_CLASS[name[0].upper()] + sum(ACCIDENTALS[a] for a in name[1:])) % 12


@functools.lru_cache(maxsize=None)
def parse_chord(label: str) -> tuple[str | None, str | None]:
    """Harte-style `root:quality[/bass]` as (root, quality). `N` is (None, None). A bare root is major.

    Raises ValueError, naming the label, for a root outside A-G with accidentals or a quality outside QUALITY_FAMILY.
    """
    body = label.split("/")[0].strip()
    if body in ("N", "", "none"):
        return None, None
    root, _, quality = body.partition(":")
    quality = quality or "maj"
    try:
        normalised = normalise_tonic(root)
    except ValueError:
        raise ValueError(f"invalid chord label {label!r}: root {root!r} is not A-G with sharps or flats") from None
    if quality not in QUALITY_FAMILY:
        raise ValueError(f"invalid chord label {label!r}: unknown quality {quality!r}")
    return normalised, quality


def chord_family(label: str) -> str:
    """`major`, `minor`, `other` (suspended, augmented, power chords) or `none`."""
    root, quality = parse_chord(label)
    return "none" if root is None else QUALITY_FAMILY[quality]


def sweep_overlap(truth: Sequence[Interval], estimate: Sequence[Interval],
                  agrees: Callable[[Interval, Interval], bool]) -> float:
    """Seconds of truth time during which `agrees(truth_interval, estimate_interval)` holds. Inputs may be unsorted."""
    truth, estimate = sorted(truth, key=lambda i: i[:2]), sorted(estimate, key=lambda i: i[:2])
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


def count_matches_sorted(truth: Sequence[float], estimate: Sequence[float], window: float) -> int:
    """The largest one-to-one matching within `window`. For one fixed window on a line, sweeping both sorted lists and
    pairing the earliest unmatched events when they fall inside it is optimal."""
    t, e = sorted(truth), sorted(estimate)
    i = j = matched = 0
    while i < len(t) and j < len(e):
        gap = e[j] - t[i]
        if abs(gap) <= window + WINDOW_EPSILON:
            matched, i, j = matched + 1, i + 1, j + 1
        elif gap < 0:
            j += 1
        else:
            i += 1
    return matched


def count_matches_assignment(truth: Sequence[float], estimate: Sequence[float], window: float) -> int:
    """The same count by an explicit assignment: every pair outside the window costs 1, so the cheapest assignment
    holds the most pairs inside it. The matrix is truth by estimate, so use it only for short lists."""
    if not len(truth) or not len(estimate):
        return 0
    gap = np.abs(np.subtract.outer(np.asarray(truth, float), np.asarray(estimate, float)))
    inside = gap <= window + WINDOW_EPSILON
    rows, columns = linear_sum_assignment(np.where(inside, 0.0, 1.0))
    return int(inside[rows, columns].sum())


def f_measure(truth: Sequence[float], estimate: Sequence[float], window: float,
              count: Callable[[Sequence[float], Sequence[float], float], int] = count_matches_sorted
              ) -> tuple[float, int]:
    """F-measure of a one-to-one matching within `window`. Returns (F, matched)."""
    if not truth and not estimate:
        return 1.0, 0
    matched = count(truth, estimate, window)
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
    estimate = [(s["startSeconds"], s["endSeconds"], key_name(s["candidates"][0]), ())
                for s in (timeline or {}).get("segments") or [] if s.get("candidates")]
    if not estimate and single and single.get("candidates"):
        estimate = [(float("-inf"), float("inf"), key_name(single["candidates"][0]), ())]
    if not estimate:
        return [missing(name, "neither audio.key_timeline nor audio.key_estimate reported a key")]
    reference = [(k["startSeconds"], k["endSeconds"], key_name(k), ()) for k in truth_keys]
    total = sum(end - start for start, end, _, _ in reference)
    agreed = sweep_overlap(reference, estimate, lambda t, e: t[2] == e[2])
    names = ", ".join(dict.fromkeys(e[2] for e in estimate))
    return [Metric(name, agreed / total if total else 0.0, f"{agreed:.1f} of {total:.1f} s agree; estimated {names}")]


class ChordEstimate(NamedTuple):
    """Estimated chords, or None with the reason they cannot be used. `note` is a caveat for every detail."""
    intervals: list[Interval] | None
    note: str = ""


def unit_starts(value: dict, obs: Observations, count: int) -> list[float] | None:
    """Start times for the `count` perUnit entries, or None when nothing says where they are.

    The sources, in order: a `startSeconds` on every entry, `unitTimes`, the beat times for `basis` beats, and a fixed
    0.5 s step for `basis` fixed. Times must be ascending and at least `count` long.
    """
    own = [u.get("startSeconds") if isinstance(u, dict) else None for u in value["perUnit"]]
    if all(start is not None for start in own):
        starts = own
    elif value.get("unitTimes") is not None:
        starts = value["unitTimes"]
    elif value.get("basis") == "beats":
        starts = (obs.get("audio.beat_grid") or {}).get("beatTimes")
    elif value.get("basis") == "fixed":
        starts = [i * FIXED_UNIT_SECONDS for i in range(count)]
    else:
        starts = None
    if starts is None or len(starts) < count or any(b < a for a, b in zip(starts, starts[1:count])):
        return None
    return list(starts[:count])


def chord_intervals(obs: Observations) -> ChordEstimate:
    """Estimated chords as (start, end, label, alternative labels).

    Full detail's `perUnit` list wins over the capped segments. Its entries are labels or `{startSeconds, chord}`
    objects. When `perUnit` is present but its times cannot be resolved the result is unusable, and the capped
    segments are never substituted for it.
    """
    value = obs.get("audio.chord_timeline")
    if value is None:
        return ChordEstimate(None, "audio.chord_timeline absent")
    note = "estimate truncated, later chords may be missing" if obs.truncated("audio.chord_timeline") else ""
    per_unit = value.get("perUnit")
    if per_unit:
        starts = unit_starts(value, obs, len(per_unit))
        if starts is None:
            return ChordEstimate(None, "perUnit times unresolved")
        step = starts[-1] - starts[-2] if len(starts) > 1 else FIXED_UNIT_SECONDS
        ends = starts[1:] + [starts[-1] + step]
        labels = [u.get("chord", "N") if isinstance(u, dict) else u for u in per_unit]
        return ChordEstimate([(s, e, str(c), ()) for s, e, c in zip(starts, ends, labels)], note)
    return ChordEstimate([(s["startSeconds"], s["endSeconds"], s["chord"], tuple(map(str, s.get("alternatives", []))))
                          for s in value.get("segments") or []], note)


def chord_metrics(obs: Observations, truth: dict) -> list[Metric]:
    names = ("chord label overlap", "chord label overlap, with alternatives", "chord root overlap",
             "chord major/minor overlap")
    estimate, note = chord_intervals(obs)
    suffix = f"; {note}" if note and estimate is not None else ""
    if estimate is None:
        wanted = (names if "chords" in truth else ()) + (("first chord",) if "firstChord" in truth else ())
        return [missing(name, note) for name in wanted]
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
            metrics.append(Metric(name, seconds / total if total else 0.0, f"{seconds:.1f} of {total:.1f} s{suffix}"))
    if "firstChord" in truth:
        first = next((label for _, _, label, _ in sorted(estimate) if parse_chord(label)[0] is not None), None)
        metrics.append(Metric("first chord", float(first is not None and
                                                 parse_chord(first) == parse_chord(truth["firstChord"])),
                              f"{first} against {truth['firstChord']}{suffix}"))
    return metrics


def melody_pieces(value: dict) -> tuple[list[float], list[float | None], float]:
    """Contour starts, pitches and the end of the last point.

    The last point lasts one median contour spacing, so truth past the contour scores as a miss. `coveredSeconds` is
    never read, because a claim of coverage must not turn a missing tail into a hit.
    """
    contour = sorted(value.get("contour") or [], key=lambda p: p["startSeconds"])
    starts = [p["startSeconds"] for p in contour]
    spacing = (statistics.median(b - a for a, b in zip(starts, starts[1:])) if len(starts) > 1 else 0.0) \
        or FIXED_UNIT_SECONDS
    return starts, [p.get("midi") for p in contour], starts[-1] + spacing if starts else 0.0


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
    suffix = "; estimate truncated, the contour stops short of the track" if obs.truncated("audio.melody") else ""
    return [Metric(name, correct / voiced if voiced else 0.0,
                   f"{correct} of {voiced} voiced truth frames of 10 ms{suffix}")]


def structure_metrics(obs: Observations, truth_segments: Sequence[dict]) -> list[Metric]:
    name = "structure boundary hit rate (3 s)"
    boundaries = [s["startSeconds"] for s in sorted(truth_segments, key=lambda s: s["startSeconds"])][1:]
    if not boundaries:
        return [not_applicable(name, "one segment, no boundaries" if truth_segments else "no segments")]
    value = obs.get("audio.structure")
    if value is None:
        return [missing(name, "audio.structure absent")]
    if not value.get("segments"):
        return [missing(name, "audio.structure has no segments")]
    estimated = sorted(s["startSeconds"] for s in value["segments"])[1:]
    _, matched = f_measure(boundaries, estimated, BOUNDARY_WINDOW, count_matches_assignment)
    precision = f"{matched / len(estimated):.2f}" if estimated else "n/a"
    return [Metric(name, matched / len(boundaries),
                   f"{matched} of {len(boundaries)} truth boundaries; precision {precision}")]


def evaluate(report: dict, truth: dict) -> list[Metric]:
    obs = Observations(report)
    metrics: list[Metric] = []
    if "tempoBpm" in truth:
        metrics += stamp("tempo", tempo_metrics(obs, truth["tempoBpm"]))
    if "beats" in truth:
        metrics += stamp("beats", beat_metrics(obs, truth["beats"]))
    metrics += stamp("meter", meter_metrics(obs, truth))
    if "keys" in truth:
        metrics += stamp("key", key_metrics(obs, truth["keys"]))
    if "chords" in truth or "firstChord" in truth:
        metrics += stamp("chords", chord_metrics(obs, truth))
    if "melody" in truth:
        metrics += stamp("melody", melody_metrics(obs, truth["melody"]))
    if "structure" in truth:
        metrics += stamp("structure", structure_metrics(obs, truth["structure"]))
    return metrics


def render(metrics: Sequence[Metric], markdown: bool) -> str:
    def cell(m: Metric) -> str:
        return "n/a" if m.score is None else f"{m.score:.3f}"

    mean, families = family_mean(metrics)
    label = f"mean of {families} families"
    unreported = sum(m.score is None and m.applicable for m in metrics)
    note = f"{unreported} not reported, counted as 0" if unreported else "all reported"
    if markdown:
        rows = ["| metric | score | detail |", "|---|---|---|"]
        rows += [f"| {m.name} | {cell(m)} | {m.detail} |" for m in metrics]
        rows.append(f"| **{label}** | **{mean:.3f}** | {note} |")
        return "\n".join(rows)
    width = max([len(m.name) for m in metrics] + [len(label)])
    lines = [f"{m.name:<{width}}  {cell(m):>5}  {m.detail}" for m in metrics]
    lines.append(f"{label:<{width}}  {mean:>5.3f}  {note}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(json.loads(args.report.read_text()), json.loads(args.truth.read_text()))
    except (ValueError, OSError) as error:
        print(f"evaluate: {error}", file=sys.stderr)
        return 2
    print(render(metrics, args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
