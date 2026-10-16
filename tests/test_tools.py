"""Tests for the development tools in `tools/`. They read only the CLI's JSON and MIDI files, never audio."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import mido
import numpy
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))

import benchmark  # noqa: E402
import evaluate  # noqa: E402
import midi_truth  # noqa: E402

TRUTH = json.loads((Path(__file__).parent / "fixtures" / "truth-synthetic.json").read_text())


def observation(prop: str, value, scope: str = "mixdown") -> dict:
    return {"id": "o1", "sourceIds": ["s1"], "property": prop, "value": value, "scope": scope,
            "provenance": "inferred", "confidence": "low"}


def report_from_truth(truth: dict, *, full: bool = True) -> dict:
    """A report in the CLI's shape whose estimates equal the truth exactly."""
    contour = []
    for beat in truth["beats"]:
        note = next((n for n in truth["melody"] if n["startSeconds"] <= beat < n["endSeconds"]), None)
        contour.append({"startSeconds": beat, "midi": note["midi"] if note else None})
    grid = {"bpm": truth["tempoBpm"], "beatCount": len(truth["beats"]), "firstBeatSeconds": truth["beats"][0],
            "tempoStability": 1.0}
    meter = {"candidates": [{"signature": truth["meter"], "score": 0.9}], "downbeatPhase": 0,
             "barCount": len(truth["downbeats"])}
    if full:
        grid["beatTimes"] = truth["beats"]
        meter["downbeatTimes"] = truth["downbeats"]
    segments = [{"startSeconds": k["startSeconds"], "endSeconds": k["endSeconds"],
                 "candidates": [{"tonic": k["tonic"], "mode": k["mode"], "correlation": 0.8}]} for k in truth["keys"]]
    ends = [s["startSeconds"] for s in truth["structure"][1:]] + [truth["keys"][-1]["endSeconds"]]
    structure = [{"letter": s["letter"], "startSeconds": s["startSeconds"], "endSeconds": end}
                 for s, end in zip(truth["structure"], ends)]
    return {"schemaVersion": "1.1.0", "observations": [
        observation("audio.duration", 8.0),
        observation("audio.tempo_estimate", {"candidates": [{"bpm": truth["tempoBpm"], "relativeStrength": 1.0}],
                                             "octaveRelatedCandidates": False}),
        observation("audio.beat_grid", grid),
        observation("audio.meter_estimate", meter),
        observation("audio.key_timeline", {"cellSeconds": 5.0, "segments": segments}),
        observation("audio.chord_timeline", {"basis": "beats", "segments": [
            {"startSeconds": c["startSeconds"], "endSeconds": c["endSeconds"], "chord": c["label"]}
            for c in truth["chords"]]}),
        observation("audio.melody", {"contour": contour, "coveredSeconds": 8.0}),
        observation("audio.structure", {"segments": structure, "distinctLetters": 2}),
    ]}


def with_value(report: dict, prop: str, value) -> dict:
    changed = copy.deepcopy(report)
    next(o for o in changed["observations"] if o["property"] == prop)["value"] = value
    return changed


def without(report: dict, prop: str) -> dict:
    changed = copy.deepcopy(report)
    changed["observations"] = [o for o in changed["observations"] if o["property"] != prop]
    return changed


def scores(metrics) -> dict:
    return {m.name: m.score for m in metrics}


class EvaluateTests(unittest.TestCase):
    def test_evaluate_scores_a_perfect_report(self):
        metrics = evaluate.evaluate(report_from_truth(TRUTH), TRUTH)
        self.assertTrue(metrics)
        for metric in metrics:
            self.assertEqual(metric.score, 1.0, metric)
        self.assertEqual(evaluate.family_mean(metrics), (1.0, 7))

    def test_evaluate_marks_missing_estimates(self):
        report = {"observations": [observation("audio.duration", 8.0)]}
        metrics = evaluate.evaluate(report, TRUTH)
        self.assertTrue(metrics)
        for metric in metrics:
            self.assertIsNone(metric.score, metric)
            self.assertIn("not reported", metric.detail)
        self.assertEqual(evaluate.family_mean(metrics), (0.0, 7))
        perfect = evaluate.evaluate(report_from_truth(TRUTH), TRUTH)
        self.assertEqual({m.name for m in metrics}, {m.name for m in perfect})

    def test_only_keys_present_in_the_truth_are_evaluated(self):
        metrics = evaluate.evaluate(report_from_truth(TRUTH), {"tempoBpm": 120})
        self.assertTrue(metrics)
        self.assertTrue(all(m.name.startswith("tempo") for m in metrics))

    def test_summary_report_cannot_score_beats_without_beat_times(self):
        metrics = scores(evaluate.evaluate(report_from_truth(TRUTH, full=False), TRUTH))
        self.assertIsNone(metrics["beat F-measure (70 ms)"])
        self.assertIsNone(metrics["downbeat F-measure (70 ms)"])
        self.assertEqual(metrics["meter"], 1.0)

    def test_tempo_is_strict_and_octave_aware(self):
        half = with_value(report_from_truth(TRUTH), "audio.beat_grid",
                          {"bpm": 60.0, "beatCount": 8, "firstBeatSeconds": 0.0, "tempoStability": 1.0})
        metrics = scores(evaluate.evaluate(half, {"tempoBpm": 120}))
        self.assertEqual(metrics["tempo accuracy (beat_grid)"], 0.0)
        self.assertEqual(metrics["tempo accuracy, octave-aware (beat_grid)"], 1.0)
        near = with_value(half, "audio.beat_grid", {"bpm": 124.0})
        self.assertEqual(scores(evaluate.evaluate(near, {"tempoBpm": 120}))["tempo accuracy (beat_grid)"], 1.0)
        far = with_value(half, "audio.beat_grid", {"bpm": 126.0})
        self.assertEqual(scores(evaluate.evaluate(far, {"tempoBpm": 120}))["tempo accuracy (beat_grid)"], 0.0)

    def test_tempo_estimate_scores_even_without_a_beat_grid(self):
        metrics = scores(evaluate.evaluate(without(report_from_truth(TRUTH), "audio.beat_grid"), {"tempoBpm": 120}))
        self.assertEqual(metrics["tempo accuracy (tempo_estimate)"], 1.0)
        self.assertIsNone(metrics["tempo accuracy (beat_grid)"])

    def test_beat_f_measure_uses_a_70_ms_window(self):
        late = with_value(report_from_truth(TRUTH), "audio.beat_grid",
                          {"bpm": 120, "beatTimes": [b + 0.06 for b in TRUTH["beats"]]})
        self.assertEqual(scores(evaluate.evaluate(late, TRUTH))["beat F-measure (70 ms)"], 1.0)
        later = with_value(late, "audio.beat_grid", {"bpm": 120, "beatTimes": [b + 0.1 for b in TRUTH["beats"]]})
        self.assertEqual(scores(evaluate.evaluate(later, TRUTH))["beat F-measure (70 ms)"], 0.0)
        half = with_value(late, "audio.beat_grid", {"bpm": 60, "beatTimes": TRUTH["beats"][::2]})
        self.assertAlmostEqual(scores(evaluate.evaluate(half, TRUTH))["beat F-measure (70 ms)"], 2 / 3, places=6)

    def test_meter_mismatch_scores_zero(self):
        wrong = with_value(report_from_truth(TRUTH), "audio.meter_estimate",
                           {"candidates": [{"signature": "3/4", "score": 0.7}, {"signature": "4/4", "score": 0.6}]})
        self.assertEqual(scores(evaluate.evaluate(wrong, TRUTH))["meter"], 0.0)

    def test_key_agreement_is_time_weighted(self):
        segments = [{"startSeconds": 0, "endSeconds": 6, "candidates": [{"tonic": "D", "mode": "minor"}]},
                    {"startSeconds": 6, "endSeconds": 8, "candidates": [{"tonic": "F", "mode": "major"}]}]
        report = with_value(report_from_truth(TRUTH), "audio.key_timeline", {"segments": segments})
        self.assertAlmostEqual(scores(evaluate.evaluate(report, TRUTH))["key agreement"], 6 / 8)

    def test_key_agreement_falls_back_to_the_single_key_estimate(self):
        report = without(report_from_truth(TRUTH), "audio.key_timeline")
        report["observations"].append(observation(
            "audio.key_estimate", {"candidates": [{"tonic": "D", "mode": "minor", "correlation": 0.8}]}))
        self.assertAlmostEqual(scores(evaluate.evaluate(report, TRUTH))["key agreement"], 0.5)

    def test_chord_overlap_separates_label_root_and_family(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 2, "label": "D:sus2"},
                            {"startSeconds": 2, "endSeconds": 4, "label": "F:maj"}]}
        estimate = {"basis": "beats", "segments": [
            {"startSeconds": 0, "endSeconds": 2, "chord": "D:min"},
            {"startSeconds": 2, "endSeconds": 4, "chord": "F:7"}]}
        report = {"observations": [observation("audio.chord_timeline", estimate)]}
        metrics = scores(evaluate.evaluate(report, truth))
        self.assertEqual(metrics["chord label overlap"], 0.0)
        self.assertEqual(metrics["chord root overlap"], 1.0)
        self.assertEqual(metrics["chord major/minor overlap"], 0.5)

    def test_chord_alternatives_count_in_their_own_metric(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 4, "label": "D:sus2"}]}
        estimate = {"basis": "beats", "segments": [
            {"startSeconds": 0, "endSeconds": 4, "chord": "A:sus4", "alternatives": ["D:sus2"]}]}
        metrics = scores(evaluate.evaluate({"observations": [observation("audio.chord_timeline", estimate)]}, truth))
        self.assertEqual(metrics["chord label overlap"], 0.0)
        self.assertEqual(metrics["chord label overlap, with alternatives"], 1.0)

    def test_chord_labels_are_normalised(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 4, "label": "Bb:maj/3"}]}
        estimate = {"segments": [{"startSeconds": 0, "endSeconds": 4, "chord": "A#:maj"}]}
        metrics = scores(evaluate.evaluate({"observations": [observation("audio.chord_timeline", estimate)]}, truth))
        self.assertEqual(metrics["chord label overlap"], 1.0)

    def test_per_unit_chords_are_used_when_full_detail_has_them(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 2, "label": "D:sus2"}]}
        estimate = {"basis": "fixed", "segments": [], "perUnit": ["D:sus2", "D:sus2", "D:sus2", "D:sus2"]}
        report = {"observations": [observation("audio.chord_timeline", estimate)]}
        self.assertEqual(scores(evaluate.evaluate(report, truth))["chord label overlap"], 1.0)
        estimate["perUnit"][2] = "N"
        self.assertEqual(scores(evaluate.evaluate(report, truth))["chord label overlap"], 0.75)

    def test_per_unit_chords_on_beats_take_their_times_from_the_beat_grid(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 3, "label": "D:sus2"}]}
        chords = {"basis": "beats", "segments": [], "perUnit": ["D:sus2", "F:maj", "D:sus2"]}
        grid = {"bpm": 40, "beatTimes": [0.0, 1.0, 2.0]}
        report = {"observations": [observation("audio.chord_timeline", chords), observation("audio.beat_grid", grid)]}
        self.assertAlmostEqual(scores(evaluate.evaluate(report, truth))["chord label overlap"], 2 / 3)

    def test_first_chord_skips_no_chord_segments(self):
        estimate = {"segments": [{"startSeconds": 0, "endSeconds": 1, "chord": "N"},
                                 {"startSeconds": 1, "endSeconds": 4, "chord": "D:sus2"}]}
        report = {"observations": [observation("audio.chord_timeline", estimate)]}
        self.assertEqual(scores(evaluate.evaluate(report, {"firstChord": "D:sus2"}))["first chord"], 1.0)
        self.assertEqual(scores(evaluate.evaluate(report, {"firstChord": "A:sus4"}))["first chord"], 0.0)

    def test_melody_raw_pitch_accuracy_allows_50_cents(self):
        truth = {"melody": [{"startSeconds": 0, "endSeconds": 2, "midi": 60}]}
        within = {"contour": [{"startSeconds": 0, "midi": 60.4}, {"startSeconds": 1, "midi": 59.6}]}
        report = {"observations": [observation("audio.melody", within)]}
        self.assertEqual(scores(evaluate.evaluate(report, truth))["melody raw pitch accuracy"], 1.0)
        half_wrong = {"contour": [{"startSeconds": 0, "midi": 60}, {"startSeconds": 1, "midi": 61}]}
        report = {"observations": [observation("audio.melody", half_wrong)]}
        self.assertAlmostEqual(scores(evaluate.evaluate(report, truth))["melody raw pitch accuracy"], 0.5, places=2)
        unvoiced = {"contour": [{"startSeconds": 0, "midi": None}, {"startSeconds": 1, "midi": 60}]}
        report = {"observations": [observation("audio.melody", unvoiced)]}
        self.assertAlmostEqual(scores(evaluate.evaluate(report, truth))["melody raw pitch accuracy"], 0.5, places=2)

    def test_structure_boundary_hit_rate_uses_a_3_second_window(self):
        truth = {"structure": [{"startSeconds": 0, "letter": "A"}, {"startSeconds": 30, "letter": "B"},
                               {"startSeconds": 60, "letter": "A"}]}
        estimate = {"segments": [{"letter": "A", "startSeconds": 0, "endSeconds": 32},
                                 {"letter": "B", "startSeconds": 32, "endSeconds": 70},
                                 {"letter": "A", "startSeconds": 70, "endSeconds": 90}]}
        report = {"observations": [observation("audio.structure", estimate)]}
        self.assertAlmostEqual(scores(evaluate.evaluate(report, truth))["structure boundary hit rate (3 s)"], 0.5)

    def test_cli_prints_a_table_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path, truth_path = Path(directory) / "r.json", Path(directory) / "t.json"
            report_path.write_text(json.dumps(without(report_from_truth(TRUTH), "audio.melody")))
            truth_path.write_text(json.dumps(TRUTH))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(evaluate.main([str(report_path), str(truth_path)]), 0)
            text = out.getvalue()
            self.assertIn("melody raw pitch accuracy", text)
            self.assertIn("not reported", text)
            self.assertIn("mean of 7 families", text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                evaluate.main([str(report_path), str(truth_path), "--markdown"])
            self.assertIn("| metric | score | detail |", out.getvalue())

    def test_a_scope_other_than_mixdown_is_used_only_when_no_mixdown_estimate_exists(self):
        stem = observation("audio.beat_grid", {"bpm": 60.0}, scope="stem")
        mix = observation("audio.beat_grid", {"bpm": 120.0})
        report = {"observations": [stem, mix]}
        self.assertEqual(scores(evaluate.evaluate(report, {"tempoBpm": 120}))["tempo accuracy (beat_grid)"], 1.0)
        report = {"observations": [stem]}
        self.assertEqual(scores(evaluate.evaluate(report, {"tempoBpm": 60}))["tempo accuracy (beat_grid)"], 1.0)


def single(prop: str, value, **extra) -> dict:
    return {"observations": [observation(prop, value)], **extra}


def truncation(prop: str) -> dict:
    return {"code": "estimate_truncated", "severity": "warning", "message": f"{prop} was cut at its cap"}


class EvaluateFixTests(unittest.TestCase):
    def test_melody_ignores_covered_seconds_so_truth_past_the_contour_fails(self):
        truth = {"melody": [{"startSeconds": 0, "endSeconds": 4, "midi": 60}]}
        contour = {"contour": [{"startSeconds": 0, "midi": 60}, {"startSeconds": 1, "midi": 60}],
                   "coveredSeconds": 4.0}
        metric = evaluate.evaluate(single("audio.melody", contour), truth)[0]
        self.assertAlmostEqual(metric.score, 0.5, places=2)
        self.assertNotIn("truncated", metric.detail)

    def test_melody_detail_reports_a_truncation_diagnostic(self):
        truth = {"melody": [{"startSeconds": 0, "endSeconds": 4, "midi": 60}]}
        contour = {"contour": [{"startSeconds": 0, "midi": 60}, {"startSeconds": 1, "midi": 60}]}
        report = single("audio.melody", contour, diagnostics=[truncation("audio.melody")])
        self.assertIn("truncated", evaluate.evaluate(report, truth)[0].detail)
        other = single("audio.melody", contour, diagnostics=[truncation("audio.structure")])
        self.assertNotIn("truncated", evaluate.evaluate(other, truth)[0].detail)

    def test_melody_pitch_on_unvoiced_truth_frames_is_ignored(self):
        truth = {"melody": [{"startSeconds": 0, "endSeconds": 1, "midi": 60}]}
        contour = {"contour": [{"startSeconds": 0, "midi": 60}, {"startSeconds": 1, "midi": 99},
                               {"startSeconds": 2, "midi": 99}]}
        self.assertEqual(evaluate.evaluate(single("audio.melody", contour), truth)[0].score, 1.0)

    def test_unsorted_truth_and_estimates_are_scored_by_time(self):
        truth = {"chords": [{"startSeconds": 3, "endSeconds": 6, "label": "F:maj"},
                            {"startSeconds": 0, "endSeconds": 3, "label": "D:sus2"}]}
        estimate = {"basis": "beats", "segments": [{"startSeconds": 3, "endSeconds": 6, "chord": "F:maj"},
                                                   {"startSeconds": 0, "endSeconds": 3, "chord": "D:sus2"}]}
        metrics = scores(evaluate.evaluate(single("audio.chord_timeline", estimate), truth))
        self.assertEqual(metrics["chord label overlap"], 1.0)
        keys = [{"startSeconds": 4, "endSeconds": 8, "tonic": "F", "mode": "major"},
                {"startSeconds": 0, "endSeconds": 4, "tonic": "D", "mode": "minor"}]
        report = with_value(report_from_truth(TRUTH), "audio.key_timeline", {"segments": [
            {"startSeconds": 4, "endSeconds": 8, "candidates": [{"tonic": "F", "mode": "major"}]},
            {"startSeconds": 0, "endSeconds": 4, "candidates": [{"tonic": "D", "mode": "minor"}]}]})
        self.assertEqual(scores(evaluate.evaluate(report, {"keys": keys}))["key agreement"], 1.0)

    def test_misaligned_chord_segments_are_scored_by_time(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 3, "label": "D:sus2"}]}
        estimate = {"basis": "beats", "segments": [{"startSeconds": 0, "endSeconds": 1, "chord": "D:sus2"},
                                                   {"startSeconds": 1, "endSeconds": 3, "chord": "F:maj"}]}
        metrics = scores(evaluate.evaluate(single("audio.chord_timeline", estimate), truth))
        self.assertAlmostEqual(metrics["chord label overlap"], 1 / 3)
        self.assertAlmostEqual(metrics["chord root overlap"], 1 / 3)

    def test_unresolved_per_unit_times_are_not_reported_never_a_silent_fallback(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 3, "label": "D:sus2"}], "firstChord": "D:sus2"}
        estimate = {"basis": "beats", "perUnit": ["X:maj"] * 3,
                    "segments": [{"startSeconds": 0, "endSeconds": 3, "chord": "D:sus2"}]}
        metrics = evaluate.evaluate(single("audio.chord_timeline", estimate), truth)
        self.assertTrue(metrics)
        for metric in metrics:
            self.assertIsNone(metric.score, metric)
            self.assertIn("not reported", metric.detail)
            self.assertIn("perUnit times unresolved", metric.detail)

    def test_chord_detail_reports_a_truncation_diagnostic(self):
        truth = {"chords": [{"startSeconds": 0, "endSeconds": 4, "label": "D:sus2"}]}
        estimate = {"segments": [{"startSeconds": 0, "endSeconds": 2, "chord": "D:sus2"}]}
        report = single("audio.chord_timeline", estimate, diagnostics=[truncation("audio.chord_timeline")])
        for metric in evaluate.evaluate(report, truth):
            self.assertIn("truncated", metric.detail)
        for metric in evaluate.evaluate(single("audio.chord_timeline", estimate), truth):
            self.assertNotIn("truncated", metric.detail)

    def test_key_falls_back_to_the_single_estimate_when_the_timeline_has_no_usable_segment(self):
        report = {"observations": [
            observation("audio.key_timeline", {"segments": [{"startSeconds": 0, "endSeconds": 4}]}),
            observation("audio.key_estimate", {"candidates": [{"tonic": "D", "mode": "minor"}]})]}
        truth = {"keys": [{"startSeconds": 0, "endSeconds": 4, "tonic": "D", "mode": "minor"}]}
        self.assertEqual(scores(evaluate.evaluate(report, truth))["key agreement"], 1.0)

    def test_matching_is_optimal_not_greedy(self):
        self.assertEqual(evaluate.f_measure([1.00, 1.03], [0.97, 0.99], 0.05), (1.0, 2))
        self.assertEqual(evaluate.f_measure([1.00, 1.03], [0.97, 0.99], 0.05, evaluate.count_matches_assignment),
                         (1.0, 2))

    def test_duplicated_estimate_beats_are_not_matched_twice(self):
        score, matched = evaluate.f_measure([1, 2, 3], [1, 1.01, 2, 2.01, 3, 3.01], 0.07)
        self.assertEqual(matched, 3)
        self.assertAlmostEqual(score, 2 * 3 / 9)

    def test_the_sorted_matcher_agrees_with_the_assignment_matcher(self):
        rng = random.Random(7)
        for _ in range(200):
            truth = [rng.uniform(0, 20) for _ in range(rng.randint(0, 12))]
            estimate = [rng.uniform(0, 20) for _ in range(rng.randint(0, 12))]
            window = rng.choice([0.05, 0.5, 3.0])
            self.assertEqual(evaluate.count_matches_sorted(truth, estimate, window),
                             evaluate.count_matches_assignment(truth, estimate, window), (truth, estimate, window))

    def test_structure_boundaries_use_an_optimal_assignment(self):
        truth = {"structure": [{"startSeconds": s, "letter": "A"} for s in (0, 10, 12)]}
        estimate = {"segments": [{"letter": "A", "startSeconds": s, "endSeconds": s + 1} for s in (0, 8, 10.5)]}
        metric = evaluate.evaluate(single("audio.structure", estimate), truth)[0]
        self.assertEqual(metric.score, 1.0)

    def test_a_one_segment_truth_structure_is_not_applicable(self):
        truth = {"structure": [{"startSeconds": 0, "letter": "A"}]}
        metric = evaluate.evaluate({"observations": []}, truth)[0]
        self.assertIsNone(metric.score)
        self.assertEqual(metric.detail, "n/a: one segment, no boundaries")
        self.assertEqual(evaluate.family_mean([metric]), (0.0, 0))
        self.assertNotIn("not reported", evaluate.render([metric], markdown=False))

    def test_an_empty_estimated_structure_is_not_reported(self):
        truth = {"structure": [{"startSeconds": 0, "letter": "A"}, {"startSeconds": 10, "letter": "B"}]}
        metric = evaluate.evaluate(single("audio.structure", {"segments": []}), truth)[0]
        self.assertIsNone(metric.score)
        self.assertIn("not reported", metric.detail)

    def test_the_summary_is_a_mean_of_per_family_means(self):
        tempo = evaluate.Metric("a", 1.0, "", "tempo"), evaluate.Metric("b", 1.0, "", "tempo")
        key = (evaluate.Metric("c", 0.0, "", "key"), evaluate.Metric("d", None, "not reported", "key"),
               evaluate.Metric("e", 1.0, "", "key"))
        self.assertEqual(evaluate.family_mean([*tempo, *key]), ((1.0 + 1 / 3) / 2, 2))
        text = evaluate.render([*tempo, *key], markdown=False)
        self.assertIn("mean of 2 families", text)
        self.assertIn("mean of 2 families", evaluate.render([*tempo, *key], markdown=True))

    def test_evaluate_stamps_every_metric_with_a_family(self):
        metrics = evaluate.evaluate(report_from_truth(TRUTH), TRUTH)
        self.assertEqual({m.family for m in metrics},
                         {"tempo", "beats", "meter", "key", "chords", "melody", "structure"})

    def test_an_invalid_chord_root_or_quality_raises_with_the_label(self):
        with self.assertRaisesRegex(ValueError, "H:maj"):
            evaluate.parse_chord("H:maj")
        with self.assertRaisesRegex(ValueError, "C:wobble"):
            evaluate.parse_chord("C:wobble")
        with self.assertRaisesRegex(ValueError, "C:wobble"):
            evaluate.chord_family("C:wobble")

    def test_chord_families_are_documented_and_consistent(self):
        expected = {"maj": "major", "maj7": "major", "7": "major", "6": "major", "min": "minor", "min7": "minor",
                    "dim": "minor", "dim7": "minor", "hdim7": "minor", "sus2": "other", "sus4": "other",
                    "aug": "other"}
        for quality, family in expected.items():
            self.assertEqual(evaluate.chord_family(f"C:{quality}"), family, quality)
        self.assertEqual(evaluate.chord_family("C"), "major")
        self.assertEqual(evaluate.chord_family("N"), "none")


def build_midi(path: Path, tracks: dict[str, list[tuple[float, float, int]]], *, bpm: float = 120,
               signature: tuple[int, int] = (4, 4), second_tempo: float | None = None) -> None:
    """Write a type 1 file. Each note is (start_beat, length_beats, pitch)."""
    ticks = 480
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("time_signature", numerator=signature[0], denominator=signature[1]))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    if second_tempo is not None:
        conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(second_tempo), time=ticks * 4))
    midi.tracks.append(conductor)
    for name, notes in tracks.items():
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
        events = []
        for start, length, pitch in notes:
            events.append((round(start * ticks), 1, pitch))
            events.append((round((start + length) * ticks), 0, pitch))
        last = 0
        for tick, is_on, pitch in sorted(events):
            track.append(mido.Message("note_on", note=pitch, velocity=90 if is_on else 0, time=tick - last))
            last = tick
        midi.tracks.append(track)
    midi.save(path)


def chord_notes(pitches: list[int], start_beat: float, length: float = 4) -> list[tuple[float, float, int]]:
    return [(start_beat, length, p) for p in pitches]


class MidiTruthTests(unittest.TestCase):
    def derive(self, tracks, *, chord_tracks=("Pad",), bass="Bass", melody=None, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.mid"
            build_midi(path, tracks, **{k: options.pop(k) for k in ("bpm", "signature", "second_tempo")
                                        if k in options})
            return midi_truth.derive_truth(path, chord_tracks=list(chord_tracks), bass_track=bass,
                                           melody_track=melody, **options)

    def test_midi_truth_labels_known_chords(self):
        pad = chord_notes([62, 64, 69], 0) + chord_notes([65, 69, 72], 4) + chord_notes([57, 61, 64, 67], 8)
        bass = [(0, 4, 38), (4, 4, 41), (8, 4, 45)]
        truth = self.derive({"Pad": pad, "Bass": bass})
        self.assertEqual([c["label"] for c in truth["chords"]], ["D:sus2", "F:maj", "A:7"])
        self.assertEqual([c["startSeconds"] for c in truth["chords"]], [0.0, 2.0, 4.0])
        self.assertEqual(truth["chords"][-1]["endSeconds"], 6.0)
        self.assertEqual(truth["tempoBpm"], 120.0)
        self.assertEqual(truth["meter"], "4/4")
        self.assertEqual(truth["beats"][:4], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(len(truth["beats"]), 12)
        self.assertEqual(truth["downbeats"], [0.0, 2.0, 4.0])

    def test_bass_separates_sus2_from_its_sus4_twin(self):
        pad = chord_notes([62, 64, 69], 0) + chord_notes([62, 64, 69], 4)
        bass = [(0, 4, 38), (4, 4, 45)]
        truth = self.derive({"Pad": pad, "Bass": bass})
        self.assertEqual([c["label"] for c in truth["chords"]], ["D:sus2", "A:sus4"])

    def test_the_bass_pitch_class_joins_the_chord(self):
        truth = self.derive({"Pad": chord_notes([64, 67], 0), "Bass": [(0, 4, 36)]})
        self.assertEqual([c["label"] for c in truth["chords"]], ["C:maj"])

    def test_a_bare_fifth_names_no_chord(self):
        truth = self.derive({"Pad": chord_notes([62, 69], 0), "Bass": [(0, 4, 38)]})
        self.assertEqual([c["label"] for c in truth["chords"]], ["N"])

    def test_the_bass_picks_the_root_of_a_chord_that_two_templates_fit(self):
        pad = chord_notes([60, 64, 67, 69], 0)
        self.assertEqual([c["label"] for c in self.derive({"Pad": pad, "Bass": [(0, 4, 36)]})["chords"]], ["C:maj"])
        self.assertEqual([c["label"] for c in self.derive({"Pad": pad, "Bass": [(0, 4, 33)]})["chords"]], ["A:min7"])

    def test_a_staccato_arpeggio_still_names_its_chord(self):
        arpeggio = [(beat + step / 4, 0.1, pitch) for beat in range(4) for step, pitch in enumerate((62, 64, 69, 62))]
        truth = self.derive({"Chord": arpeggio, "Bass": [(0, 4, 38)]}, chord_tracks=("Chord",))
        self.assertEqual([c["label"] for c in truth["chords"]], ["D:sus2"])

    def test_a_beat_with_no_chord_notes_is_n(self):
        pad = chord_notes([62, 64, 69], 0, 2) + chord_notes([65, 69, 72], 4, 4)
        truth = self.derive({"Pad": pad, "Bass": [(0, 2, 38), (4, 4, 41)]})
        self.assertEqual([c["label"] for c in truth["chords"]], ["D:sus2", "N", "F:maj"])

    def test_omit_no_chord_leaves_the_resting_span_out(self):
        pad = chord_notes([62, 64, 69], 0, 2) + chord_notes([65, 69, 72], 4, 4)
        truth = self.derive({"Pad": pad, "Bass": [(0, 2, 38), (4, 4, 41)]}, omit_no_chord=True)
        self.assertEqual([(c["label"], c["startSeconds"]) for c in truth["chords"]], [("D:sus2", 0.0), ("F:maj", 2.0)])

    def test_repeats_merge_and_offset_shifts_every_time(self):
        pad = chord_notes([62, 64, 69], 0, 8)
        truth = self.derive({"Pad": pad, "Bass": [(0, 8, 38)]}, offset=1.5)
        self.assertEqual(truth["chords"], [{"startSeconds": 1.5, "endSeconds": 5.5, "label": "D:sus2"}])
        self.assertEqual(truth["beats"][0], 1.5)
        self.assertEqual(truth["downbeats"], [1.5, 3.5])

    def test_the_melody_track_becomes_midi_notes_in_seconds(self):
        lead = [(0, 1, 74), (1, 1, 76), (3, 1, 77)]
        truth = self.derive({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)], "Lead": lead}, melody="Lead")
        self.assertEqual(truth["melody"], [
            {"startSeconds": 0.0, "endSeconds": 0.5, "midi": 74},
            {"startSeconds": 0.5, "endSeconds": 1.0, "midi": 76},
            {"startSeconds": 1.5, "endSeconds": 2.0, "midi": 77}])

    def test_overlapping_melody_notes_are_cut_and_reported(self):
        lead = [(0, 2, 74), (1, 1, 76)]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            truth = self.derive({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)], "Lead": lead},
                                melody="Lead")
        self.assertEqual([(n["startSeconds"], n["endSeconds"]) for n in truth["melody"]], [(0.0, 0.5), (0.5, 1.0)])
        self.assertIn("overlap", stderr.getvalue())

    def test_the_duration_extends_the_beat_grid_past_the_last_note(self):
        truth = self.derive({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)]}, duration=5.0)
        self.assertEqual(len(truth["beats"]), 10)

    def test_a_tempo_change_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tempo"):
            self.derive({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)]}, second_tempo=140)

    def test_an_unknown_track_name_lists_the_available_ones(self):
        with self.assertRaisesRegex(ValueError, "Pad"):
            self.derive({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)]}, chord_tracks=("Nope",))

    def test_three_four_meter_puts_downbeats_every_three_beats(self):
        pad = chord_notes([62, 64, 69], 0, 6)
        truth = self.derive({"Pad": pad, "Bass": [(0, 6, 38)]}, signature=(3, 4))
        self.assertEqual(truth["meter"], "3/4")
        self.assertEqual(truth["downbeats"], [0.0, 1.5])

    def test_cli_writes_json_that_the_evaluator_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.mid"
            build_midi(path, {"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)]})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = midi_truth.main([str(path), "--chord-tracks", "Pad", "--bass-track", "Bass"])
            self.assertEqual(code, 0)
            truth = json.loads(out.getvalue())
            metrics = evaluate.evaluate({"observations": [observation("audio.chord_timeline", {
                "segments": [{"startSeconds": 0, "endSeconds": 2, "chord": "D:sus2"}]})]}, truth)
            self.assertEqual(scores(metrics)["chord label overlap"], 1.0)

    def derive_quietly(self, tracks, **options):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            truth = self.derive(tracks, **options)
        return truth, stderr.getvalue()

    def test_unterminated_notes_close_at_the_tracks_final_tick_with_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hanging.mid"
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            conductor = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)),
                                        mido.MetaMessage("time_signature", numerator=4, denominator=4)])
            pad = mido.MidiTrack([mido.MetaMessage("track_name", name="Pad")])
            pad += [mido.Message("note_on", note=n, velocity=90, time=0) for n in (62, 64, 69)]
            pad.append(mido.MetaMessage("end_of_track", time=4 * 480))
            bass = mido.MidiTrack([mido.MetaMessage("track_name", name="Bass"),
                                   mido.Message("note_on", note=38, velocity=90, time=0),
                                   mido.Message("note_off", note=38, time=4 * 480)])
            midi.tracks += [conductor, pad, bass]
            midi.save(path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                truth = midi_truth.derive_truth(path, chord_tracks=["Pad"], bass_track="Bass")
        self.assertEqual([(c["label"], c["endSeconds"]) for c in truth["chords"]], [("D:sus2", 2.0)])
        self.assertIn("3 unterminated", stderr.getvalue())

    def test_the_duration_clips_chords_and_melody_to_the_audio(self):
        tracks = {"Pad": chord_notes([62, 64, 69], 0, 16), "Bass": [(0, 16, 38)], "Lead": [(0, 2, 74), (5, 4, 76)]}
        truth, _ = self.derive_quietly(tracks, melody="Lead", duration=3.0)
        self.assertEqual(truth["chords"][-1]["endSeconds"], 3.0)
        self.assertEqual(truth["melody"], [{"startSeconds": 0.0, "endSeconds": 1.0, "midi": 74},
                                           {"startSeconds": 2.5, "endSeconds": 3.0, "midi": 76}])
        truth, _ = self.derive_quietly(tracks, melody="Lead", duration=4.0, offset=1.0)
        self.assertEqual(truth["chords"], [{"startSeconds": 1.0, "endSeconds": 4.0, "label": "D:sus2"}])
        self.assertEqual(truth["melody"], [{"startSeconds": 1.0, "endSeconds": 2.0, "midi": 74},
                                           {"startSeconds": 3.5, "endSeconds": 4.0, "midi": 76}])
        self.assertTrue(all(t < 4.0 for t in truth["beats"] + truth["downbeats"]))

    def test_times_below_zero_after_the_offset_are_dropped_with_a_warning(self):
        tracks = {"Pad": chord_notes([62, 64, 69], 0, 8) + chord_notes([65, 69, 72], 8, 8),
                  "Bass": [(0, 8, 38), (8, 8, 41)], "Lead": [(0, 1, 74), (4, 1, 76)]}
        truth, stderr = self.derive_quietly(tracks, melody="Lead", offset=-2.0)
        self.assertEqual(truth["beats"][0], 0.0)
        self.assertEqual(truth["downbeats"][0], 0.0)
        self.assertEqual(truth["chords"][0], {"startSeconds": 0.0, "endSeconds": 2.0, "label": "D:sus2"})
        self.assertEqual(truth["melody"], [{"startSeconds": 0.0, "endSeconds": 0.5, "midi": 76}])
        self.assertTrue(all(t >= 0 for t in truth["beats"] + truth["downbeats"]))
        self.assertIn("before 0", stderr)

    def test_a_positive_offset_does_not_warn(self):
        _, stderr = self.derive_quietly({"Pad": chord_notes([62, 64, 69], 0), "Bass": [(0, 4, 38)]}, offset=1.0)
        self.assertEqual(stderr, "")

    def test_help_says_which_way_the_offset_shifts_and_warns_about_compound_meters(self):
        text = " ".join(midi_truth.build_parser().format_help().split())
        self.assertIn("audio time = MIDI time + offset", text)
        self.assertIn("later", text)
        self.assertIn("quarter note", text)
        self.assertIn("6/8", text)
        self.assertIn("quantized", text)

    def test_a_note_starting_inside_a_beat_counts_for_that_beat(self):
        late = [(0.9, 0.05, 62), (0.9, 0.05, 64), (0.9, 0.05, 69)]
        truth = self.derive({"Pad": late, "Bass": [(0, 4, 38)]})
        self.assertEqual([(c["label"], c["startSeconds"], c["endSeconds"]) for c in truth["chords"]],
                         [("D:sus2", 0.0, 0.5), ("N", 0.5, 2.0)])


class BenchmarkTests(unittest.TestCase):
    def test_measure_reports_wall_time_and_the_childs_own_peak_rss(self):
        small = benchmark.measure([sys.executable, "-c", "pass"])
        big = benchmark.measure([sys.executable, "-c", "x = bytearray(150 * 2**20); x[::4096] = b'1' * len(x[::4096])"])
        self.assertEqual((small.returncode, big.returncode), (0, 0))
        self.assertGreater(small.seconds, 0)
        self.assertGreater(big.peak_mb, small.peak_mb + 100)

    def test_synthesized_file_has_the_requested_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.wav"
            benchmark.synthesize(path, seconds=3.5, rate=8000, channels=2)
            info = sf.info(str(path))
            self.assertEqual((info.samplerate, info.channels), (8000, 2))
            self.assertAlmostEqual(info.duration, 3.5, places=3)

    def test_synthesis_streams_in_blocks_and_stays_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.wav", Path(directory) / "b.wav"
            benchmark.synthesize(a, seconds=2.0, rate=8000, channels=1, block_seconds=0.7)
            benchmark.synthesize(b, seconds=2.0, rate=8000, channels=1, block_seconds=2.0)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_the_six_hour_run_is_opt_in(self):
        self.assertEqual([case.label for case in benchmark.select_cases(six_hour=False)], ["4 minute stereo 44.1 kHz"])
        self.assertEqual(len(benchmark.select_cases(six_hour=True)), 2)

    def test_verdict_names_the_budget(self):
        case = benchmark.CASES[0]
        ok = benchmark.Measurement(returncode=0, seconds=case.max_seconds - 1, peak_mb=case.max_mb - 1)
        self.assertEqual(benchmark.verdict(case, ok), "within budget")
        slow = benchmark.Measurement(returncode=0, seconds=case.max_seconds + 1, peak_mb=case.max_mb - 1)
        self.assertIn("time", benchmark.verdict(case, slow))
        failed = benchmark.Measurement(returncode=1, seconds=1, peak_mb=1)
        self.assertIn("exit", benchmark.verdict(case, failed))

    def test_a_child_that_outlives_the_timeout_is_killed_and_reported_failed(self):
        started = time.perf_counter()
        result = benchmark.measure([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
        self.assertLess(time.perf_counter() - started, 20)
        self.assertTrue(result.timed_out)
        self.assertNotEqual(result.returncode, 0)
        case = benchmark.CASES[0]
        self.assertIn("timed out", benchmark.verdict(case, result))

    def test_an_interrupt_kills_the_child_before_propagating(self):
        real_sleep = time.sleep
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pid"
            code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"

            def interrupt(seconds):
                while not (marker.exists() and marker.read_text()):
                    real_sleep(0.01)
                raise KeyboardInterrupt

            with mock.patch.object(benchmark.time, "sleep", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.measure([sys.executable, "-c", code, str(marker)], timeout=60)
            pid = int(marker.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_the_stereo_case_has_decorrelated_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            benchmark.synthesize(path, seconds=6.0, rate=8000, channels=2)
            data, _ = sf.read(str(path))
        self.assertLess(abs(numpy.corrcoef(data[:, 0], data[:, 1])[0, 1]), 0.5)
        self.assertGreater(numpy.std(data[:, 0]), 0.01)

    def test_the_timeout_flag_defaults_to_ten_minutes(self):
        self.assertEqual(benchmark.build_parser().parse_args([]).timeout, 600.0)
        self.assertEqual(benchmark.build_parser().parse_args(["--timeout", "5"]).timeout, 5.0)


if __name__ == "__main__":
    unittest.main()
