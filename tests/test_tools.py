"""Tests for the development tools in `tools/`. They read only the CLI's JSON and MIDI files, never audio."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mido
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
        self.assertEqual(evaluate.overall(metrics), 1.0)

    def test_evaluate_marks_missing_estimates(self):
        report = {"observations": [observation("audio.duration", 8.0)]}
        metrics = evaluate.evaluate(report, TRUTH)
        self.assertTrue(metrics)
        for metric in metrics:
            self.assertIsNone(metric.score, metric)
            self.assertIn("not reported", metric.detail)
        self.assertEqual(evaluate.overall(metrics), 0.0)
        self.assertEqual({m.name for m in metrics}, {m.name for m in evaluate.evaluate(report_from_truth(TRUTH), TRUTH)})

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
            self.assertIn("overall", text)
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


if __name__ == "__main__":
    unittest.main()
