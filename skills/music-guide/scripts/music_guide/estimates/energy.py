"""Segment energy descriptors and transparent, deliberately weak mood heuristics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .common import FULL_LIMIT, Method, Report, ReportContext, Skip
from .store import FeatureStore, bin_sum

SUMMARY_SEGMENTS = 64
MOOD_FORMULA = ("arousal = mean of available loudness (-30..-8 LUFS), onset rate (0..8/s), "
                "tempo (60..180 BPM), brightness (500..4000 Hz), each clipped to 0..1; "
                "valence = mean of available major chord share (or key-mode fallback), brightness, tempo; "
                "labels ranked by Euclidean distance to quadrant centers (0.25 or 0.75). "
                "Distances are heuristic ranks, not trained probabilities.")


@dataclass(frozen=True)
class EnergyTimeline:
    segments: tuple[dict, ...]
    detail_intervals: tuple[dict, ...]
    segmented_by: str
    mean_loudness: float | None
    mean_brightness: float | None
    onset_rate: float
    basis: str
    detail_key: str = "perWindow"

    def report(self, ctx: ReportContext) -> Report:
        segments, cut = ctx.cap(self.segments, FULL_LIMIT if ctx.full else SUMMARY_SEGMENTS, "energy segments")
        value = {"segmentedBy": self.segmented_by, "segments": segments}
        cuts = [cut]
        if ctx.full:
            value[self.detail_key], cut = ctx.cap(self.detail_intervals, FULL_LIMIT, "energy detail intervals")
            cuts.append(cut)
        method = Method("segment-loudness-brightness-onsets",
                        "mean K-weighted energy converted by -0.691 + 10*log10(mean energy) per segment, "
                        "feature-band centroid brightness after low-pass decimation near 11 kHz, and spectral-flux "
                        "peak count per second. This brightness omits upper-band energy and differs from the "
                        "native-band audio.spectral_centroid measurement. "
                        "Segment LUFS are ungated descriptors rather than integrated gated program loudness. "
                        f"Detailed intervals use {self.basis}. "
                        "Uniform segments widen beyond 10 s on long files so at most 64 cover the whole recording. "
                        "These measures cannot distinguish emotional intensity from mix level or distortion.")
        return ctx.estimate("audio.energy_timeline", value, "descriptors", method, cuts)


def _lufs(energy: float) -> float | None:
    return round(float(-0.691 + 10 * np.log10(energy)), 3) if energy > 0 else None


def _segments(store, edges: np.ndarray, peaks: np.ndarray, loudness, brightness) -> tuple[dict, ...]:
    durations = np.diff(edges)
    values = []
    if loudness is not None:
        total = bin_sum(loudness, store.row_rate, edges)
        count = bin_sum(np.ones(len(loudness)), store.row_rate, edges)
        loud = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    else:
        loud = None
    if brightness is not None:
        total = bin_sum(brightness, store.row_rate, edges)
        count = bin_sum(np.ones(len(brightness)), store.row_rate, edges)
        bright = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    else:
        bright = None
    counts = np.diff(np.searchsorted(peaks, edges))
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        values.append({"startSeconds": round(float(start), 3), "endSeconds": round(float(end), 3),
                       "loudnessLufs": _lufs(float(loud[i])) if loud is not None else None,
                       "brightnessHz": round(float(bright[i]), 2) if bright is not None else None,
                       "onsetsPerSecond": round(float(counts[i] / durations[i]), 3)})
    return tuple(values)


def energy(store: FeatureStore, *, structure, beats, meter=None) -> EnergyTimeline | Skip:
    if (store.seconds < 2 or not len(store.flux)
            or (not np.any(store.flux > 0) and not np.any(store.pcp > 0))):
        return Skip("energy needs at least 2 s of non-silent audio.")
    loudness = getattr(store, "loudness", None)
    brightness = getattr(store, "brightness", None)
    threshold = max(float(np.mean(store.flux) + 0.5 * np.std(store.flux)), 1e-12)
    indices = signal.find_peaks(store.flux, height=threshold,
                                distance=max(1, int(0.08 * store.onset_rate)))[0]
    if len(store.flux) > 1 and store.flux[0] >= threshold and store.flux[0] > store.flux[1]:
        indices = np.r_[0, indices]
    peaks = indices / store.onset_rate
    peaks = peaks[peaks < store.seconds]
    if isinstance(structure, Skip) or structure is None:
        edges = np.unique(np.r_[np.arange(0, store.seconds, max(10.0, store.seconds / SUMMARY_SEGMENTS)), store.seconds])
        segmented_by = "uniform"
    else:
        edges = structure.boundaries
        segmented_by = "structure"
    if meter is not None and not isinstance(meter, Skip) and len(meter.downbeats):
        downbeats = np.asarray(meter.downbeats)
        bars = np.unique(np.r_[downbeats[(downbeats >= 0) & (downbeats < store.seconds)], store.seconds])
        basis, detail_key = "inferred downbeat intervals", "perBar"
    elif beats is not None and not isinstance(beats, Skip):
        bars = np.unique(np.r_[0.0, beats.times, store.seconds])
        basis, detail_key = "tracked beat intervals because meter was unavailable", "perBeat"
    else:
        bars = np.unique(np.r_[np.arange(0, store.seconds, 2.0), store.seconds])
        basis, detail_key = "fixed 2 s intervals because a beat grid was unavailable", "perWindow"
    mean_loudness = _lufs(float(np.mean(loudness))) if loudness is not None and len(loudness) else None
    mean_brightness = float(np.mean(brightness)) if brightness is not None and len(brightness) else None
    segments = _segments(store, edges, peaks, loudness, brightness)
    if segmented_by == "structure" and getattr(structure, "letters", None) is not None:
        segments = tuple({"letter": letter, **segment} for letter, segment in zip(structure.letters, segments))
    return EnergyTimeline(segments,
                          _segments(store, bars, peaks, loudness, brightness), segmented_by,
                          mean_loudness, mean_brightness, len(peaks) / store.seconds, basis, detail_key)


@dataclass(frozen=True)
class Mood:
    arousal: float
    valence: float
    labels: tuple[str, ...]
    alternatives: tuple[str, ...]
    inputs: dict[str, object]

    def report(self, ctx: ReportContext) -> Report:
        value = {"arousal": round(self.arousal, 3), "valence": round(self.valence, 3),
                 "labels": list(self.labels), "alternatives": list(self.alternatives),
                 "inputs": self.inputs, "formula": MOOD_FORMULA}
        method = Method("descriptor-quadrant-mood-heuristic",
                        "available loudness, onset, tempo, brightness and harmony terms normalized into arousal "
                        "and valence and ranked by distance to four quadrant centers. Inputs include raw descriptors and "
                        "their clipped normalized terms; loudness uses an ungated whole-file energy mean. This is an untrained "
                        "low-confidence heuristic, not emotion recognition or a probability. Lyrics, performance "
                        "intent, culture, context and the listener's response are unavailable; alternatives remain plausible.")
        return ctx.estimate("audio.mood_estimate", value, "heuristic scores", method)


def _scaled(value: float | None, low: float, high: float) -> float | None:
    return None if value is None else round(float(np.clip((value - low) / (high - low), 0, 1)), 4)


def mood(energy: EnergyTimeline | Skip, *, tempo=None, rhythm=None, chords=None, key=None) -> Mood | Skip:
    if isinstance(energy, Skip):
        return Skip(f"mood needs energy descriptors, which were unavailable: {energy.reason}")
    bpm = None if tempo is None or isinstance(tempo, Skip) else float(tempo.candidates[0][0])
    major = None
    if chords is not None and not isinstance(chords, Skip):
        pitched = float(chords.major_share + chords.minor_share)
        if pitched > 0:
            major = float(chords.major_share)
    key_mode = None
    if major is None and key is not None and not isinstance(key, Skip):
        from .harmony import KEYS
        key_mode = 1.0 if KEYS[int(np.argmax(key.scores))][1] == "major" else 0.0
    inputs = {"loudness": _scaled(energy.mean_loudness, -30, -8),
              "onsetRate": _scaled(energy.onset_rate, 0, 8), "tempo": _scaled(bpm, 60, 180),
              "brightness": _scaled(energy.mean_brightness, 500, 4000),
              "majorShare": major, "keyMode": key_mode,
              "loudnessLufs": energy.mean_loudness, "brightnessHz": energy.mean_brightness,
              "onsetsPerSecond": energy.onset_rate, "tempoBpm": bpm,
              "majorShareFrom": "chords" if major is not None else "key" if key_mode is not None else None}
    arousal_terms = [inputs[name] for name in ("loudness", "onsetRate", "tempo", "brightness")
                     if inputs[name] is not None]
    harmony = major if major is not None else key_mode
    valence_terms = [value for value in (harmony, inputs["brightness"], inputs["tempo"]) if value is not None]
    if not arousal_terms or not valence_terms:
        return Skip("mood lacks enough descriptors for arousal and valence; no missing input is assigned a default.")
    arousal, valence = float(np.mean(arousal_terms)), float(np.mean(valence_terms))
    centers = (("energetic/uplifting", 0.75, 0.75), ("tense/aggressive", 0.75, 0.25),
               ("calm/peaceful", 0.25, 0.75), ("sad/dark", 0.25, 0.25))
    labels = sorted(centers, key=lambda item: (arousal - item[1]) ** 2 + (valence - item[2]) ** 2)
    return Mood(arousal, valence, tuple(item[0] for item in labels[:2]),
                tuple(item[0] for item in labels[2:]), inputs)
