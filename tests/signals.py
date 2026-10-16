"""Synthetic signals whose musical answers are known by construction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "music-guide" / "scripts"))

from music_guide.estimates.store import FeatureRecorder, FeatureStore  # noqa: E402

RATE = 48000


def tone(hz: float, seconds: float, rate: int = RATE, amplitude: float = 0.2, harmonics: int = 1) -> np.ndarray:
    """A sine, or an additive tone whose harmonic h has amplitude 1/h, faded in and out over 10 ms."""
    n = np.arange(int(round(seconds * rate)))
    wave = sum(np.sin(2 * np.pi * hz * h * n / rate) / h for h in range(1, harmonics + 1) if hz * h < rate / 2)
    fade = np.minimum(1, np.minimum(n, n[::-1]) / (0.01 * rate))
    return amplitude * fade * wave


def midi_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _burst(rate: int, hz: float | None, seconds: float, seed: int) -> np.ndarray:
    n = np.arange(int(seconds * rate))
    decay = np.exp(-n / (0.012 * rate))
    if hz is None:
        return np.random.default_rng(seed).standard_normal(len(n)) * decay
    return np.sin(2 * np.pi * hz * n / rate) * np.exp(-n / (0.05 * rate))


def click_track(bpm: float, beats_per_bar: int, seconds: float, swing: float = 1.0, rate: int = RATE, *,
                accent: bool = True, offbeats: bool = False, pickup: int = 0, end_bpm: float | None = None,
                start: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clicks on every beat, with a low kick and a louder click on each downbeat when `accent` is set.

    `offbeats` adds a quieter click between beats at long:short ratio `swing` (1.0 straight, 2.0 triplet swing).
    `pickup` beats come before the first downbeat. `end_bpm` ramps the tempo linearly across the file.
    Returns the mono samples, the beat times and the downbeat times in seconds.
    """
    samples = np.zeros(int(seconds * rate))
    click = _burst(rate, None, 0.04, 1)
    kick = _burst(rate, 60.0, 0.2, 2)
    beats, time = [], start
    while time < seconds - 0.3:
        beats.append(time)
        local = bpm if end_bpm is None else bpm + (end_bpm - bpm) * time / seconds
        time += 60.0 / local
    beats = np.array(beats)
    downbeat = (np.arange(len(beats)) - pickup) % beats_per_bar == 0

    def place(at: float, sound: np.ndarray, gain: float) -> None:
        first = int(round(at * rate))
        part = sound[:len(samples) - first]
        samples[first:first + len(part)] += gain * part

    for time, strong in zip(beats, downbeat):
        place(time, click, 0.5 if strong and accent else 0.3)
        if strong and accent:
            place(time, kick, 0.6)
    if offbeats:
        for here, after in zip(beats[:-1], beats[1:]):
            place(here + (after - here) * swing / (1 + swing), click, 0.15)
    return samples, beats, beats[downbeat]


def progression(chords: list[tuple[int, ...]], beats_per_chord: int, bpm: float, rate: int = RATE,
                repeats: int = 1) -> np.ndarray:
    """Each chord is a tuple of MIDI notes, sounded as 6-harmonic tones over a bass an octave below its lowest note."""
    length = 60.0 / bpm * beats_per_chord
    sounds = []
    for notes in chords:
        voices = [tone(midi_hz(note), length, rate, 0.1, harmonics=6) for note in notes]
        voices.append(tone(midi_hz(min(notes) - 12), length, rate, 0.15, harmonics=6))
        sounds.append(np.sum(voices, axis=0))
    return np.concatenate(sounds * repeats)


def record(samples: np.ndarray, rate: int, chunk: int = 4096) -> FeatureStore:
    block = samples if samples.ndim == 2 else samples[:, None]
    recorder = FeatureRecorder(rate, block.shape[1], len(block))
    for begin in range(0, len(block), chunk):
        recorder.feed(block[begin:begin + chunk])
    return recorder.finish()
