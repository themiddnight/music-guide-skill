"""Streaming spectral, stereo and musical-feature meters plus low-confidence estimators."""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

BANDS = (("sub", 0.0, 60.0), ("low", 60.0, 250.0), ("low_mid", 250.0, 500.0), ("mid", 500.0, 2000.0),
         ("high_mid", 2000.0, 6000.0), ("high", 6000.0, math.inf))
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Krumhansl and Kessler (1982) probe-tone ratings, tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FEATURE_RATE = 11025
# Per-channel FFTs cost time linearly in channels, so wider files are folded to this many groups.
MAX_FEATURE_CHANNELS = 8
TEMPO_RANGE = (60.0, 200.0)
MIN_TEMPO_SECONDS = 8.0
MIN_KEY_SECONDS = 2.0
# Keys are scored per 5 s chroma cell. A key change costs KEY_SWITCH_PENALTY in summed
# correlation, so a new key must outscore the old one over about 25 to 50 s, not for one chord.
# Penalties from 1.0 to 3.0 all matched the three listener-labelled reference songs.
KEY_CELL_SECONDS = 5.0
KEY_SWITCH_PENALTY = 2.0
MIN_KEY_TIMELINE_SECONDS = 30.0
KEYS = tuple((tonic, mode, np.roll(profile, index)) for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR))
             for index, tonic in enumerate(PITCH_CLASSES))


class Framer:
    """Cuts a stream into frames of `size` samples every `hop` samples along axis 0."""

    def __init__(self, size: int, hop: int) -> None:
        self.size, self.hop = size, hop
        self.buffer: np.ndarray | None = None
        self.framed = False

    def push(self, samples: np.ndarray) -> np.ndarray:
        self.buffer = samples if self.buffer is None else np.concatenate([self.buffer, samples])
        if len(self.buffer) < self.size:
            return self._empty()
        count = 1 + (len(self.buffer) - self.size) // self.hop
        frames = sliding_window_view(self.buffer, self.size, axis=0)[::self.hop][:count].copy()
        self.buffer = self.buffer[count * self.hop:]
        self.framed = True
        return frames

    def flush(self) -> np.ndarray:
        """Zero-pads the unframed tail, so a stream shorter than one frame still yields one."""
        covered = self.size - self.hop if self.framed else 0
        if self.buffer is None or len(self.buffer) <= covered:
            return self._empty()
        padded = np.zeros((self.size, *self.buffer.shape[1:]))
        padded[:len(self.buffer)] = self.buffer
        self.buffer = None
        return sliding_window_view(padded, self.size, axis=0)[:1].copy()

    def _empty(self) -> np.ndarray:
        shape = () if self.buffer is None else self.buffer.shape[1:]
        return np.zeros((0, *shape, self.size))


class SpectrumMeter:
    """Long-term average power spectrum summed over channels (Hann, 50% overlap)."""

    def __init__(self, rate: int, size: int = 4096) -> None:
        self.framer = Framer(size, size // 2)
        self.window = signal.windows.hann(size, sym=False)
        self.freqs = np.fft.rfftfreq(size, 1 / rate)
        self.power = np.zeros(len(self.freqs))
        # `power` is stored relative to `scale` squared, so a very loud or very quiet signal never over- or underflows.
        self.scale = 0.0

    def feed(self, block: np.ndarray) -> None:
        self._add(self.framer.push(block))

    def _add(self, frames: np.ndarray) -> None:
        peak = float(np.max(np.abs(frames))) if frames.size else 0.0
        if not peak > 0:
            return
        if peak > self.scale:
            self.power *= (self.scale / peak) ** 2
            self.scale = peak
        spectrum = np.fft.rfft(frames / peak * self.window, axis=-1)
        self.power += np.sum(spectrum.real ** 2 + spectrum.imag ** 2, axis=(0, 1)) * (peak / self.scale) ** 2

    def result(self) -> tuple[dict | None, float | None]:
        self._add(self.framer.flush())
        power, freqs = self.power[1:], self.freqs[1:]
        total = float(np.sum(power))
        if not total > 0:
            return None, None
        shares = {}
        for name, low, high in BANDS:
            energy = float(np.sum(power[(freqs >= low) & (freqs < high)]))
            shares[name] = round(10 * math.log10(energy / total), 2) if energy > 0 else None
        return shares, float(np.sum(freqs * power) / total)


class StereoMeter:
    def __init__(self) -> None:
        self.left = self.right = self.cross = self.mid = self.side = 0.0

    def feed(self, block: np.ndarray) -> None:
        left, right = block[:, 0], block[:, 1]
        self.left += float(left @ left)
        self.right += float(right @ right)
        self.cross += float(left @ right)
        mid, side = (left + right) / 2, (left - right) / 2
        self.mid += float(mid @ mid)
        self.side += float(side @ side)

    def result(self) -> tuple[float | None, float | None]:
        correlation = None
        if self.left > 0 and self.right > 0:
            # Two square roots, not one over the product, so a very quiet or loud pair neither underflows nor overflows.
            correlation = max(-1.0, min(1.0, self.cross / (math.sqrt(self.left) * math.sqrt(self.right))))
        ratio = 10 * math.log10(self.side / self.mid) if self.side > 0 and self.mid > 0 else None
        return correlation, ratio


class MusicFeatures:
    """Decimates each channel to about 11 kHz and averages channel FFT magnitudes for onset and chroma.

    Blocks are samples by channels; frames are frames by channels by samples. Averaging magnitudes
    preserves independently inverted channels. More than MAX_FEATURE_CHANNELS channels are first
    averaged into that many groups, channel i into group i mod MAX_FEATURE_CHANNELS, so adjacent
    channels such as a stereo pair stay apart. Only compact features are retained for long files.
    """

    def __init__(self, rate: int) -> None:
        self.factor = max(1, rate // FEATURE_RATE)
        self.rate = rate / self.factor
        # A history of whole output periods keeps every block start on the decimation grid.
        self.history = 8 * self.factor
        self.lowpass = signal.firwin(self.history + 1, 0.8 / self.factor)
        self.pending: np.ndarray | None = None
        self.onset = Framer(1024, 256)
        self.onset_window = signal.windows.hann(1024, sym=False)
        self.previous: np.ndarray | None = None
        self.flux: list[np.ndarray] = []
        self.chroma_frames = Framer(8192, 4096)
        self.chroma_window = signal.windows.hann(8192, sym=False)
        freqs = np.fft.rfftfreq(8192, 1 / self.rate)
        audible = (freqs >= 65.0) & (freqs <= min(2100.0, 0.4 * self.rate))
        self.chroma_bins = np.flatnonzero(audible)
        chroma_class = np.round(69 + 12 * np.log2(freqs[audible] / 440.0)).astype(int) % 12
        self.chroma_matrix = np.eye(12)[chroma_class]
        self.chroma_cells = np.zeros((0, 12))
        self.chroma_frame_count = 0
        self.samples = 0
        self.channels: int | None = None
        self.fold: np.ndarray | None = None

    def feed(self, block: np.ndarray) -> None:
        block = self._fold(block)
        samples = self._decimate(block) if self.factor > 1 else block
        self.samples += len(samples)
        self._onsets(self.onset.push(samples))
        self._chroma(self.chroma_frames.push(samples))

    def _fold(self, block: np.ndarray) -> np.ndarray:
        if self.channels is None:
            self.channels = block.shape[1]
            if self.channels > MAX_FEATURE_CHANNELS:
                groups = np.arange(self.channels) % MAX_FEATURE_CHANNELS
                self.fold = np.eye(MAX_FEATURE_CHANNELS)[groups] / np.bincount(groups)
        return block if self.fold is None else block @ self.fold

    def _decimate(self, block: np.ndarray) -> np.ndarray:
        if self.pending is None:
            self.pending = np.zeros((self.history, block.shape[1]))
        buffer = np.concatenate([self.pending, block])
        outputs = (len(buffer) - 1) // self.factor + 1
        decimated = signal.upfirdn(self.lowpass, buffer, down=self.factor, axis=0)[self.history // self.factor:outputs]
        self.pending = buffer[outputs * self.factor - self.history:]
        return decimated

    def _onsets(self, frames: np.ndarray) -> None:
        if not frames.size:
            return
        magnitude = np.mean(np.abs(np.fft.rfft(frames * self.onset_window, axis=-1)), axis=1)
        magnitude = np.log1p(100 * magnitude)
        if self.previous is None:
            self.previous = magnitude[0]
        diffs = np.diff(np.vstack([self.previous, magnitude]), axis=0)
        self.flux.append(np.sum(np.maximum(diffs, 0.0), axis=1))
        self.previous = magnitude[-1]

    def _chroma(self, frames: np.ndarray) -> None:
        if frames.size:
            magnitude = np.mean(np.abs(np.fft.rfft(frames * self.chroma_window, axis=-1)), axis=1)
            per_frame = magnitude[:, self.chroma_bins] @ self.chroma_matrix
            centers = (np.arange(len(per_frame)) + self.chroma_frame_count) * self.chroma_frames.hop
            centers = (centers + self.chroma_frames.size / 2) / self.rate
            cells = (centers // KEY_CELL_SECONDS).astype(int)
            if cells[-1] >= len(self.chroma_cells):
                grown = np.zeros((cells[-1] + 1, 12))
                grown[:len(self.chroma_cells)] = self.chroma_cells
                self.chroma_cells = grown
            np.add.at(self.chroma_cells, cells, per_frame)
            self.chroma_frame_count += len(per_frame)

    def finish(self) -> None:
        self._chroma(self.chroma_frames.flush())

    @property
    def chroma(self) -> np.ndarray:
        return np.sum(self.chroma_cells, axis=0)

    @property
    def seconds(self) -> float:
        return self.samples / self.rate

    def pitch_class_profile(self) -> dict | None:
        total = float(np.sum(self.chroma))
        if self.seconds < MIN_KEY_SECONDS or not total > 0:
            return None
        return {name: round(float(value) / total, 4) for name, value in zip(PITCH_CLASSES, self.chroma)}

    def tempo(self) -> dict | None:
        envelope = np.concatenate(self.flux) if self.flux else np.zeros(0)
        frame_rate = self.rate / self.onset.hop
        if self.seconds < MIN_TEMPO_SECONDS or len(envelope) < 2:
            return None
        return estimate_tempo(envelope, frame_rate)

    def key(self) -> dict | None:
        if self.pitch_class_profile() is None:
            return None
        scored = key_scores(self.chroma_cells)
        if scored is None:
            return None
        scores, usable = scored
        return _candidates(np.mean(scores[usable], axis=0))

    def key_timeline(self) -> dict | None:
        if self.seconds < MIN_KEY_TIMELINE_SECONDS or self.pitch_class_profile() is None:
            return None
        return estimate_key_timeline(self.chroma_cells, self.seconds)


def estimate_tempo(envelope: np.ndarray, frame_rate: float) -> dict | None:
    centered = envelope - np.mean(envelope)
    size = 1 << int(2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    acf = np.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2, size)[:len(centered)]
    if not acf[0] > 0:
        return None
    acf /= acf[0]
    shortest = int(60 * frame_rate / TEMPO_RANGE[1]) - 1
    longest = min(len(acf) - 2, int(math.ceil(60 * frame_rate / TEMPO_RANGE[0])) + 1)
    if longest <= shortest + 1:
        return None
    peaks = [lag for lag in signal.find_peaks(acf[:longest + 2])[0] if shortest <= lag <= longest]
    candidates = []
    for lag in peaks:
        before, here, after = acf[lag - 1], acf[lag], acf[lag + 1]
        curvature = before - 2 * here + after
        offset = 0.5 * (before - after) / curvature if curvature < 0 else 0.0
        bpm = 60 * frame_rate / (lag + offset)
        if TEMPO_RANGE[0] <= bpm <= TEMPO_RANGE[1] and here > 0:
            candidates.append((float(here), round(bpm, 1)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0][0]
    top = [{"bpm": bpm, "relativeStrength": round(strength / best, 3)} for strength, bpm in candidates[:3]]
    octave = any(abs(a["bpm"] / b["bpm"] - 2) < 0.06 for a in top for b in top)
    return {"candidates": top, "octaveRelatedCandidates": octave}


def key_scores(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Correlation of each cell's chroma with the 24 Krumhansl-Kessler profiles, and which cells have pitch spread.

    Cells without spread (silence) score 0 for every key, so they never favour a change.
    """
    usable = np.std(cells, axis=1) > 0
    if not np.any(usable):
        return None
    centered = cells[usable] - cells[usable].mean(axis=1, keepdims=True)
    profiles = np.array([profile - profile.mean() for _, _, profile in KEYS])
    scores = np.zeros((len(cells), len(KEYS)))
    scores[usable] = (centered @ profiles.T) / np.outer(np.linalg.norm(centered, axis=1),
                                                        np.linalg.norm(profiles, axis=1))
    return scores, usable


def estimate_key_timeline(cells: np.ndarray, seconds: float) -> dict | None:
    """Viterbi path over cells: stay in a key for free, switch for KEY_SWITCH_PENALTY."""
    scored = key_scores(cells)
    if scored is None:
        return None
    scores, usable = scored
    total, back = scores[0].copy(), []
    for row in scores[1:]:
        switch = total.max() - KEY_SWITCH_PENALTY
        back.append(np.where(total >= switch, np.arange(len(KEYS)), int(total.argmax())))
        total = np.maximum(total, switch) + row
    path = [int(total.argmax())]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    path.reverse()
    runs: list[list[int]] = []
    for index, key in enumerate(path):
        if runs and runs[-1][2] == key:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, key])
    return {"cellSeconds": KEY_CELL_SECONDS,
            "segments": [{"startSeconds": round(start * KEY_CELL_SECONDS, 1),
                          "endSeconds": round(seconds if index == len(runs) - 1 else end * KEY_CELL_SECONDS, 1),
                          **_candidates(np.mean(scores[start:end][usable[start:end]], axis=0))}
                         for index, (start, end, _) in enumerate(runs)]}


def _candidates(scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")[:3]
    return {"candidates": [{"tonic": KEYS[i][0], "mode": KEYS[i][1], "correlation": round(float(scores[i]), 3)}
                           for i in order]}
