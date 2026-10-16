"""Streaming spectral, stereo and musical-feature meters plus low-confidence estimators."""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

BANDS = (("sub", 0.0, 60.0), ("low", 60.0, 250.0), ("low_mid", 250.0, 2000.0),
         ("high_mid", 2000.0, 6000.0), ("high", 6000.0, math.inf))
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Krumhansl and Kessler (1982) probe-tone ratings, tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FEATURE_RATE = 11025
TEMPO_RANGE = (60.0, 200.0)
MIN_TEMPO_SECONDS = 8.0
MIN_KEY_SECONDS = 2.0


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
        correlation = self.cross / math.sqrt(self.left * self.right) if self.left > 0 and self.right > 0 else None
        ratio = 10 * math.log10(self.side / self.mid) if self.side > 0 and self.mid > 0 else None
        return correlation, ratio


class MusicFeatures:
    """Downmixes and decimates to about 11 kHz, then keeps an onset envelope and a chroma sum.

    Only these compact features are retained, so memory stays bounded for long files.
    """

    def __init__(self, rate: int) -> None:
        self.factor = max(1, rate // FEATURE_RATE)
        self.rate = rate / self.factor
        # A history of whole output periods keeps every block start on the decimation grid.
        self.history = 8 * self.factor
        self.lowpass = signal.firwin(self.history + 1, 0.8 / self.factor)
        self.pending = np.zeros(self.history)
        self.onset = Framer(1024, 256)
        self.onset_window = signal.windows.hann(1024, sym=False)
        self.previous: np.ndarray | None = None
        self.flux: list[np.ndarray] = []
        self.chroma_frames = Framer(8192, 4096)
        self.chroma_window = signal.windows.hann(8192, sym=False)
        freqs = np.fft.rfftfreq(8192, 1 / self.rate)
        audible = (freqs >= 65.0) & (freqs <= min(2100.0, 0.4 * self.rate))
        self.chroma_bins = np.flatnonzero(audible)
        self.chroma_class = np.round(69 + 12 * np.log2(freqs[audible] / 440.0)).astype(int) % 12
        self.chroma = np.zeros(12)
        self.samples = 0

    def feed(self, block: np.ndarray) -> None:
        mono = block @ np.full(block.shape[1], 1 / block.shape[1])
        if self.factor > 1:
            mono = self._decimate(mono)
        self.samples += len(mono)
        self._onsets(self.onset.push(mono))
        self._chroma(self.chroma_frames.push(mono))

    def _decimate(self, mono: np.ndarray) -> np.ndarray:
        buffer = np.concatenate([self.pending, mono])
        outputs = (len(buffer) - 1) // self.factor + 1
        decimated = signal.upfirdn(self.lowpass, buffer, down=self.factor)[self.history // self.factor:outputs]
        self.pending = buffer[outputs * self.factor - self.history:]
        return decimated

    def _onsets(self, frames: np.ndarray) -> None:
        if not frames.size:
            return
        magnitude = np.log1p(100 * np.abs(np.fft.rfft(frames * self.onset_window, axis=-1)))
        if self.previous is None:
            self.previous = magnitude[0]
        diffs = np.diff(np.vstack([self.previous, magnitude]), axis=0)
        self.flux.append(np.sum(np.maximum(diffs, 0.0), axis=1))
        self.previous = magnitude[-1]

    def _chroma(self, frames: np.ndarray) -> None:
        if frames.size:
            magnitude = np.abs(np.fft.rfft(frames * self.chroma_window, axis=-1))[:, self.chroma_bins]
            self.chroma += np.bincount(self.chroma_class, weights=np.sum(magnitude, axis=0), minlength=12)

    def finish(self) -> None:
        self._chroma(self.chroma_frames.flush())

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
        return estimate_key(self.chroma)


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


def estimate_key(chroma: np.ndarray) -> dict | None:
    if not np.std(chroma) > 0:
        return None
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            correlation = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            scores.append((correlation, PITCH_CLASSES[tonic], mode))
    scores.sort(reverse=True)
    return {"candidates": [{"tonic": tonic, "mode": mode, "correlation": round(score, 3)}
                           for score, tonic, mode in scores[:3]]}
