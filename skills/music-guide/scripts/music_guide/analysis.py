"""Streaming framing, long-term spectrum and stereo meters."""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

BANDS = (("sub", 0.0, 60.0), ("low", 60.0, 250.0), ("low_mid", 250.0, 500.0), ("mid", 500.0, 2000.0),
         ("high_mid", 2000.0, 6000.0), ("high", 6000.0, math.inf))


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
