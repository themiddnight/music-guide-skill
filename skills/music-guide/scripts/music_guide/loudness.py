"""Streaming loudness per ITU-R BS.1770-4 / EBU R128 and oversampled true peak."""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

ABSOLUTE_GATE_LUFS = -70.0
MIN_LOUDNESS_RATE = 8000


def k_weighting_sos(rate: int) -> np.ndarray:
    # Analog prototypes fitted to the BS.1770-4 48 kHz biquads (the libebur128 derivation),
    # re-discretised by bilinear transform so the response holds at any sample rate.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
             1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def _lufs(mean_square: np.ndarray | float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return -0.691 + 10 * np.log10(mean_square)


class LoudnessMeter:
    """Accumulates channel-summed K-weighted energy per 100 ms segment.

    Every BS.1770 window (400 ms momentary, 3 s short-term) is a run of whole segments,
    so the 75% gating-block overlap and a 10 Hz short-term rate fall out of one array.
    All channel weights are 1.0, which is exact for mono, L/R and C.
    """

    def __init__(self, rate: int, channels: int) -> None:
        self.rate = rate
        self.sos = k_weighting_sos(rate)
        self.zi = np.zeros((2, 2, channels))
        self.position = 0
        self.next_segment = 1
        self.pending = 0.0
        self.chunks: list[np.ndarray] = []

    def feed(self, block: np.ndarray) -> None:
        filtered, self.zi = signal.sosfilt(self.sos, block, axis=0, zi=self.zi)
        cumulative = np.cumsum(np.sum(filtered * filtered, axis=1))
        start, end = self.position, self.position + len(block)
        last = ((end + 1) * 10 - 1) // self.rate
        if last >= self.next_segment:
            local = np.arange(self.next_segment, last + 1) * self.rate // 10 - start
            ends = cumulative[local - 1]
            sums = np.diff(ends, prepend=0.0)
            sums[0] += self.pending
            self.chunks.append(sums)
            self.pending = float(cumulative[-1] - ends[-1])
            self.next_segment = last + 1
        else:
            self.pending += float(cumulative[-1])
        self.position = end

    def segment_means(self) -> np.ndarray:
        """K-weighted channel-summed mean squares on the recorder's 100 ms grid.

        Include the last partial segment using its actual sample count, so short
        clips and the final feature row keep their measured energy.
        """
        energy = np.concatenate(self.chunks) if self.chunks else np.zeros(0)
        bounds = np.arange(len(energy) + 1) * self.rate // 10
        lengths = np.diff(bounds)
        if self.position > bounds[-1]:
            energy = np.append(energy, self.pending)
            lengths = np.append(lengths, self.position - bounds[-1])
        means = np.divide(energy, lengths, out=np.zeros_like(energy), where=lengths > 0)
        means.flags.writeable = False
        return means

    def windows(self, segments: int) -> tuple[np.ndarray, np.ndarray]:
        """Mean square and start time in seconds of every complete window of n segments."""
        energy = np.concatenate(self.chunks) if self.chunks else np.zeros(0)
        if len(energy) < segments:
            return np.zeros(0), np.zeros(0)
        bounds = np.arange(len(energy) + 1) * self.rate // 10
        sums = np.convolve(energy, np.ones(segments), mode="valid")
        lengths = bounds[segments:] - bounds[:-segments]
        return sums / lengths, bounds[:-segments] / self.rate

    def result(self) -> dict:
        momentary, _ = self.windows(4)
        short_term, starts = self.windows(30)
        return {
            "integrated": _integrated(momentary),
            "max_momentary": _max_lufs(momentary),
            "max_short_term": _max_lufs(short_term),
            "range": _loudness_range(short_term),
            "timeline": _timeline(short_term, starts),
            "has_momentary_block": momentary.size > 0,
            "has_short_term_block": short_term.size > 0,
        }


def _max_lufs(mean_squares: np.ndarray) -> float | None:
    if not mean_squares.size or not np.max(mean_squares) > 0:
        return None
    return float(_lufs(np.max(mean_squares)))


def _integrated(blocks: np.ndarray) -> float | None:
    # BS.1770-4 section 2.8: absolute gate -70 LUFS, then relative gate 10 LU below the
    # loudness of the absolutely gated blocks.
    levels = _lufs(blocks)
    above_absolute = blocks[levels > ABSOLUTE_GATE_LUFS]
    if not above_absolute.size:
        return None
    relative_gate = _lufs(np.mean(above_absolute)) - 10
    gated = blocks[(levels > ABSOLUTE_GATE_LUFS) & (levels > relative_gate)]
    return float(_lufs(np.mean(gated)))


def _loudness_range(short_term: np.ndarray) -> float | None:
    # EBU Tech 3342: -70 LUFS absolute gate, -20 LU relative gate, 10th to 95th percentile.
    levels = _lufs(short_term)
    above_absolute = short_term[levels > ABSOLUTE_GATE_LUFS]
    if not above_absolute.size:
        return None
    relative_gate = _lufs(np.mean(above_absolute)) - 20
    gated = levels[(levels > ABSOLUTE_GATE_LUFS) & (levels > relative_gate)]
    low, high = np.percentile(gated, [10, 95])
    return float(high - low)


def _timeline(short_term: np.ndarray, starts: np.ndarray, limit: int = 64) -> list[dict] | None:
    if not short_term.size:
        return None
    picks = np.unique(np.round(np.linspace(0, len(short_term) - 1, min(limit, len(short_term)))).astype(int))
    levels = _lufs(short_term[picks])
    return [{"startSeconds": round(float(start), 3),
             "lufs": round(float(level), 2) if math.isfinite(level) else None}
            for start, level in zip(starts[picks], levels)]


class TruePeakMeter:
    """Polyphase FIR oversampling (4x, or 2x above 96 kHz), streamed with a carried tail."""

    HALF_TAPS = 8

    def __init__(self, rate: int, channels: int) -> None:
        self.factor = 4 if rate <= 96000 else 2
        taps = signal.firwin(2 * self.HALF_TAPS * self.factor + 1, 1 / self.factor, window=("kaiser", 6.5))
        taps = np.concatenate([taps * self.factor, np.zeros(self.factor - 1)])
        # A cutoff of exactly 1/factor puts the sinc zeros on the input grid, so phase 0
        # reproduces the input samples and only the in-between phases need computing.
        self.phases = [taps[phase::self.factor] for phase in range(1, self.factor)]
        self.tail = np.zeros((2 * self.HALF_TAPS, channels))
        self.peak = 0.0

    def feed(self, block: np.ndarray) -> None:
        samples = np.concatenate([self.tail, block])
        self.tail = samples[-len(self.tail):]
        for channel in samples.T:
            for phase in self.phases:
                self.peak = max(self.peak, float(np.max(np.abs(np.convolve(channel, phase, mode="valid")))))

    def finish(self) -> float:
        self.feed(np.zeros_like(self.tail))
        return self.peak
