"""The feature store: compact per-frame features recorded in the one streaming pass over the audio.

Two time grids. Onset frame j is labelled at decimated sample j * ONSET_HOP, so its time is j / onset_rate. Row k
is the 100 ms of original samples [k * rate // 10, (k + 1) * rate // 10), the LoudnessMeter segment grid, and its
spectrum is one window centred on that span.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..analysis import Framer

FEATURE_RATE = 11025
# Per-channel FFTs cost time linearly in channels, so wider files are folded to this many groups.
MAX_FEATURE_CHANNELS = 8
ONSET_SIZE, ONSET_HOP = 1024, 128
# Log-magnitude flux under a Hann window peaks while an attack is still about 320 samples (29 ms) after the window
# centre, measured on clicks at 44.1 and 48 kHz. Frames are labelled at that point so onset times match attacks.
ONSET_LEAD = 320
ONSET_BAND_EDGES = (150.0, 2000.0)
ROW_RATE = 10
ROW_SIZE = 8192
ROW_BATCH = 32
PCP_RANGE = (65.0, 2100.0)


@dataclass(frozen=True, kw_only=True)
class FeatureStore:
    """Finished features. Arrays are float32 and read-only."""

    onset: np.ndarray  # (frames, 3) positive log-magnitude flux below 150 Hz, 150-2000 Hz and above 2000 Hz
    onset_rate: float
    pcp: np.ndarray  # (rows, 12) linear-bin FFT magnitude per pitch class, 65-2100 Hz, A4 = 440 Hz, untuned
    row_rate: float = float(ROW_RATE)
    seconds: float
    channels: int
    folded: bool

    @property
    def flux(self) -> np.ndarray:
        return self.onset.sum(axis=1, dtype=np.float64)


class _Rows:
    """A float32 table preallocated from the header's frame count and grown only if the header undercounted."""

    def __init__(self, width: int, capacity: int) -> None:
        self.data = np.zeros((max(capacity, 16), width), dtype=np.float32)
        self.length = 0

    def extend(self, values: np.ndarray) -> None:
        end = self.length + len(values)
        if end > len(self.data):
            grown = np.zeros((max(end, len(self.data) * 3 // 2), self.data.shape[1]), dtype=np.float32)
            grown[:self.length] = self.data[:self.length]
            self.data = grown
        self.data[self.length:end] = values
        self.length = end

    def freeze(self) -> np.ndarray:
        frozen = self.data if self.length == len(self.data) else self.data[:self.length].copy()
        frozen.flags.writeable = False
        return frozen


class FeatureRecorder:
    """Decimates to about 11 kHz and averages channel FFT magnitudes, so independently inverted channels survive.

    Blocks are samples by channels. More than MAX_FEATURE_CHANNELS channels are averaged into that many groups,
    channel i into group i mod MAX_FEATURE_CHANNELS, so adjacent channels such as a stereo pair stay apart.
    """

    def __init__(self, rate: int, channels: int, expected_frames: int) -> None:
        self.rate, self.channels, self.frames = rate, channels, 0
        self.factor = max(1, rate // FEATURE_RATE)
        self.decimated_rate = rate / self.factor
        # A history of whole output periods keeps every block start on the decimation grid.
        self.history = 8 * self.factor
        self.lowpass = signal.firwin(self.history + 1, 0.8 / self.factor)
        self.pending: np.ndarray | None = None
        self.fold = None
        groups = channels
        if channels > MAX_FEATURE_CHANNELS:
            index = np.arange(channels) % MAX_FEATURE_CHANNELS
            self.fold = np.eye(MAX_FEATURE_CHANNELS)[index] / np.bincount(index)
            groups = MAX_FEATURE_CHANNELS
        decimated = expected_frames // self.factor + 1

        self.onset_framer = Framer(ONSET_SIZE, ONSET_HOP)
        self.onset_framer.push(np.zeros((ONSET_SIZE // 2 + ONSET_LEAD, groups)))
        self.onset_window = signal.windows.hann(ONSET_SIZE, sym=False)
        band = np.searchsorted(ONSET_BAND_EDGES, np.fft.rfftfreq(ONSET_SIZE, 1 / self.decimated_rate), side="right")
        self.onset_bands = np.eye(3)[band]
        self.previous: np.ndarray | None = None
        self.onset = _Rows(3, decimated // ONSET_HOP + 2)

        self.row_window = signal.windows.hann(ROW_SIZE, sym=False)
        freqs = np.fft.rfftfreq(ROW_SIZE, 1 / self.decimated_rate)
        audible = (freqs >= PCP_RANGE[0]) & (freqs <= min(PCP_RANGE[1], 0.4 * self.decimated_rate))
        self.pcp_bins = np.flatnonzero(audible)
        self.pcp_matrix = np.eye(12)[np.round(69 + 12 * np.log2(freqs[audible] / 440.0)).astype(int) % 12]
        # Decimated samples from absolute index `buffer_start` on; the leading zeros centre row 0's window.
        self.buffer = np.zeros((ROW_SIZE // 2, groups))
        self.buffer_start = -ROW_SIZE // 2
        self.next_row = 0
        self.pcp = _Rows(12, -(-expected_frames * ROW_RATE // rate) + 1)

    def feed(self, block: np.ndarray) -> None:
        self.frames += len(block)
        if self.fold is not None:
            block = block @ self.fold
        samples = self._decimate(block) if self.factor > 1 else block
        self._onsets(self.onset_framer.push(samples))
        self.buffer = np.concatenate([self.buffer, samples])
        self._rows(self._row_starts(None))

    def finish(self) -> FeatureStore:
        self._onsets(self.onset_framer.flush())
        self.buffer = np.concatenate([self.buffer, np.zeros((ROW_SIZE, self.buffer.shape[1]))])
        self._rows(self._row_starts(-(-self.frames * ROW_RATE // self.rate)))
        return FeatureStore(onset=self.onset.freeze(), onset_rate=self.decimated_rate / ONSET_HOP,
                            pcp=self.pcp.freeze(), seconds=self.frames / self.rate, channels=self.channels,
                            folded=self.fold is not None)

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
        magnitude = np.log1p(100 * np.mean(np.abs(np.fft.rfft(frames * self.onset_window, axis=-1)), axis=1))
        if self.previous is None:
            self.previous = magnitude[0]
        diffs = np.diff(np.vstack([self.previous, magnitude]), axis=0)
        self.onset.extend(np.maximum(diffs, 0.0) @ self.onset_bands)
        self.previous = magnitude[-1]

    def _row_starts(self, rows: int | None) -> np.ndarray:
        """Window starts, as absolute decimated indices, of the next rows whose window is complete.

        With `rows`, the stream has ended and exactly the rows before that index are wanted.
        """
        available = self.buffer_start + len(self.buffer)
        last = rows if rows is not None else (available * self.factor * ROW_RATE) // self.rate + 1
        k = np.arange(self.next_row, max(self.next_row, last))
        twice_centre = k * self.rate // ROW_RATE + (k + 1) * self.rate // ROW_RATE
        starts = (twice_centre + self.factor) // (2 * self.factor) - ROW_SIZE // 2
        return starts if rows is not None else starts[starts + ROW_SIZE <= available]

    def _rows(self, starts: np.ndarray) -> None:
        # Batches of ROW_BATCH windows bound the transient copy at low sample rates and many channel groups.
        for begin in range(0, len(starts), ROW_BATCH):
            batch = starts[begin:begin + ROW_BATCH]
            frames = self.buffer[(batch - self.buffer_start)[:, None] + np.arange(ROW_SIZE)]
            spectrum = np.fft.rfft(frames * self.row_window[:, None], axis=1)
            magnitude = np.mean(np.abs(spectrum), axis=2)
            self.pcp.extend(magnitude[:, self.pcp_bins] @ self.pcp_matrix)
        self.next_row += len(starts)
        keep = self.next_row * self.rate // ROW_RATE // self.factor - ROW_SIZE
        if keep > self.buffer_start:
            self.buffer = self.buffer[keep - self.buffer_start:]
            self.buffer_start = keep


def bin_sum(values: np.ndarray, rate: float, edges: np.ndarray) -> np.ndarray:
    """Sum of the rows of `values` (row i at time i / rate) falling between each pair of consecutive edges in seconds."""
    index = np.clip(np.ceil(np.asarray(edges) * rate - 1e-9).astype(int), 0, len(values))
    cumulative = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0, dtype=np.float64)])
    return cumulative[index[1:]] - cumulative[index[:-1]]


def projected_bytes(seconds: float, rate: int) -> int:
    """Size of the finished store for a file of this length and sample rate."""
    onset_frames = seconds * rate / max(1, rate // FEATURE_RATE) / ONSET_HOP
    return int(4 * (3 * onset_frames + 12 * seconds * ROW_RATE))
