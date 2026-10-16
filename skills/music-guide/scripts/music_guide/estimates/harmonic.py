"""Compact harmonic note salience for the shared streaming feature recorder.

This is a fixed dictionary fit, not a trained transcription model. The first sixty seconds lock the tuning;
only their compact magnitude spectra on a logarithmic frequency grid are buffered, never their audio or full FFTs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks

MIDI = np.arange(33, 106)
MELODY_ROWS = 18_000
TUNING_ROWS = 600
BINS_PER_SEMITONE = 3
NNLS_ITERATIONS = 18


class HarmonicFeatures:
    """A bounded tuning buffer and a nonnegative least-squares harmonic dictionary."""

    def __init__(self, frequencies: np.ndarray) -> None:
        self.frequencies = frequencies
        self.silent_peak = (len(frequencies) - 1) / 2 * 10 ** (-70 / 20)
        highest = min(142.0, 69 + 12 * np.log2(frequencies[-1] / 440))
        self.log_midi = np.arange(30.0, highest, 1 / BINS_PER_SEMITONE)
        hz = 440 * 2 ** ((self.log_midi - 69) / 12)
        bins = hz / frequencies[1]
        self.left = np.minimum(bins.astype(int), len(frequencies) - 2)
        self.fraction = bins - self.left
        self.pending: list[np.ndarray] = []
        self.rows = 0
        self.tuning_vector = 0j
        self.tuning_weight = 0.0
        self.tuning_cents = 0.0
        self.dictionary: csr_matrix | None = None
        self.gram: csr_matrix | None = None

    def feed(self, magnitude: np.ndarray) -> np.ndarray:
        spectrum = magnitude[:, self.left] * (1 - self.fraction) + magnitude[:, self.left + 1] * self.fraction
        spectrum[np.max(magnitude, axis=1) < self.silent_peak] = 0
        if self.dictionary is not None:
            return self._activations(spectrum)
        count = min(len(magnitude), TUNING_ROWS - self.rows)
        for row in magnitude[:count]:
            self._tuning(row)
        self.rows += count
        self.pending.append(spectrum[:count].astype(np.float32))
        if self.rows < TUNING_ROWS:
            return np.empty((0, len(MIDI)), dtype=np.float32)
        initial = self.finish()
        return np.concatenate([initial, self._activations(spectrum[count:])]) if count < len(spectrum) else initial

    def finish(self) -> np.ndarray:
        if self.dictionary is not None:
            return np.empty((0, len(MIDI)), dtype=np.float32)
        if self.tuning_weight and abs(self.tuning_vector) >= 0.3 * self.tuning_weight:
            self.tuning_cents = float(np.angle(self.tuning_vector) * 100 / (2 * np.pi))
        self._dictionary()
        spectra = np.concatenate(self.pending) if self.pending else np.empty((0, len(self.log_midi)))
        self.pending.clear()
        return self._activations(spectra)

    def _tuning(self, magnitude: np.ndarray) -> None:
        peaks = find_peaks(magnitude)[0]
        peaks = peaks[(self.frequencies[peaks] >= 65) & (self.frequencies[peaks] <= 2100)]
        if not len(peaks) or np.max(magnitude) < self.silent_peak:
            return
        peaks = peaks[magnitude[peaks] > np.max(magnitude) * 0.1]
        logarithm = np.log(np.maximum(magnitude, 1e-20))
        curvature = logarithm[peaks - 1] - 2 * logarithm[peaks] + logarithm[peaks + 1]
        offsets = np.divide(0.5 * (logarithm[peaks - 1] - logarithm[peaks + 1]), curvature,
                            out=np.zeros(len(peaks)), where=np.abs(curvature) > 1e-12)
        hz = (peaks + np.clip(offsets, -0.5, 0.5)) * self.frequencies[1]
        midi = 69 + 12 * np.log2(hz / 440)
        weights = magnitude[peaks] ** 2
        vector = np.sum(weights * np.exp(2j * np.pi * midi))
        if abs(vector) >= 0.3 * weights.sum():
            self.tuning_vector += vector
            self.tuning_weight += float(weights.sum())

    def _dictionary(self) -> None:
        dictionary = np.zeros((len(self.log_midi), len(MIDI)))
        for harmonic in range(1, 9):
            partial = MIDI + 12 * np.log2(harmonic) + self.tuning_cents / 100
            distance = (self.log_midi[:, None] - partial) * BINS_PER_SEMITONE
            hz = 440 * 2 ** ((partial - 69) / 12)
            width = np.maximum(0.65, 12 / np.log(2) * BINS_PER_SEMITONE * self.frequencies[1] * 0.7 / hz)
            dictionary += 0.6 ** (harmonic - 1) * np.exp(-0.5 * (distance / width) ** 2)
        # Whitening the templates as well as the signal preserves the dictionary's harmonic amplitude ratios.
        dictionary = np.maximum(dictionary - uniform_filter1d(dictionary, 37, axis=0, mode="nearest"), 0)
        dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0), 1e-12)
        dictionary[dictionary < 1e-5] = 0
        self.dictionary = csr_matrix(dictionary)
        self.gram = csr_matrix(dictionary.T @ dictionary)

    def _activations(self, spectrum: np.ndarray) -> np.ndarray:
        if not len(spectrum):
            return np.empty((0, len(MIDI)), dtype=np.float32)
        whitened = np.maximum(spectrum - uniform_filter1d(spectrum, 37, axis=1, mode="nearest"), 0)
        target = self.dictionary.T.dot(whitened.T).T
        activation = np.maximum(target, 1e-12)
        for _ in range(NNLS_ITERATIONS):
            activation *= target / np.maximum(self.gram.dot(activation.T).T, 1e-12)
        mean = np.mean(spectrum, axis=1)
        flatness = np.exp(np.mean(np.log(np.maximum(spectrum, 1e-12)), axis=1)) / np.maximum(mean, 1e-12)
        activation[flatness > 0.65] = 0
        # Very weak residual fits from mismatched overtone envelopes should not create a new pitch class.
        activation[activation < np.max(activation, axis=1, keepdims=True) * 0.025] = 0
        return activation.astype(np.float32)


def chroma(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bass A1..G#3 and treble A3..A7, with a one-octave crossfade around their shared boundary."""
    bass_weight = np.clip((64 - MIDI) / 12, 0, 1)
    treble_weight = np.clip((MIDI - 45) / 12, 0, 1)
    projection = np.eye(12)[MIDI % 12]
    return (activations * treble_weight) @ projection, (activations * bass_weight) @ projection


def narrow_chroma(activations: np.ndarray) -> np.ndarray:
    """Chord register G2..A5, full C3..C5 and faded above it to reduce interference from high leads."""
    weights = np.interp(MIDI, (43, 48, 72, 82), (0, 1, 1, 0), left=0, right=0)
    return (activations * weights) @ np.eye(12)[MIDI % 12]
