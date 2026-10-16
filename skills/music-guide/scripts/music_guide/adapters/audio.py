from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..analysis import MusicFeatures, SpectrumMeter, StereoMeter
from ..loudness import MIN_LOUDNESS_RATE, LoudnessMeter, TruePeakMeter
from ..model import Confidence, Diagnostic, Evidence, Provenance
from .common import MAX_AUDIO_BYTES

LOSSY_SUBTYPES = {"MPEG_LAYER_I", "MPEG_LAYER_II", "MPEG_LAYER_III", "VORBIS", "OPUS"}
BLOCK_FRAMES = 65536
# Each block is held several times over by the meters, so wide files read fewer frames at once.
BLOCK_SAMPLES = 2 ** 20
MIN_BLOCK_FRAMES = 1024
MAX_AUDIO_SECONDS = 6 * 3600
# +120 dBFS. Real recordings never get near it, and every squared sum stays finite below it.
MAX_SAMPLE_MAGNITUDE = 1e6


@dataclass
class Fact:
    property: str
    value: Any
    unit: str
    method: str
    reason: str
    confidence: Confidence = "high"
    provenance: Provenance = "measured"
    time_domain: str = "static"


def block_frames(channels: int) -> int:
    return max(MIN_BLOCK_FRAMES, min(BLOCK_FRAMES, BLOCK_SAMPLES // channels))


def _too_long() -> ValueError:
    return ValueError(f"audio is longer than the {MAX_AUDIO_SECONDS // 3600} hours this tool measures")


def _decibels(value: float) -> float | None:
    return None if value == 0 else 20 * math.log10(value)


def audio(path: Path, source_id: str, evidence: Evidence, scope: str = "mixdown") -> None:
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("audio file exceeds 4 GiB limit")
    try:
        file = sf.SoundFile(path)
    except (sf.LibsndfileError, RuntimeError, TypeError) as error:
        raise ValueError(f"not a decodable audio file: {error}") from error
    with file:
        if file.channels <= 0 or file.samplerate <= 0:
            raise ValueError("audio file has no readable channels or sample rate")
        if file.frames > MAX_AUDIO_SECONDS * file.samplerate:
            raise _too_long()
        try:
            measured = _measure(file)
        except (sf.LibsndfileError, RuntimeError) as error:
            raise ValueError(f"audio decoding failed: {error}") from error
        header = (file.format, file.subtype, file.channels, file.samplerate)
    _report(header, measured, source_id, evidence, scope)


def _measure(file: sf.SoundFile) -> dict:
    rate, channels = file.samplerate, file.channels
    loudness = LoudnessMeter(rate, channels) if rate >= MIN_LOUDNESS_RATE else None
    true_peak = TruePeakMeter(rate, channels)
    spectrum = SpectrumMeter(rate)
    stereo = StereoMeter() if channels == 2 else None
    features = MusicFeatures(rate)
    peak = sum_squares = 0.0
    frames = threshold_count = 0
    for block in file.blocks(blocksize=block_frames(channels), dtype="float64", always_2d=True):
        if not np.isfinite(block).all():
            raise ValueError("audio contains nonfinite samples")
        absolute = np.abs(block)
        block_peak = float(np.max(absolute))
        if block_peak > MAX_SAMPLE_MAGNITUDE:
            raise ValueError("audio sample magnitude exceeds finite measurement range")
        peak = max(peak, block_peak)
        sum_squares += float(np.sum(block * block))
        if not math.isfinite(sum_squares):
            raise ValueError("audio energy exceeds finite measurement range")
        frames += len(block)
        if frames > MAX_AUDIO_SECONDS * rate:
            raise _too_long()
        threshold_count += int(np.count_nonzero(absolute >= 1.0))
        for meter in (loudness, true_peak, spectrum, stereo, features):
            if meter is not None:
                meter.feed(block)
    if frames <= 0:
        raise ValueError("audio file has no readable audio frames")
    rms = math.sqrt(sum_squares / (frames * channels))
    if peak > 0 and rms == 0:
        raise ValueError("audio energy is below finite measurement range")
    features.finish()
    bands, centroid = spectrum.result()
    return {
        "frames": frames, "peak": peak, "rms": rms, "threshold_count": threshold_count,
        "true_peak": max(peak, true_peak.finish()), "oversampling": true_peak.factor,
        "loudness": loudness.result() if loudness else None,
        "bands": bands, "centroid": centroid,
        "stereo": stereo.result() if stereo else None,
        "features": features,
    }


def _report(header: tuple, measured: dict, source_id: str, evidence: Evidence, scope: str) -> None:
    file_format, subtype, channels, rate = header
    lossy = subtype in LOSSY_SUBTYPES or file_format == "MP3"
    signal_confidence: Confidence = "medium" if lossy else "high"
    lossy_note = (" Lossy decode: values describe the decoded signal, sample peak can exceed the"
                  " original, and codec delay or padding can shift timing.") if lossy else ""
    decoded = f"Measured from {file_format}/{subtype} samples decoded by libsndfile."
    facts = _header_facts(header, measured, decoded, signal_confidence, lossy_note)
    facts += _loudness_facts(measured, channels, rate, signal_confidence, lossy_note)
    facts += _balance_facts(measured, signal_confidence, lossy_note)
    facts += _estimate_facts(measured["features"], lossy_note)
    for fact in facts:
        evidence.add(source_id, fact.property, fact.value, fact.unit, scope, fact.time_domain,
                     "whole file", fact.method, fact.provenance, fact.confidence, fact.reason)
    _diagnose(measured, lossy, channels, rate, {fact.property for fact in facts}, source_id, evidence)


def _header_facts(header: tuple, measured: dict, decoded: str, confidence: Confidence, note: str) -> list[Fact]:
    file_format, subtype, channels, rate = header
    peak, rms = measured["peak"], measured["rms"]
    silent = "dBFS null means digital silence."
    basic = "soundfile+numpy"
    return [
        Fact("audio.format", file_format, "label", "libsndfile", "Container format reported by libsndfile."),
        Fact("audio.subtype", subtype, "label", "libsndfile", "Sample encoding reported by libsndfile."),
        Fact("audio.duration", measured["frames"] / rate, "seconds", basic,
             "Decoded frame count divided by sample rate." + note, confidence),
        Fact("audio.channels", channels, "count", basic, "Read from the decoded stream header."),
        Fact("audio.sample_rate", rate, "Hz", basic, "Read from the decoded stream header."),
        Fact("audio.sample_peak", _decibels(peak), "dBFS", basic,
             (silent if peak == 0 else decoded) + note, confidence),
        Fact("audio.rms", _decibels(rms), "dBFS", basic,
             (silent if rms == 0 else decoded + " RMS over all channels' samples, unweighted.") + note, confidence),
        Fact("audio.samples_at_or_above_full_scale", measured["threshold_count"], "samples", basic,
             decoded + note, confidence),
        Fact("audio.true_peak", _decibels(measured["true_peak"]), "dBTP", f"fir-oversample-{measured['oversampling']}x",
             (silent if peak == 0 else f"{measured['oversampling']}x polyphase FIR oversampling (BS.1770-4 Annex 2"
              " approach, own filter); never reported below the sample peak.") + note, confidence),
        Fact("audio.crest_factor", None if peak == 0 else 20 * math.log10(peak / rms), "dB", basic,
             ("Undefined for digital silence." if peak == 0 else "Sample peak dBFS minus RMS dBFS.") + note, confidence),
    ]


def _loudness_facts(measured: dict, channels: int, rate: int, confidence: Confidence, note: str) -> list[Fact]:
    result = measured["loudness"]
    method = "bs1770-4"
    if channels > 2:
        confidence = "low" if confidence == "medium" else "medium"
        note += (" More than two channels: every channel weighted 1.0 because no channel map is read;"
                 " BS.1770 surround weights (1.41 for Ls/Rs, LFE excluded) were not applied.")
    elif channels == 1:
        note += " Mono counted as one channel per BS.1770 (no dual-mono +3 dB)."
    if result is None:
        reason = f"Sample rate {rate} Hz is below the {MIN_LOUDNESS_RATE} Hz floor for K-weighting." + note
        return [Fact(name, None, unit, method, reason, confidence) for name, unit in LOUDNESS_PROPERTIES]
    gated = "400 ms blocks, 75% overlap, -70 LUFS absolute and -10 LU relative gates."
    short = "3 s window every 100 ms."
    values = [
        (result["integrated"], "Integrated loudness, " + gated,
         "No 400 ms block above the -70 LUFS gate (silence, near silence or shorter than 400 ms)."),
        (result["max_momentary"], "Maximum 400 ms momentary loudness.",
         "No complete 400 ms block with energy."),
        (result["max_short_term"], "Maximum short-term loudness, " + short,
         "No complete 3 s window with energy."),
        (result["range"], "EBU Tech 3342 loudness range from short-term values, -70 LUFS and -20 LU gates, 10th-95th percentile.",
         "No 3 s short-term value above the -70 LUFS gate."),
    ]
    facts = [Fact(name, value, unit, method, (reason if value is not None else missing) + note, confidence)
             for (name, unit), (value, reason, missing) in zip(LOUDNESS_PROPERTIES, values)]
    timeline = result["timeline"]
    facts.append(Fact("audio.short_term_loudness_timeline",
                      None if timeline is None else {"points": timeline}, "LUFS", method,
                      ("Short-term loudness (" + short + ") sampled at up to 64 evenly spaced window starts;"
                       " lufs null means a silent window." if timeline else "Shorter than one 3 s window.") + note,
                      confidence, time_domain="seconds"))
    return facts


LOUDNESS_PROPERTIES = (("audio.integrated_loudness", "LUFS"), ("audio.max_momentary_loudness", "LUFS"),
                       ("audio.max_short_term_loudness", "LUFS"), ("audio.loudness_range", "LU"))


def _balance_facts(measured: dict, confidence: Confidence, note: str) -> list[Fact]:
    bands, centroid = measured["bands"], measured["centroid"]
    spectral = "long-term-spectrum"
    silent = ("Undefined for digital silence." if measured["peak"] == 0 else
              "Undefined because the signal is too short or too quiet for a windowed spectrum.")
    facts = [
        Fact("audio.band_energy", bands, "dB_rel", spectral,
             (silent if bands is None else "Share of total energy per band in dB (sub <60 Hz, low 60-250 Hz,"
              " low_mid 250-2000 Hz, high_mid 2-6 kHz, high >6 kHz) from a Hann-windowed 4096-point average power"
              " spectrum summed over channels, DC bin excluded; null band means no energy.") + note, confidence),
        Fact("audio.spectral_centroid", centroid, "Hz", spectral,
             (silent if centroid is None else "Power-weighted mean frequency of the long-term average spectrum.") + note,
             confidence),
    ]
    if measured["stereo"] is not None:
        correlation, ratio = measured["stereo"]
        facts += [
            Fact("audio.stereo_correlation", correlation, "coefficient", "stereo-energy",
                 ("Undefined because a channel is silent." if correlation is None else
                  "Zero-lag L/R correlation, sum(L*R)/sqrt(sum(L^2)*sum(R^2)), no mean removal.") + note, confidence),
            Fact("audio.side_to_mid_ratio", ratio, "dB", "stereo-energy",
                 ("Undefined because mid or side energy is zero (identical, inverted or silent channels)."
                  if ratio is None else "10*log10(side energy / mid energy) with M=(L+R)/2, S=(L-R)/2.") + note,
                 confidence),
        ]
    return facts


def _estimate_facts(features: MusicFeatures, note: str) -> list[Fact]:
    facts = []
    profile = features.pitch_class_profile()
    if profile is not None:
        facts.append(Fact("audio.pitch_class_profile", profile, "ratio", "chroma-fft",
                          "Share of 65-2100 Hz spectral magnitude per pitch class (A4=440 Hz, downmixed, about 11 kHz)."
                          " Overtones fold into other classes, so this is spectral energy, not played notes." + note,
                          "medium"))
    tempo = features.tempo()
    if tempo is not None:
        facts.append(Fact("audio.tempo_estimate", tempo, "BPM", "onset-autocorrelation",
                          "Estimate, not a fact: top autocorrelation peaks of a spectral-flux onset envelope within"
                          " 60-200 BPM. Half- and double-time readings are equally consistent with periodicity alone;"
                          " rubato, swing and tempo changes are averaged away." + note, "low", "inferred"))
    key = features.key()
    if key is not None:
        facts.append(Fact("audio.key_estimate", key, "key", "krumhansl-kessler-chroma",
                          "Estimate, not a fact: correlation of the whole-file chroma with Krumhansl-Kessler major and"
                          " minor profiles. Modal, non-diatonic or pentatonic music (for example Hirajoshi), key changes"
                          " and detuning can be mislabelled; read audio.pitch_class_profile directly." + note,
                          "low", "inferred"))
    return facts


def _diagnose(measured: dict, lossy: bool, channels: int, rate: int, emitted: set[str],
              source_id: str, evidence: Evidence) -> None:
    def add(code: str, severity: str, message: str) -> None:
        evidence.diagnostics.append(Diagnostic(code, severity, message, source_id))

    if lossy:
        add("lossy_decode", "info", "Lossy codec: measurements describe the decoded signal. Sample and true peak can"
            " exceed the original master, and codec delay or padding can add silence and shift timing.")
    if measured["peak"] == 0:
        add("digital_silence", "info", "The audio is digital silence; peak and RMS dBFS are null.")
    if measured["threshold_count"]:
        add("possible_clipping", "warning", f"{measured['threshold_count']} sample(s) reach or exceed 0 dBFS. This is a"
            " potential clipping indicator, not proof of audible distortion.")
    loudness = measured["loudness"]
    if loudness is None:
        add("loudness_unmeasurable", "info", f"Loudness not measured: {rate} Hz is below {MIN_LOUDNESS_RATE} Hz.")
    elif loudness["integrated"] is None:
        add("loudness_unmeasurable", "info", "Integrated loudness is null: no 400 ms block exceeds the -70 LUFS"
            " absolute gate (silence, near silence, or audio shorter than 400 ms).")
    if channels > 2:
        add("channel_weighting", "info", f"{channels} channels: loudness weights every channel 1.0 because no channel"
            " map is read, and stereo metrics are skipped.")
    features: MusicFeatures = measured["features"]
    for name in ("tempo", "key"):
        if f"audio.{name}_estimate" not in emitted:
            add("estimate_skipped", "info", f"No {name} estimate: {features.seconds:.1f} s of audio without enough"
                " usable signal for a periodicity or pitch-class reading.")
