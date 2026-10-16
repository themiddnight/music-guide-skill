from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from ..model import Diagnostic, Evidence
from .common import MAX_AUDIO_BYTES

def audio(path: Path, source_id: str, evidence: Evidence, scope: str = "mixdown") -> None:
    if path.suffix.lower() != ".wav":
        raise ValueError("mixdown must be WAV (PCM or IEEE float)")
    if path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("mixdown exceeds 4 GiB limit")
    with sf.SoundFile(path) as file:
        if file.format not in {"WAV", "WAVEX"} or not (file.subtype.startswith("PCM_") or file.subtype in {"FLOAT", "DOUBLE"}):
            raise ValueError("WAV subtype must be PCM or IEEE float")
        if file.frames <= 0 or file.channels <= 0 or file.samplerate <= 0:
            raise ValueError("mixdown has no readable audio frames")
        peak = 0.0
        sum_squares = 0.0
        sample_count = 0
        threshold_count = 0
        for block in file.blocks(blocksize=65536, dtype="float64", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("mixdown contains nonfinite samples")
            absolute = np.abs(block)
            if float(np.max(absolute)) > 1e150:
                raise ValueError("mixdown sample magnitude exceeds finite measurement range")
            peak = max(peak, float(np.max(absolute)))
            sum_squares += float(np.sum(block * block))
            if not math.isfinite(sum_squares):
                raise ValueError("mixdown energy exceeds finite measurement range")
            sample_count += block.size
            threshold_count += int(np.count_nonzero(absolute >= 1.0))
        rms = math.sqrt(sum_squares / sample_count)
        if peak > 0 and rms == 0:
            raise ValueError("audio energy is below finite measurement range")
        facts = [
            ("audio.duration", file.frames / file.samplerate, "seconds"),
            ("audio.channels", file.channels, "count"),
            ("audio.sample_rate", file.samplerate, "Hz"),
            ("audio.sample_peak", None if peak == 0 else 20 * math.log10(peak), "dBFS"),
            ("audio.rms", None if rms == 0 else 20 * math.log10(rms), "dBFS"),
            ("audio.samples_at_or_above_full_scale", threshold_count, "samples"),
        ]
    for key, value, unit in facts:
        evidence.add(source_id, key, value, unit, scope, "static",
                     "whole file", "soundfile+numpy", "measured", "high",
                     "Calculated from decoded WAV samples; dBFS null means digital silence." if value is None else
                     "Directly measured from decoded WAV samples or file header.")
    if peak == 0:
        evidence.diagnostics.append(Diagnostic("digital_silence", "info",
            "The mixdown is digital silence; peak and RMS dBFS are null.", source_id))
    if threshold_count:
        evidence.diagnostics.append(Diagnostic("possible_clipping", "warning",
            f"{threshold_count} sample(s) reach or exceed 0 dBFS. This is a potential clipping indicator, not proof of audible distortion.", source_id))


