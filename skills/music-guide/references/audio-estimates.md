# Audio estimate contract

The builtin backend uses one streaming decode to collect compact frame features. Each analyzer reads these features, not the audio again. The raw pitch-class profile remains a measured spectral summary. Musical interpretations are `inferred`, have `low` confidence, and state their method and limitations in `confidenceReason`.

Select the backend with `--analysis-backend builtin`. No pretrained model or remote service runs. Project-only requests do not load the numerical audio runtime. A failed analyzer produces `estimate_failed` without discarding successful measurements. Insufficient evidence produces `estimate_skipped`.

| Property | Interpretation | Main limitation |
| --- | --- | --- |
| `audio.beat_grid` | Dynamic-programming pulse positions from spectral attacks and tempo candidates | Half or double tempo, sparse attacks, and expressive timing can change the chosen pulse |
| `audio.meter_estimate` | Accent-based meter and downbeat phase candidates | A repeating accent pattern does not prove a written time signature |
| `audio.swing_ratio` | Relative offbeat timing | Sparse or ambiguous subdivisions can make swing unavailable |
| `audio.chord_timeline` | Template and Viterbi chord estimates from harmonic-suppressed chroma | Dense mixes, detuning, non-chord tones, inversions, and missing roots can change labels |
| `audio.rhythm` | Attack density, syncopation, pulse clarity, microtiming, and a formula-based danceability score | Attack detection includes percussion and pitched sound together; scores do not measure a listener's desire to dance |
| `audio.structure` | Feature-change boundaries and repeated section letters | Similar sound does not prove the same composition or a verse/chorus role |
| `audio.energy_timeline` | Weighted energy, spectral brightness, and attack density by segment | Signal energy does not establish perceived excitement or balance of individual instruments |
| `audio.mood_estimate` | Ranked alternatives from normalized energy, tempo, brightness, attack density, and harmony | An untrained heuristic does not recognize emotion or genre |
| `audio.melody` | Predominant pitch contour, range, and voiced share | Accompaniment or bass can be chosen as the melody in a polyphonic mix |

## Harmonic interpretation

Chord qualities cover major, minor, suspended second, suspended fourth, diminished, major seventh, minor seventh, dominant seventh, and no chord. Suspended chords can have identical pitch-class sets under different roots. Read alternatives rather than treating the first label as uniquely identified. Bass estimates remain low confidence.

When no beat grid is available, harmonic and melody summaries use fixed 0.5-second windows and state that basis. No inferred beat positions are silently substituted. MIDI and project chords remain independent stored intent. Compare only after checking render version, initial silence, tempo maps, and timing offsets.

Key estimates retain the major/minor chroma method unless an evaluated replacement improves the available checks. The key vocabulary cannot name Hirajoshi or other modal scales. A borrowed chord is not sufficient evidence of modulation.

## Rhythm and structure

Audio syncopation quantizes detected attacks against an inferred meter. Stored-note rhythm uses available note positions and explicit meter information. Neither establishes note duration or performance articulation from attacks alone. Missing or conflicting stored meter prevents bar-based claims rather than silently assuming 4/4.

Structure uses bounded comparisons between chroma and timbre features, then assigns letters by similarity. Files shorter than 45 seconds do not receive a structure estimate. Novelty peaks and section boundaries are approximate. Do not rename A and B to verse and chorus without additional evidence.

## Energy and mood

Arousal averages the available normalized loudness, attack density, tempo, and brightness terms. Valence uses available harmony, brightness, and tempo terms. Label candidates are ranked by distance to four quadrant centers. The report states the actual inputs and formula. Missing terms remain null. Scores and distances are not calibrated probabilities.

Present the best two mood labels with alternatives and a low-confidence qualification. The user's description takes precedence over the heuristic. Segment energy uses K-weighted mean-square measurements when the sample rate supports that filter. It does not relabel unweighted RMS as LUFS.

## Output and resource limits

Summary output limits chord segments and melody contour points to 256 and structure and energy segments to 64. Full lists are limited to 50,000 entries. The report emits `estimate_truncated` whenever an estimate discards reportable entries. Inspect that diagnostic before comparing whole-song metrics.

Melody features cover at most the first 30 minutes, including in full mode. Longer input produces a truncation diagnostic and reports its covered duration. Audio input retains the existing six-hour and 4 GiB limits. These limits do not promise a runtime or memory budget on every computer. See the verification record for actual measurements.

A supplied stem is measured and analyzed under its own source scope. Its contour can isolate a part, but the tool does not verify that the stem contains the lead. No automatic stem alignment or separation runs.
