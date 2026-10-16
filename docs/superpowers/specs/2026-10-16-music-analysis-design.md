# Music analysis design

Approved in chat on 2026-10-16. Add audio analysis for chords, beat, downbeat, meter, swing, rhythm and syncopation, groove and danceability, song structure, energy, mood and melody. Add rhythm measures for stored notes in MIDI and project files. Every result is evidence that the skill interprets, not a verdict.

## Decisions made in chat

- Dependencies stay light: numpy, scipy and soundfile only. A backend slot (`--analysis-backend`) lets a heavier model such as madmom or BTC plug in later. Only `builtin` ships now.
- Chord vocabulary: major, minor, sus2, sus4, diminished, major seventh, minor seventh, dominant seventh, and no chord (`N`). The bass note is reported at low confidence.
- Mood is reported as measured descriptors, arousal and valence scores, and mood labels. Labels are `inferred`, low confidence, name the formula, and list alternatives. `AGENTS.md`, `SKILL.md` and `references/domains.md` change to allow this.
- In scope as well: song structure, meter, downbeat, swing, danceability and groove, and melody.

## Constraints kept from the existing design

- Audio is read once as a stream. Memory stays bounded at the existing 6 hour cap.
- All estimates are `inferred`. Stored project facts stay `extracted`. Estimates never raise automatic conflicts.
- Files are data. Nothing is uploaded or executed.
- Summary output stays bounded. `--detail full` adds per-beat and per-frame detail.
- Warnings are never silent. Every limit that drops data produces a diagnostic.

## Architecture

### Feature store

The audio stream feeds one `FeatureStore`. It keeps compact per-frame features at a fixed frame rate, as float32 arrays that grow by blocks:

- onset strength, about 86 frames per second at the existing 11 kHz feature rate;
- treble and bass chroma from an NNLS-style harmonic-suppressed log-frequency spectrum, about 10 frames per second, with a global tuning estimate;
- frame loudness and band energies for energy and timbre;
- pitch salience for melody, about 10 frames per second over roughly five octaves, kept only for the first 30 minutes.

Analyzers are pure functions over the finished store. No analyzer reads audio. The existing whole-file accumulators (loudness, spectrum, stereo) stay as they are. The existing chroma cells and onset envelope move into the store, so key and tempo read the same features as the new analyzers.

### Analyzers

Each analyzer takes the store plus the outputs of the analyzers it depends on, and returns facts and diagnostics. The dependency order is fixed:

1. **Beat grid.** Tempo candidates from onset autocorrelation, then beat positions by dynamic programming (Ellis 2007). Meter (2, 3, 4 or 6 beats per bar, reported as 2/4, 3/4, 4/4 or 6/8) and downbeat phase from bass-onset and chroma-change accents per beat position. Swing ratio from onset positions between beats.
2. **Chords.** Beat-synchronous treble and bass chroma, scored against the chord templates, decoded by Viterbi with a high self-transition. Output is a chord timeline in seconds and beats, with the bass note when it differs from the root.
3. **Key.** The current Viterbi key timeline stays the baseline. A chord-aware variant scores keys from chord functions. It replaces the baseline only if it does at least as well on the synthetic tests and the three reference songs.
4. **Rhythm and groove.** Onset density, Longuet-Higgins and Lee syncopation over the beat grid, pulse clarity, microtiming spread, and a danceability score that names its formula.
5. **Structure.** Beat-synchronous chroma and timbre self-similarity, a novelty curve, and boundaries snapped to downbeats. Segments get letters A, B, C by similarity. Letters only, never "verse" or "chorus".
6. **Energy and mood.** An energy timeline per segment and per bar, brightness, arousal and valence scores, and mood labels from four groups: energetic or uplifting, tense or aggressive, calm or peaceful, sad or dark. The two best labels are reported with the scores that produced them.
7. **Melody.** Predominant pitch from harmonic salience, tracked by Viterbi with voicing. Output is pitch range, median pitch, voiced share, and a contour reduced per beat. With `--stem`, each stem gets its own melody estimate, which is far more reliable.

### Backend slot

`--analysis-backend` selects a registry entry. A backend may replace the beat, chord or melody analyzer and must return the same fact shapes. `builtin` is the only entry. An unknown name is a usage error.

### Stored notes

MIDI and project note summaries gain per-track rhythm measures from note starts in beats: onset density per bar, syncopation against the stored meter, and a swing ratio. These are `extracted` measures of stored intent.

## Output

New properties, all `inferred` with low confidence unless stated:

| Property | Summary value | Full detail adds |
|---|---|---|
| `audio.beat_grid` | tempo, beat count, first beat, tempo stability | every beat time |
| `audio.meter_estimate` | meter candidates, downbeat phase | every downbeat time |
| `audio.swing_ratio` | ratio and the beats it used | |
| `audio.chord_timeline` | merged chord segments, capped | per-beat chords |
| `audio.chord_summary` | chord share by duration, major and minor share | |
| `audio.rhythm` | onset density, syncopation, pulse clarity, microtiming, danceability | |
| `audio.structure` | lettered segments with times | novelty curve |
| `audio.energy_timeline` | energy and brightness per segment | per bar |
| `audio.mood_estimate` | arousal, valence, two labels with alternatives | |
| `audio.melody` | range, median, voiced share, contour per beat | per frame pitch |
| `project.track_rhythm` (`extracted`) | density, syncopation, swing per track | |

Each reason names the method, what it cannot see, and the limits that applied.

## Limits and failure handling

- Files shorter than the minimum each analyzer needs give no fact and an `estimate_skipped` diagnostic naming the reason.
- Melody stops at 30 minutes with a diagnostic. Summary timelines are capped with a truncation warning.
- Budget for a 4 minute stereo mixdown: under 15 s total and under 400 MB resident, against today's 3.1 s and 129 MB on Winterstory.
- Budget at the 6 hour cap: memory stays under 1 GB.

## Evaluation

- Synthetic signals with known answers for every analyzer: click and drum patterns at known tempo, meter and swing; triad and seventh progressions; syncopated and straight patterns; sections that repeat; a sung or synthesized melody line.
- Reference songs, read-only. Ground truth given by the user: Winterstory is E major, then one bar of D major, then G major to the end, at 100 BPM in Cubase; itipiso is D minor throughout, 92 BPM and 4/4 in Logic; the trance track is D Hirajoshi at 128 BPM, its SOLNA file stores the chords, and its first chord is Dsus2.
- A development script compares reports with a ground truth file and prints chord overlap, tempo error and key agreement. The ground truth for the user's songs stays outside the repository.
- `docs/verification.md` records the measured numbers, including failures.

## Delivery

Four phases, each tested and committed on `feat/music-guide-mvp` before the next starts:

1. Feature store, beat grid, meter, downbeat and swing.
2. Chords and the chord-aware key experiment.
3. Rhythm, groove, danceability, stored-note rhythm, and structure.
4. Energy, mood, melody, the backend slot, docs and the final evaluation.

Nothing is pushed.
