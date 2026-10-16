# Music Analysis Implementation Plan

## Completion record, 2026-10-17

The analyzer families, stored-note rhythm, builtin backend, output caps, and documentation are implemented in 0.3.0. The production key method remains the baseline after the offline chord-function candidate tied all six available checks. Real-reference results, independent review fixes, installation checks, and resource measurements are recorded in [verification](../../verification.md). The detailed checkboxes below preserve the original plan rather than asserting every planned experiment was performed exactly as written.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inferred audio analysis for beat, meter, swing, chords, rhythm, groove, structure, energy, mood and melody. Add extracted rhythm measures for stored notes.

**Architecture:** One streaming `FeatureRecorder` fills a frozen float32 `FeatureStore` on two time grids. A straight-line `analyze()` runs pure analyzers whose parameters declare their dependencies and which return `Result | Skip`. Each result builds its own facts through one estimate factory. A guarded render loop turns outcomes into facts and diagnostics.

**Tech Stack:** Python 3.10+, numpy, scipy, soundfile, unittest.

**Spec:** `docs/superpowers/specs/2026-10-16-music-analysis-design.md`

## Global Constraints

- Runtime dependencies stay numpy, scipy, soundfile, mido, defusedxml and pyaaf2. No new package.
- The audio is streamed once. Builtin analyzers never read audio.
- Every new audio estimate is `provenance: "inferred"`, `confidence: "low"`. Its reason starts with `Estimate, not a fact:`, names the method, and names what it cannot see. `audio.pitch_class_profile` stays `measured` and `medium`.
- Stored-note rhythm is `project.track_rhythm`, `extracted`, at the producing format's grade.
- Summary caps: 256 chord segments, 64 structure and energy segments, 256 melody contour points. Every per-beat or per-frame list in full detail is capped at 50,000. Each cut adds an `estimate_truncated` warning and a sentence in the reason.
- Diagnostic codes: `estimate_skipped` (info), `estimate_failed` (warning), `estimate_truncated` (warning), `rhythm_meter_unknown` (info).
- Melody covers the first 1800 s only.
- Budget: a 4 minute stereo file in under 15 s and 400 MB resident. A 6 hour file in under 1 GB.
- The suite runs clean with `PYTHONWARNINGS=error uv run --with-requirements requirements-dev.txt python -m unittest discover -s tests -q`.
- `claude plugin validate --strict .` passes before every commit.
- The user's reference files are read-only, and the user's truth files stay outside the repository.
- Nothing is pushed.

## Review Focus

1. **Music with no steady pulse** (ambient pads, rubato piano). The beat grid skips with a reason. Chords, structure and melody still run on a fixed 0.5 s basis and say so in the value (`"basis": "fixed"`). Test in Task 5.
2. **Silence, near silence and very short files.** No analyzer produces NaN or fails. Each skips with `estimate_skipped`, and loudness facts survive. Test in Task 1.
3. **Detuned recordings** (a 30 cent offset). Chord roots do not shift, because tuning is estimated from the first 60 s of non-silent audio. Test in Task 4.
4. **Projects whose meter comes after the notes** (MusicXML and MuseScore emit notes before meters) **and projects with meter maps.** Stored-note rhythm uses the finished staged meters, so it never depends on adapter order. Test in Task 8.
5. **A lossy MP3 or a surround file.** The estimates still run, and the existing lossy and channel-fold notes appear in every estimate reason. Test in Task 1.

---

## Design synthesis

Two design candidates were built and cross-judged. The sketches and the verdict are kept outside the repository. This plan takes candidate 1's skeleton because it has the smaller public surface:

- a straight-line `analyze()`;
- required dependencies as positional parameters and optional ones as keyword-only `Outcome` parameters;
- post-adapter stored-note rhythm;
- a plain `Backend` dataclass;
- three-band onset flux.

From candidate 2 it takes:

- each result's own `report(ctx)` through one `ctx.estimate` factory, which removes a hand-synced property table;
- isolation around both running and reporting an analyzer;
- tests that assert `estimate_failed` never appears;
- the fixed-basis fallback when there is no beat grid;
- NNLS note activations shared by chroma and melody;
- tuning locked from the first 60 s;
- per-segment tempo autocorrelation for bounded memory;
- stems running only tempo, beats, key and melody;
- honest tolerances for the value drift that phase 1 causes.

## File map

| File | Responsibility | Task |
|---|---|---|
| `skills/music-guide/scripts/music_guide/model.py` | gains `Fact` (moved from `adapters/audio.py`) and `native()` for numpy scalars | 1 |
| `music_guide/estimates/common.py` | `Skip`, `Limit`, `Method`, `Outcome`, `ReportContext.estimate`, `capped()` | 1 |
| `music_guide/estimates/store.py` | `FeatureRecorder`, `FeatureStore`, grids, `bin_sum`, `beat_sum`, `projected_bytes` | 1, 4, 9, 10 |
| `music_guide/estimates/__init__.py` | `analyze()`, `render()`, `Backend`, `BACKENDS` | 1, 12 |
| `music_guide/estimates/beats.py` | tempo (moved), beat grid, meter and downbeat, swing | 1, 2 |
| `music_guide/estimates/harmony.py` | pitch profile and key (moved), chords, chord-aware key | 1, 5, 6 |
| `music_guide/estimates/groove.py` | audio rhythm and danceability | 7 |
| `music_guide/estimates/structure.py` | lettered structure | 9 |
| `music_guide/estimates/energy.py` | energy timeline and mood | 10 |
| `music_guide/estimates/melody.py` | melody | 11 |
| `music_guide/rhythm.py` | pure Python `Meter`, `BarGrid`, LHL syncopation, swing ratio, onset density | 2, 7 |
| `music_guide/analysis.py` | keeps `Framer`, `SpectrumMeter`, `StereoMeter`, `BANDS`; loses `MusicFeatures` and the estimators | 1 |
| `music_guide/adapters/audio.py`, `cli.py` | integration, `--analysis-backend` | 1, 12 |
| `music_guide/project.py`, `notes.py`, `adapters/midi.py`, `cli.py` | stored-note rhythm | 8 |
| `tools/evaluate.py`, `tools/midi_truth.py`, `tools/benchmark.py` | development scripts, outside the skill payload | 3 |
| `tests/signals.py` | synthetic signals with known answers | 1 onward |
| `tests/test_estimates_*.py` | one test module per analyzer group | each task |

---

## Phase 1: feature store, beat grid, meter, swing

### Task 1: Feature store and the analyzer pipeline, with existing estimates migrated

**Files:**
- Create: `estimates/common.py`, `estimates/store.py`, `estimates/__init__.py`, `estimates/beats.py`, `estimates/harmony.py`, `tests/signals.py`, `tests/test_estimates_pipeline.py`
- Modify: `model.py`, `analysis.py`, `adapters/audio.py`, `tests/test_audio_metrics.py`

**Interfaces:**
- Produces:
  - `Fact` in `model.py`, the same NamedTuple fields as today.
  - `native(value) -> JSON-native value`, which converts numpy scalars and arrays recursively.
  - `Skip(reason: str, code: Literal["estimate_skipped", "estimate_failed"] = "estimate_skipped")`.
  - `Limit(message: str)`.
  - `Method(name: str, reason: str)`.
  - `Outcome = R | Skip`.
  - `ReportContext(detail: Detail, note: str)` with `.estimate(property, value, unit, method: Method, limits=()) -> Fact`. This is the only constructor for estimate facts, it always stamps inferred and low, and an empty method reason raises.
  - `ReportContext.cap(items, limit, what) -> (list, Limit | None)`.
- Store:
  - `FeatureRecorder(rate, channels, expected_frames)`.
  - `.feed(block: ndarray[frames, channels])`.
  - `.finish(loudness_segments=None) -> FeatureStore`.
  - `FeatureStore` fields: `onset: ndarray[n, 3]` (low <150 Hz, mid 150 to 2000 Hz, high >2000 Hz spectral flux; hop 128 at the decimated rate), `onset_rate: float`, `pcp: ndarray[rows, 12]` (today's untuned linear chroma), `row_rate = 10.0`, `seconds`, `channels`, `folded: bool`.
  - Row k covers samples `[k*rate//10, (k+1)*rate//10)`, the same 100 ms grid as `LoudnessMeter` segments.
  - Arrays are read-only after `finish`.
  - The decimator, the fold to 8 channel groups and the channel-averaged FFT magnitudes move over from `MusicFeatures` unchanged.
- Pipeline:
  - `analyze(store, backend: Backend, role: Literal["mixdown", "stem"]) -> list[tuple[str, Outcome]]`. Each entry is a subject label and its outcome, in spec order. A stem runs tempo, beats, key and melody only.
  - `render(outcomes, ctx) -> tuple[list[Fact], list[Diagnostic]]`. It wraps each `result.report(ctx)` in try/except, so a failure becomes `estimate_failed` naming the exception type. A `Skip` becomes its diagnostic, and a required dependency's Skip makes the dependent `Skip("needs <label>, which was not produced: <reason>")`.
- Results:
  - `TempoCandidates.report` emits `audio.tempo_estimate`, with today's value shape.
  - `KeyResult.report` emits `audio.key_estimate` and `audio.key_timeline`, with today's shapes.
  - The pitch profile emits `audio.pitch_class_profile` (measured, medium) through `ReportContext`'s non-estimate path.
- Tempo autocorrelation is summed per 2^16-frame segment instead of one whole-file FFT.

- [ ] **Step 1:** Write `tests/signals.py`:
  - `tone`, `click_track(bpm, beats_per_bar, seconds, swing=1.0, rate=48000)` returning samples and the true beat and downbeat times;
  - `progression(chords, beats_per_chord, bpm)`, using additive tones with 6 harmonics per note and a bass an octave down;
  - `record(samples, rate, chunk=4096) -> FeatureStore`.
- [ ] **Step 2:** Write `tests/test_estimates_pipeline.py`:
  - `test_store_is_invariant_to_chunk_size`: `onset` and `pcp` are equal for chunk 113 and 65536 at 192 kHz.
  - `test_projected_bytes_at_six_hours`: under 400 MB at 22,050 Hz.
  - `test_failed_analyzer_keeps_loudness_and_names_dependents`: a test backend whose beats function raises. The result has an `estimate_failed` warning for the beat grid, dependents have `estimate_skipped` naming it, and `audio.integrated_loudness` is present.
  - `test_failed_report_is_isolated`: a result whose `report` raises.
  - `test_silence_and_short_files_skip_without_failure`: 0.3 s, 5 s, and 30 s of digital silence. No `estimate_failed`, and every estimate is skipped with a reason.
  - `test_every_estimate_is_inferred_low_with_method_reason`: over a 40 s progression in summary and in full detail. Summary values are a subset of full values, and every fact validates against the schema.
  - `test_lossy_and_fold_notes_reach_every_estimate_reason`: a 12-channel file shows the fold note, and an MP3 written by soundfile shows the lossy note.
- [ ] **Step 3:** Run the new tests and confirm they fail on missing modules.
- [ ] **Step 4:** Implement the interfaces above and move `Fact`, `estimate_tempo`, the key estimators and the constants out of `analysis.py`. Delete `MusicFeatures` and do not add a compatibility wrapper. `adapters/audio.py` feeds `FeatureRecorder` in `_measure`, then calls `render(analyze(...), ReportContext(detail, note))`. The old `estimate_skipped` loop in `_diagnose` goes away, because `render` owns it, and the wording `No key estimate` / `No tempo estimate` is kept.
- [ ] **Step 5:** Migrate the two `MusicFeatures` tests in `tests/test_audio_metrics.py` to `record()` and the store. Keep every other existing assertion. Where a hop change moves a value, loosen only that assertion to the tolerance the behavior needs, and give the reason in the commit message.
- [ ] **Step 6:** Recalibrate the key timeline. Run the three reference songs (paths in the scratchpad truth file) and check that Winterstory gives E major then G major, itipiso D minor throughout, and the trance track D minor throughout. If the hop change breaks this, adjust `KEY_SWITCH_PENALTY` inside the range that passes all three songs and the synthetic key tests. Record the passing range.
- [ ] **Step 7:** Run the full suite and `claude plugin validate --strict .`. Both must be clean.
- [ ] **Step 8:** Commit `Move audio estimates onto a feature store and isolated analyzer pipeline`.

### Task 2: Beat grid, meter, downbeat and swing

**Files:**
- Create: `music_guide/rhythm.py`, `tests/test_estimates_beats.py`
- Modify: `estimates/beats.py`, `estimates/__init__.py`

**Interfaces:**
- Consumes: `FeatureStore.onset`, `TempoCandidates` from Task 1.
- Produces:
  - `beat_grid(store, tempo: TempoCandidates) -> BeatGrid | Skip`, with `BeatGrid(times: ndarray, bpm: float, stability: float, method)` holding at least 8 strictly increasing finite times.
  - `meter(store, beats: BeatGrid) -> MeterEstimate | Skip`, with `MeterEstimate(candidates: tuple[(signature: str, score: float), ...], downbeat_phase: int, downbeats: ndarray)`. Signatures are drawn from `2/4, 3/4, 4/4, 6/8`.
  - `swing(store, beats: BeatGrid) -> Swing | Skip`, with `Swing(ratio: float, pairs: int)`. It skips below 16 pairs.
  - `rhythm.Meter(numerator, denominator)` with `.weights(subdivision) -> list[int]`, the LHL metric hierarchy.
  - `rhythm.swing_ratio(onsets_in_beats) -> float | None`.
- Facts:
  - `audio.beat_grid` has `{bpm, beatCount, firstBeatSeconds, tempoStability}` in summary. Full detail adds `beatTimes`.
  - `audio.meter_estimate` has `{candidates, downbeatPhase, barCount}`. Full detail adds `downbeatTimes`.
  - `audio.swing_ratio` has `{ratio, pairs}`.
- Method:
  - Beats come from Ellis (2007) dynamic programming over the summed onset flux. The tightness penalty is on log interval ratio, and the DP is vectorized in blocks.
  - Meter and downbeat use the per-beat-position mean of low-band flux and of pcp change.
  - Swing is the median ratio of the long to the short interval for onsets between consecutive beats.

- [ ] **Step 1:** Write tests:
  - `test_beats_follow_clicks`: at 90, 120 and 150 BPM, bpm is within 1, and 95% of true beats have an estimate within 30 ms.
  - `test_offset_start_and_tempo_ramp`: a ramp from 100 to 108 BPM gives lower `tempoStability` than steady time.
  - `test_meter_and_downbeat_from_accents`: 3/4 and 4/4 at offsets 0..n-1 give the top signature and `downbeatPhase` equal to the truth.
  - `test_unaccented_clicks_do_not_claim_a_confident_meter`: the score margin between the top two candidates is below 0.1.
  - `test_swing_ratio`: straight gives 1.0 within 0.15, and triplet swing gives 2.0 within 0.35.
  - `test_noise_has_no_beat_grid`: white noise skips with a reason.
  - `test_rhythm_meter_weights`: 4/4 at sixteenths gives `[0,-4,-3,-4,-2,-4,-3,-4,-1,-4,-3,-4,-2,-4,-3,-4]`.
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite clean.
- [ ] **Step 5:** Commit `Estimate beat grid, meter, downbeat and swing`.

### Task 3: Evaluation and benchmark tools

**Files:**
- Create: `tools/evaluate.py`, `tools/midi_truth.py`, `tools/benchmark.py`, `tests/test_tools.py`, `tests/fixtures/truth-synthetic.json`

**Interfaces:**
- `tools/evaluate.py REPORT.json TRUTH.json [--markdown]` reads only the CLI's JSON.
  - The truth shape is optional keys: `tempoBpm`, `meter`, `beats[]`, `downbeats[]`, `keys[] {startSeconds,endSeconds,tonic,mode}`, `chords[] {startSeconds,endSeconds,label}` (Harte-style labels such as `D:sus2`, `A#:maj`, `A:7`), `firstChord`, `melody[] {startSeconds,endSeconds,midi}`, `structure[] {startSeconds,letter}`.
  - It prints, for each key it can evaluate:
    - tempo error and octave-aware error;
    - beat F-measure at a 70 ms window;
    - meter match;
    - key time agreement;
    - chord overlap (label, root, major/minor family);
    - first chord;
    - melody raw pitch accuracy within 50 cents;
    - boundary hit rate at 3 s.
  - An estimate that is missing prints `not reported` and counts as a failure.
- `tools/midi_truth.py MIDI.mid --chord-tracks NAMES --melody-track NAME --bass-track NAME [--offset SECONDS] > TRUTH.json` derives truth from a constant-tempo MIDI file:
  - beats and downbeats from the tempo and meter;
  - chords per beat, by matching the pitch-class set of the chord and bass tracks to the chord templates, then merging repeats;
  - melody from the named monophonic track.
- `tools/benchmark.py` prints wall time and peak RSS for a 4 minute synthetic stereo file and a 6 hour synthetic mono 8 kHz file, both written to a temporary directory.

- [ ] **Step 1:** Write `tests/test_tools.py`:
  - `test_evaluate_scores_a_perfect_report`: a report built from `truth-synthetic.json` scores 1.0.
  - `test_evaluate_marks_missing_estimates`.
  - `test_midi_truth_labels_known_chords`: a mido-built file with Dsus2, F and A7 gives those labels and beat times.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite.
- [ ] **Step 5:** Generate the trance truth outside the repository:

  ```bash
  python tools/midi_truth.py ~/Downloads/untitled.mid --chord-tracks Pad --bass-track Bass --melody-track Lead > <scratch>/trance.truth.json
  ```

  Add Winterstory and itipiso truth files by hand (tempo, meter and keys from the user's statements). Run `tools/evaluate.py` and `tools/benchmark.py`.
- [ ] **Step 6:** Record the phase-1 numbers in `docs/verification.md`, failures included.
- [ ] **Step 7:** Commit `Add reference evaluation and benchmark tools`.

## Phase 2: chords and the chord-aware key

### Task 4: NNLS note activations, tuning, treble and bass chroma

**Files:**
- Modify: `estimates/store.py`
- Test: `tests/test_estimates_store.py`

**Interfaces:**
- `FeatureStore` gains:
  - `treble: ndarray[rows, 12]`;
  - `bass: ndarray[rows, 12]`;
  - `activations: ndarray[min(rows, 18000), 73]` (A1..A7 in semitones, first 1800 s, for melody);
  - `tuning_cents: float`.
- Method:
  - Log-frequency spectrum at 3 bins per semitone from the row FFT.
  - Spectral whitening by a running mean over one octave.
  - Non-negative activations against a note dictionary of 8 harmonics with amplitude 0.6^(h-1), by multiplicative NNLS updates. Use a fixed iteration count chosen in this task to meet the time budget.
  - Bass chroma weights notes A1..G#3 and treble weights notes A3..A7, with overlapping ramps.
  - Tuning is the circular mean of sub-semitone peak offsets over the first 60 s of non-silent rows. Rows before the lock are buffered (at most 600) and processed when it fires.

- [ ] **Step 1:** Write tests:
  - `test_detuned_chord_keeps_its_roots`: a C major triad 30 cents sharp gives `tuning_cents` within 10 of 30 and a treble chroma argmax set of {C, E, G}.
  - `test_overtones_do_not_create_a_major_third`: a single D note with harmonics gives treble F# energy below 10% of D. Raw `pcp` for the same note is allowed to show more.
  - `test_bass_chroma_follows_the_bass_note`: F/A gives bass argmax A.
  - `test_projected_bytes_with_activations`: still under 400 MB at 6 h.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite, and run `tools/benchmark.py` for the 4 minute budget.
- [ ] **Step 5:** Commit `Record harmonic-suppressed treble and bass chroma with tuning`.

### Task 5: Chord timeline and chord summary

**Files:**
- Modify: `estimates/harmony.py`, `estimates/__init__.py`
- Test: `tests/test_estimates_chords.py`

**Interfaces:**
- `chords(store, *, beats: Outcome[BeatGrid]) -> ChordTimeline | Skip`.
  - `ChordTimeline(per_unit: tuple[Chord, ...], unit_times: ndarray, basis: Literal["beats", "fixed"])`.
  - `Chord(root: str | None, quality: str | None, bass: str | None)`. `N` has all three None.
  - Qualities are `maj, min, sus2, sus4, dim, maj7, min7, 7`.
- `ChordTimeline.segments()` and `.shares()` are derived from `per_unit`.
- Facts:
  - `audio.chord_timeline` has `{basis, segments: [{startSeconds, endSeconds, chord: "D:sus2", bass?, alternatives?}]}`, capped at 256. Full detail adds `perUnit`.
  - `audio.chord_summary` has `{shares: top 12 by duration, majorShare, minorShare, otherShare, noChordShare}`.
- Method:
  - Beat-synchronous median of treble and bass chroma, or 0.5 s units without beats.
  - Cosine score against binary templates, with the bass chroma adding weight for the root, or for the bass when it is a chord tone.
  - Viterbi with a self-transition probability chosen in this task and a uniform switch.
  - When the two best templates share their pitch classes (for example D:sus2 and A:sus4), the segment lists the other as an alternative unless the bass separates them.

- [ ] **Step 1:** Write tests:
  - `test_triads_and_sevenths_in_all_roots`: duration-weighted label overlap ≥ 0.8 and root overlap ≥ 0.9 over all 8 qualities in all 12 roots.
  - `test_no_chord_over_silence`.
  - `test_slash_chord_reports_bass`.
  - `test_sus2_lists_its_sus4_twin_without_bass_evidence`.
  - `test_without_a_beat_grid_uses_fixed_units`: a pad with no percussion gives `basis == "fixed"`.
  - `test_summary_is_capped_with_a_warning`: a long, fast progression gives 256 segments and `estimate_truncated`.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Run `tools/evaluate.py` on the trance truth and record chord overlap and first chord in `docs/verification.md`.
- [ ] **Step 6:** Commit `Estimate chord timeline and chord summary`.

### Task 6: Chord-aware key experiment

**Files:**
- Modify: `estimates/harmony.py`
- Test: `tests/test_audio_metrics.py` (existing key tests), `tests/test_estimates_chords.py`

**Interfaces:**
- `key(store, *, chords: Outcome[ChordTimeline])` keeps today's output shapes.
- Variant B scores each 5 s cell by a weighted sum of the chord-template profile correlation and the key-membership share of its chords, with the tonic, dominant and subdominant weighted up.
- It uses the same Viterbi and penalty mechanism.

- [ ] **Step 1:** Add `test_borrowed_major_chord_stays_in_minor_key` (D minor with one D major bar) and `test_dominant_modulation_detected_within_60_s` (C to G, 60 s each).
- [ ] **Step 2:** Run both variants over all key tests and the three reference songs with `tools/evaluate.py`.
- [ ] **Step 3:** Keep variant B only if it passes every test the baseline passes and agrees with the truth on all three songs at least as well. Delete the losing variant and its code.
- [ ] **Step 4:** Record both variants' numbers in `docs/verification.md`.
- [ ] **Step 5:** Run the suite.
- [ ] **Step 6:** Commit `Decide the key estimator by evaluation`, naming the winner and the numbers.

## Phase 3: rhythm, groove, stored notes, structure

### Task 7: Syncopation, groove and danceability

**Files:**
- Modify: `music_guide/rhythm.py`
- Create: `estimates/groove.py`, `tests/test_estimates_groove.py`

**Interfaces:**
- `rhythm.syncopation(onsets_in_beats: list[float], meter: Meter, subdivision=4) -> float`. This is Longuet-Higgins and Lee, normalized per bar.
- `rhythm.onset_density(onsets_in_beats, beats_per_bar) -> float`.
- `groove(store, beats: BeatGrid, tempo: TempoCandidates, *, meter: Outcome[MeterEstimate]) -> Rhythm | Skip`.
- Fact `audio.rhythm` has:
  - `onsetsPerSecond`;
  - `onsetsPerBar` (null without a meter);
  - `syncopation` (null without a meter);
  - `pulseClarity` (the beat-period autocorrelation peak relative to its neighbors, 0..1);
  - `microtimingMs` (the median absolute onset deviation from the nearest sixteenth);
  - `danceability` (0..1);
  - `danceabilityFormula`, a string naming its weighted terms.

- [ ] **Step 1:** Write tests:
  - `test_syncopation_orders_straight_offbeat_syncopated`: straight < offbeat < syncopated, and straight equals 0.
  - `test_audio_syncopation_matches_rhythm_module`: for a click pattern, the audio value equals `rhythm.syncopation` on the generating onsets within 0.05.
  - `test_danceability_prefers_steady_pulse`: 120 BPM with a steady kick scores higher than randomly timed clicks.
  - `test_rhythm_without_meter_nulls_bar_measures`.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Commit `Estimate rhythm, syncopation and danceability`.

### Task 8: Stored-note rhythm

**Files:**
- Modify: `model.py` (`Evidence.note_starts`), `project.py`, `adapters/midi.py`, `cli.py`
- Test: `tests/test_stored_rhythm.py`

**Interfaces:**
- `Evidence.note_starts: dict[(source_id, scope), NoteStarts]`. It is never serialized.
  - `NoteStarts(starts_beats: tuple[float], method, locator, confidence, grade_reason)`.
  - The scope string is the full track scope including any alternative prefix, so DAWproject alternatives cannot collide.
- `ProjectFacts.track_notes` and the MIDI adapter record entries at the same confidence they emit for `track_notes`, so a graded format keeps its grade.
- `project.attach_track_rhythm(staged: Evidence, source_id)` is called by `cli.build` after each adapter returns.
  - It builds a `rhythm.BarGrid` from the staged meter map, or from the static meter.
  - It emits `project.track_rhythm` `{onsetsPerBar, syncopation, swingRatio, meterSource}` per track through the listed-property cap.
  - With no meter, or a meter it cannot place, the bar measures are null and a `rhythm_meter_unknown` info diagnostic is added.
- `summarize_notes`, `project.track_notes` and `midi.track_summary` values stay byte-identical.

- [ ] **Step 1:** Write tests:
  - `test_midi_straight_and_swung_eighths`;
  - `test_meter_after_notes_in_musicxml_still_scores`;
  - `test_meter_map_changes_bar_grid`;
  - `test_no_meter_gives_null_and_diagnostic`;
  - `test_graded_format_keeps_its_grade`, using an FL Studio fixture;
  - `test_existing_note_summaries_unchanged`, comparing against values captured before the change;
  - `test_midi_only_run_does_not_import_numpy`, as a subprocess `sys.modules` check.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Run the trance MIDI and check that the Beat and Lead tracks get plausible values.
- [ ] **Step 6:** Commit `Report rhythm measures for stored notes`.

### Task 9: Timbre and structure

**Files:**
- Modify: `estimates/store.py` (`timbre: ndarray[rows, 12]`, log energy in 12 log-spaced bands)
- Create: `estimates/structure.py`, `tests/test_estimates_structure.py`

**Interfaces:**
- `structure(store, *, beats: Outcome[BeatGrid], meter: Outcome[MeterEstimate]) -> Structure | Skip`. It skips below 45 s.
- `Structure(boundaries: ndarray, letters: tuple[str, ...], novelty: ndarray, basis)`.
- Method:
  - Per-unit standardized treble chroma plus timbre.
  - A banded self-similarity of at most 64 units each side of the diagonal, never a full matrix.
  - Foote novelty with a checkerboard kernel, peaks snapped to downbeats when a meter exists.
  - Letters assigned by agglomerating segment mean features under a similarity threshold. At most 500 segments.
- Fact `audio.structure` has `{segments: [{letter, startSeconds, endSeconds}], distinctLetters}`. Full detail adds `novelty`.

- [ ] **Step 1:** Write tests:
  - `test_abab_and_aaba`: letters match, and boundaries are within one bar.
  - `test_homogeneous_piece_is_one_segment`.
  - `test_short_file_skips`.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite and the benchmark.
- [ ] **Step 5:** Commit `Estimate lettered song structure`.

## Phase 4: energy, mood, melody, backend, docs

### Task 10: Energy timeline and mood

**Files:**
- Modify: `estimates/store.py` (`loudness: ndarray[rows] | None` from `LoudnessMeter` 100 ms segments, `brightness: ndarray[rows]` as the spectral centroid of the row FFT), `loudness.py` (expose segment mean squares)
- Create: `estimates/energy.py`, `tests/test_estimates_energy.py`

**Interfaces:**
- `energy(store, *, structure: Outcome[Structure], beats: Outcome[BeatGrid]) -> EnergyTimeline | Skip`.
  - Fact `audio.energy_timeline` has `{segmentedBy: "structure" | "uniform", segments: [{startSeconds, endSeconds, loudnessLufs, brightnessHz, onsetsPerSecond}]}`.
  - Full detail adds `perBar`.
- `mood(energy: EnergyTimeline, *, tempo, rhythm, chords, key) -> Mood | Skip`.
  - `arousal` is the mean of available normalized terms: loudness (-30 to -8 LUFS), onset rate (0 to 8 per s), tempo (60 to 180 BPM) and brightness (500 to 4000 Hz).
  - `valence` is the mean of available terms: major share of chords or the key mode, brightness, and tempo.
  - Labels by quadrant: energetic/uplifting (high arousal, high valence), tense/aggressive (high arousal, low valence), calm/peaceful (low arousal, high valence), sad/dark (low, low). Ranked by distance to the quadrant centers (0.75, 0.75) and so on.
  - Fact `audio.mood_estimate` has `{arousal, valence, labels: [best two], alternatives: [other two], inputs: {term: value or null}, formula}`.

- [ ] **Step 1:** Write tests:
  - `test_energy_follows_sections`: segment loudness is ordered as built.
  - `test_uniform_segments_without_structure`.
  - `test_loud_bright_major_fast_is_energetic`.
  - `test_quiet_dark_minor_slow_is_calm_or_sad`.
  - `test_mood_inputs_null_for_skipped_dependencies`.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Commit `Estimate energy timeline and mood`.

### Task 11: Melody

**Files:**
- Create: `estimates/melody.py`, `tests/test_estimates_melody.py`

**Interfaces:**
- `melody(store, *, beats: Outcome[BeatGrid]) -> Melody | Skip`.
- Method:
  - Salience from `store.activations` restricted to E2 to C7, with treble chroma energy of accompaniment subtracted.
  - Viterbi over pitch states plus an unvoiced state, with a transition cost proportional to the interval.
  - Voicing by a salience threshold relative to the row maximum.
- Fact `audio.melody` has `{lowMidi, highMidi, medianMidi, voicedShare, contour: [{startSeconds, midi|null}] per beat or per 0.5 s, coveredSeconds}`. Full detail adds a per-row pitch.
- More than 1800 s gives `estimate_truncated`.
- A stem runs melody too, and its reason says that a stem isolates the part.

- [ ] **Step 1:** Write tests:
  - `test_melody_alone`: median within 1 semitone, range within 2, voiced share within 0.1.
  - `test_melody_over_pad`: the same within looser bounds named in the test.
  - `test_stem_melody_reason`.
  - `test_melody_cap`: patch the cap small in-process and check `estimate_truncated`.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Run `tools/evaluate.py` against the trance Lead track and record the numbers.
- [ ] **Step 6:** Commit `Estimate predominant melody`.

### Task 12: Backend slot

**Files:**
- Modify: `estimates/__init__.py`, `cli.py`
- Test: `tests/test_estimates_pipeline.py`

**Interfaces:**
- `Backend(name, beats=beat_grid, meter=meter, chords=chords, melody=melody)` is a frozen dataclass.
- `BACKENDS = {"builtin": Backend("builtin")}`.
- `--analysis-backend` uses `choices=("builtin",)` as a literal, so project-only runs never import numpy.
- A backend's `Method.name` prefixes `observation.method`.

- [ ] **Step 1:** Write tests:
  - `test_backend_choices_match_registry`;
  - `test_unknown_backend_exits_2`;
  - `test_backend_method_names_reach_facts`, using a test backend.
- [ ] **Step 2:** Run them and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the suite.
- [ ] **Step 5:** Commit `Add the analysis backend slot`.

### Task 13: Documentation, rules and the final evaluation

**Files:**
- Modify: `AGENTS.md`, `skills/music-guide/SKILL.md`, `references/domains.md`, `references/formats.md`, `references/intents.md`, `README.md`, `docs/verification.md`

- [ ] **Step 1:** Change the mood rule in `AGENTS.md` and the matching guidance in `SKILL.md` and `references/domains.md`. Mood labels from `audio.mood_estimate` may be stated as low-confidence estimates with their formula and alternatives, never as fact, and the user's own description wins.
- [ ] **Step 2:** Document every new property in `references/formats.md`: method, limits, caps and what each cannot see. Add reading guidance to `SKILL.md`: chords as estimates, sus twins, the beat-grid basis, syncopation for audio versus stored notes, structure letters, and melody reliability without stems.
- [ ] **Step 3:** Run the final `tools/evaluate.py` on all three songs and `tools/benchmark.py`. Record every number in `docs/verification.md`, including failures and the budgets met or missed.
- [ ] **Step 4:** Run the suite and `claude plugin validate --strict .`.
- [ ] **Step 5:** Commit `Document the music analysis estimates and record the evaluation`.
