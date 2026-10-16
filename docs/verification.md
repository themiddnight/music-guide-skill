# Verification record

## Version 0.3.0: shared music analysis

Verified on macOS arm64 on 2026-10-17. The pre-change baseline at `3bb8811` passed 360 tests. The integrated pipeline passed 430 tests before the independent review fixes and 434 tests after them, using `PYTHONWARNINGS=error`. A fresh reviewer independently passed the 20 focused regression tests and 12 extra sample-energy comparisons at relative tolerance 1e-12. No Python warning remained.

### Implemented behavior

One streaming decode records spectral flux, raw pitch classes, tuned harmonic note activations, wide and narrow treble chroma, bass chroma, compact timbre, feature-band brightness, and shared K-weighted energy. The existing raw pitch-class key baseline remains unchanged. The pipeline now adds chord timelines, audio and stored-note rhythm, lettered structure, energy descriptors, mood alternatives, and predominant melody. Stems retain their own source scope and receive an independent melody estimate.

All audio musical estimates use low confidence and inferred provenance. Every method states its limits. Missing prerequisites produce skips, failures remain isolated from independent estimates, and output caps produce diagnostics. Full output uses second-based timeline coordinates. The melody store stops at 18,000 rows, covering at most 30 minutes. No pretrained model, semantic listening, stem separation, or automatic rendering runs.

Tests use constructed features and recorded synthetic signals. Harmonic tests cover all 12 roots and eight qualities, detuning, overtones, inverted channels, noisy and near-silent input, low pads under high leads, and suspended-chord alternatives. Structure tests include ABAB, AABA, and homogeneous chord progressions with drums. These synthetic checks are easier than real music and do not establish broad transcription accuracy.

### Reference results and their limits

All three provided source sets ran through the real full-detail CLI and the evidence schema. No source failed and no analyzer reported `estimate_failed`. Files stayed read-only and outside the repository.

| Check | Trance | Itipiso | Winterstory |
| --- | --- | --- | --- |
| Top tracked pulse | 128.1 BPM | 92.0 BPM | 100.0 BPM |
| Top meter | 4/4 | 4/4 | 4/4 |
| Baseline key timeline | D minor proxy | D minor | E major, then G major |

The trance track is listener-labelled D Hirajoshi. D minor is only its closest available major/minor proxy, not a correct scale name. Winterstory has a listener-confirmed brief D major passage before G major. Its exact transition times were not independently annotated, so key scoring excludes 150 to 190 seconds. The remaining 209.6 seconds test E major at the start and G major at the end without using the estimator's own boundary as truth.

An offline chord-function key candidate matched the baseline on all three limited listener checks and on synthetic distant modulation, a 14-second borrowed parallel-major passage, and static C major. Both scored full agreement on these six checks. This is a tie on a small calibration set, not evidence of improvement or general accuracy. The production baseline remains in place. The candidate and its transparent score formula live in `tools/key_ab.py`, outside the installed runtime.

For trance, `tools/midi_truth.py` derived comparison labels from Chord and Pad, with Bass and Lead as the corresponding stored parts. At zero offset and 70 ms timing tolerance, the audio report gave:

| Stored-reference comparison | Agreement |
| --- | --- |
| Beat F-measure | 0.9916 |
| Downbeat F-measure | 0.9917 |
| Exact chord-label duration overlap | 0.6116 |
| Chord-root duration overlap | 0.6304 |
| Major/minor chord-family overlap | 0.6314 |
| Melody raw pitch agreement, 0.5-semitone tolerance | 0.4291 |

These are agreement with note-derived stored intent, not independently labelled audible truth. The result does not support a claim of 70 to 80 percent chord accuracy. The melody result is weak for this mix and must not be described as reliable automatic transcription.

The first audio chord candidate is D:sus2 with A:sus4 as an equivalent pitch-class alternative. The SOLNA opening also stores D:sus2. However, a naive reconstruction from its explicit chord bars and loop repeat counts covers 221.25 seconds, while the render lasts 225.05 seconds. Its naive chord labels agree with MIDI-derived labels for only 55.9 percent of that reconstructed interval. The application's full scheduling semantics and alignment have not been independently verified, so SOLNA is not promoted to audible ground truth. No valid SOLNA-based accuracy score is claimed.

There are no independently annotated structure, groove, mood, or melody references for the other songs. Mood labels are untrained heuristic alternatives, and the selected predominant pitch can follow bass or accompaniment. No emotional accuracy or genre-recognition claim is made.

### Independent review and regressions

The independent review identified three defects. Regressions now check duration weighting for a one-sample tail, LHL anticipation across a bar boundary, and an oversized MusicXML meter map that previously could allocate billions of metrical ticks. Meter maps now obey the same numerator and denominator bounds as static meters and produce a visible diagnostic. Energy integration weights the actual sample duration of the final row rather than treating it as a full 100 ms. The public audio attack detector is shared by rhythm and energy.

A comment review corrected misleading descriptions of logarithmic frequency grids, chord timing basis, density-fit limits, and dependent-analyzer failures. The offline key experiment was removed from the installed runtime.

### Resource measurements

The [raw benchmark runs](benchmarks/music-analysis-0.3.0.json) retain each timing, resource value, exit status, and work check.

The final runtime at `8afb305` was measured with Python 3.14.3 on an 8-core macOS arm64 machine. The four-minute stereo 44.1 kHz case ran five times in fresh interpreters. Wall time had a median of **2.836 seconds**, range **2.820 to 3.138 seconds**. Peak RSS ranged from **128.4 to 135.4 MiB**. Every run was below the 15-second and 400-MiB budgets.

The six-hour mono 8 kHz case ran once on the final code. It completed in **47.674 seconds** with **348.8 MiB** peak RSS, below the 1-GiB memory budget. This is a single synthetic workload, not a promise for six-hour stereo files, every sample rate, or every computer. An earlier exploratory run under different machine load took 76.5 seconds and 283 MiB. That run is not a controlled before/after comparison.

The files were generated immediately before measurement, so their filesystem cache was warm. Generation time was excluded, while CLI startup, hashing, decoding, feature extraction, analysis, and JSON output were included. The host had background applications running and a starting load average of 6.47 on 8 cores. No speedup over another implementation is claimed. Each timed child exited successfully with empty stderr, all required analyzer families present, and no `estimate_failed`. RSS came from that child's `wait4` resource usage, not the parent's fixture-generation peak.

A separate four-minute profiler run, excluded from these timings, identified NumPy correlation, FFTs, feature-row construction, and signal filtering among the hottest operations. Numerical processing dominates the measured workload rather than file reads. The machine's external load and memory state varied, so the results establish the tested budgets rather than a stable throughput guarantee.

### Installation and warnings

The 0.3.0 plugin installed through the native local marketplace commands of both Codex and Claude Code in isolated configuration directories. Every one of the 51 skill payload files matched the repository byte for byte. Each installed CLI ran a 60-second synthetic stereo mixdown from an unrelated directory and produced all supported analyzer families without `estimate_failed`. The standalone installer also copied the skill successfully. Both the marketplace and plugin manifests passed strict Claude validation. Normal user profiles were not changed.

The first isolated Codex attempt warned that its test configuration directory did not exist. Creating that directory fixed the test setup and the subsequent native installations completed without that warning. No warning was suppressed. Reference clipping indicators and chord-share truncation diagnostics remain visible because they describe input or bounded output, not a build defect. Lossy decode and partial native-project support remain explicit in source diagnostics.

### Reproduce the checks

Run from the repository using an interpreter with the development requirements:

```sh
PYTHONWARNINGS=error uv run --with-requirements requirements-dev.txt python -m unittest discover -s tests -q
uv run --with-requirements requirements-dev.txt python tools/evaluate.py report.json truth.json --markdown
uv run --with-requirements requirements-dev.txt python tools/key_ab.py /path/to/mixdown.wav --truth truth.json
uv run --with-requirements requirements-dev.txt python tools/benchmark.py --six-hour --timeout 600
claude plugin validate --strict .
```

`tools/evaluate.py` reports only the families present in the truth file. Do not compare totals across different family sets. Store a full-detail report for beat, downbeat, and melody comparisons. Personal truth files and music are not committed. The benchmark checks that all supported analyzer families ran and rejects `estimate_failed` output rather than treating a fast failure as success.

## Version 0.2.0: broad format support

Verified on macOS arm64 on 2026-10-16 with Python 3.14.3 and 3.14.8, Claude Code plugin validation, and uv 0.12.21. Linux and Windows have not been run.

**Automated behavior.** `PYTHONWARNINGS=error uv run --with-requirements requirements-dev.txt python -m unittest discover -s tests -q` passes all 258 tests on Python 3.14.3. Adding `--python /opt/homebrew/bin/python3` after `uv run` passes the same suite on Python 3.14.8. Both runs report no warnings. Most CLI tests run the CLI in process through `tests/support.py`; real subprocess tests remain for interpreter isolation, the Python version and missing-dependency exit 3, installed copies, the AAF child process, startup cost, and one end-to-end smoke test per adapter group that must match the in-process output exactly.

**Audio measurements.** Loudness, loudness range and true peak were checked against ITU-R BS.1770-4 reference levels, EBU Tech 3341 cases 1, 3, 4 and 5 (-23.0 LUFS within 0.1) at 44.1, 48 and 96 kHz, and EBU Tech 3342 loudness-range cases. An independent ffmpeg `ebur128` run agreed within 0.05 LU on integrated loudness and range for every case. An independent reviewer matched integrated loudness against a brute-force BS.1770-4 implementation to about 1e-13 LU for arbitrary block sizes, and found true peak within -0.11 to +0.008 dB of a band-limited ideal for tones up to 0.45 fs.

**Review regressions.** Simultaneous distinct score meters and tempos remain in their event maps, emit disagreement diagnostics, and produce no false static claim. MusicXML and MuseScore fixtures cover conflicting meters. The CLI uses `analyze.py` in usage messages regardless of the host main module. Stereo feature extraction averages per-channel FFT magnitudes before chroma and spectral flux, preserving pitch and periodicity under independent channel polarity flips. Tests cover antiphase chords and clicks, separate pitches in each channel, and 192 kHz decimation across 113-frame and 65,536-frame input blocks. Feature extraction folds files wider than 8 channels into 8 groups; a 60 s, 1024-channel input then takes 1.6 s instead of growing with every channel (measured on this machine), and a test confirms that duplicated channels fold to the same profile as the original 8. The key timeline separates a synthetic E major section from a G major section at the 30 s boundary, keeps D minor through a 15 s D major passage or a single D major chord, keeps an unmodulated file as one segment, and is omitted under 30 s. A synthetic C to G major change splits only when each section lasts about 90 s. The user labelled the three reference songs by ear: Winterstory is E major, then one bar of D major, then G major to the end; itipiso is D minor throughout; the trance track is D Hirajoshi, closest to D minor throughout. A first version that labelled 25 s windows gave a false 15 s D major segment in Winterstory, a false D major opening in itipiso and a false A minor opening in the trance track. The Viterbi version gives E major to 175 s then G major, D minor throughout, and D minor throughout, and the whole-file key for itipiso changed from D major to D minor. Every switching penalty from 1.0 to 3.0 gave these results on the three songs, which are the only real calibration data. The score event maps retain no part attribution because the shared adapter data does not store it.

**Format parsers.** Every project format has synthetic fixtures built from public documentation and open-source readers, listed per format in the input contract. Only Logic Pro and Cubase were checked against real projects. Three independent reviewers then fuzzed and attacked the parsers. Their 44 verified findings (crashes on malformed numbers, decompression and parsing denial of service, a FIFO hang in Logic packages, unbounded output, misleading confidence) were fixed with a regression test each, apart from two declined sub-items. Ableton clips marked disabled are still read, because no public source confirms the field. Per-note `project.note` values do not flag placeholder velocities; the track summary's confidence and reason count them instead.

**Real reference checks.** The three user reference sets were rerun on the final code. Every report validated against the evidence schema and no source failed.

- WAV, MIDI and SOLNA ("D Hirajoshi" trance sketch): MIDI and SOLNA both store 128 BPM with no conflict. The audio pitch-class profile's five strongest classes are D, A, F, A# and E, the D Hirajoshi set. The audio tempo candidates are 64.1 and 128.3 BPM.
- MP3 with a Logic Pro 12.4 project: the MP3 decodes directly, with a `lossy_decode` diagnostic. The package metadata gives 92 BPM, 4/4, D minor, 48 kHz and 34 tracks, and 92 BPM is among the audio tempo candidates. A user declaration of 128 BPM raised a tempo conflict as intended.
- WAV with a Cubase 11 project: the application, 44.1 kHz, 100 BPM and 4/4 come from structural metadata, and 28 track names with kinds come from byte patterns at low confidence. The audio tempo candidates include 99.9 BPM.

**Model response checks.** Two fresh subagents used only SKILL.md and the tool to answer realistic requests: scale adherence, loudness and drop ideas for the trance sketch, and a Spotify mix critique with a request to name the tracks causing low-mid buildup for the Cubase set. Both cited observation IDs, kept measured, extracted and inferred evidence apart, labelled outside knowledge, and declined to name tracks from whole-file measurements. Their feedback led to General MIDI percussion flags, a separate 250 to 500 Hz band, and delivery-target and scale-check guidance. Two runs are a smoke check, not a quality benchmark.

**Packaging.** `claude plugin validate --strict` passes. A copy installed with `install.py` ran from an unrelated directory through `uv run --no-project --python 3.12 --with-requirements`. The Codex validator was not rerun for 0.2.0.

## Version 0.1.0: MVP

Verified on macOS arm64 on 2026-10-16 with Python 3.14.8, Codex CLI 0.162.1, and Claude Code 2.1.300. Python 3.10+ is the declared floor; other Python versions and operating systems have not been run here.

### Automated behavior

The final suite has 12 test methods with parameterized subcases. Run it from the repository with the interpreter that has the skill requirements installed:

```sh
PYTHONWARNINGS=error .venv/bin/python -m unittest discover -s tests -v
```

Both the implementer and independent review ran the suite. The suite is run with warnings treated as errors. The first 11 methods also passed a separate reviewer run. The WAVE Extensible regression was added after checking real references. Coverage includes WAV unsigned 8-bit and signed 16/24/32-bit PCM, 32/64-bit float, known RMS/peak values, stereo, silence, nonfinite samples, numeric underflow, full-scale diagnostics, MIDI notes/tempo maps, separate MIDI parts, SOLNA optional bass-note null and future versions, static conflicts and equal numeric claims, DAWproject transport scope/units, missing structure, malformed or unsupported ZIP/XML/JSON, and optional-source isolation.

Installer checks cover a copied skill run from an unrelated directory, repeated install before and after use, collision preservation, dry-run without filesystem changes, and destination symlink refusal. JSON evidence is validated against the canonical schema, including source and observation references.

An independent reviewer found three defects in the first implementation. Regression tests reproduced lost mixdown output for unsupported ZIP compression or oversized SOLNA numbers, incorrect tempo-unit labeling, and a false zero track count for missing Structure. Commit `0c38239` fixes them and adds coverage. A fresh independent reviewer approved that commit after rerunning the suite.

### Native packaging

The portable root manifest validated against the official Agent Plugins 1.0.0 schema. The bundled skill passed the Codex skill-creator validator. Claude plugin and marketplace manifests passed `claude plugin validate --strict --json` with no warnings or errors.

Both hosts installed `music-guide@music-guide-local` from this local repository through their native marketplace commands. Tests used isolated `CODEX_HOME` and `CLAUDE_CONFIG_DIR` directories, leaving normal user settings unchanged. Claude's component inventory showed one skill and no agents, hooks, or MCP servers.

For both installed host copies, every payload file matched the final source by SHA-256. Their CLIs ran from an unrelated directory with synthetic mixdown, stem, MIDI, SOLNA, DAWproject, and unsupported native project inputs. Output passed the evidence schema and exact known WAV measurement checks.

Claude's plugin validator also offered non-warning advice to add a published repository installation line. At the time no publication URL existed, so the README documented the local marketplace installation; it now also documents the GitHub source. No build, test, or validation warning remains outstanding. Expected source diagnostics, such as possible clipping and unsupported projects, remain visible in the JSON report.

### Real reference checks

After the synthetic suite, the user supplied three local reference sets. All original files remained read-only and outside the repository. The first set's WAV, MIDI, and SOLNA version 12 parsed successfully and its 4,077 observations validated against the evidence schema. Its MIDI tempo event and SOLNA tempo both stored 128 BPM. This comparison does not establish tempo from audio.

The second set used an MP3 plus a Logic package. Direct MP3 support remains outside the CLI contract. An available local FFmpeg decoded the MP3 into a temporary 24-bit PCM WAV. That WAV used a WAVE Extensible header, exposing a legitimate rejection in the original adapter. Synthetic PCM24 and float WAVE Extensible tests reproduced it before the format check was fixed. The derived WAV then measured successfully. The Logic package remained inventoried as unsupported.

The third set's WAV measured successfully and its Cubase project remained inventoried as unsupported. Every resulting report passed the canonical schema. No native Logic or Cubase project contents were parsed.

FFmpeg warned that the MP3 container duration was estimated from bitrate. The measurement uses the decoded WAV frame count instead of that estimate, and the conversion warning remains recorded in the private handoff report. The first reference WAV had 1,229 interleaved sample values at full scale. The tool retained a possible-clipping diagnostic without claiming audible distortion or altering the mix. Unsupported Logic/Cubase diagnostics were retained as intended fallback behavior.

### Limits

Committed fixtures remain synthetic. The local real-reference runs are additional smoke checks, not a redistributable corpus or an exhaustive format-compatibility claim. SOLNA coverage is an inspected subset, not complete format compatibility. DAWproject coverage is metadata only, with stored or deflated project.xml entries.

The CLI does not estimate key, tempo, genre, or mood from audio. It does not measure LUFS or true peak, reconstruct nested arrangements, align stems, render music, or generate artwork. The conversational skill supplies explanations and proposals within available evidence. Model response quality across real music requests has not been evaluated end to end.

The normal user profile has not been installed or configured. Nothing has been pushed, published, or submitted as a pull request.
