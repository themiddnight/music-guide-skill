# Verification record

## Version 0.2.0: broad format support

Verified on macOS arm64 on 2026-10-16 with Python 3.14.8, Claude Code plugin validation, and uv 0.12.21. Linux and Windows have not been run.

**Automated behavior.** `PYTHONWARNINGS=error .venv/bin/python -m unittest discover -s tests -v` runs 244 tests in about 34 s with no warnings. Most CLI tests run the CLI in process through `tests/support.py`; real subprocess tests remain for interpreter isolation, the Python version and missing-dependency exit 3, installed copies, the AAF child process, startup cost, and one end-to-end smoke test per adapter group that must match the in-process output exactly.

**Audio measurements.** Loudness, loudness range and true peak were checked against ITU-R BS.1770-4 reference levels, EBU Tech 3341 cases 1, 3, 4 and 5 (-23.0 LUFS within 0.1) at 44.1, 48 and 96 kHz, and EBU Tech 3342 loudness-range cases. An independent ffmpeg `ebur128` run agreed within 0.05 LU on integrated loudness and range for every case. An independent reviewer matched integrated loudness against a brute-force BS.1770-4 implementation to about 1e-13 LU for arbitrary block sizes, and found true peak within -0.11 to +0.008 dB of a band-limited ideal for tones up to 0.45 fs.

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
