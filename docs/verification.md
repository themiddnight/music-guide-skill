# MVP verification record

Verified on macOS arm64 on 2026-10-16 with Python 3.14.8, Codex CLI 0.162.1, and Claude Code 2.1.300. Python 3.10+ is the declared floor; other Python versions and operating systems have not been run here.

## Automated behavior

The final suite has 12 test methods with parameterized subcases. Run it from the repository with the interpreter that has the skill requirements installed:

```sh
PYTHONWARNINGS=error .venv/bin/python -m unittest discover -s tests -v
```

Both the implementer and independent review ran the suite. The suite is run with warnings treated as errors. The first 11 methods also passed a separate reviewer run. The WAVE Extensible regression was added after checking real references. Coverage includes WAV unsigned 8-bit and signed 16/24/32-bit PCM, 32/64-bit float, known RMS/peak values, stereo, silence, nonfinite samples, numeric underflow, full-scale diagnostics, MIDI notes/tempo maps, separate MIDI parts, SOLNA optional bass-note null and future versions, static conflicts and equal numeric claims, DAWproject transport scope/units, missing structure, malformed or unsupported ZIP/XML/JSON, and optional-source isolation.

Installer checks cover a copied skill run from an unrelated directory, repeated install before and after use, collision preservation, dry-run without filesystem changes, and destination symlink refusal. JSON evidence is validated against the canonical schema, including source and observation references.

An independent reviewer found three defects in the first implementation. Regression tests reproduced lost mixdown output for unsupported ZIP compression or oversized SOLNA numbers, incorrect tempo-unit labeling, and a false zero track count for missing Structure. Commit `0c38239` fixes them and adds coverage. A fresh independent reviewer approved that commit after rerunning the suite.

## Native packaging

The portable root manifest validated against the official Agent Plugins 1.0.0 schema. The bundled skill passed the Codex skill-creator validator. Claude plugin and marketplace manifests passed `claude plugin validate --strict --json` with no warnings or errors.

Both hosts installed `music-guide@music-guide-local` from this local repository through their native marketplace commands. Tests used isolated `CODEX_HOME` and `CLAUDE_CONFIG_DIR` directories, leaving normal user settings unchanged. Claude's component inventory showed one skill and no agents, hooks, or MCP servers.

For both installed host copies, every payload file matched the final source by SHA-256. Their CLIs ran from an unrelated directory with synthetic mixdown, stem, MIDI, SOLNA, DAWproject, and unsupported native project inputs. Output passed the evidence schema and exact known WAV measurement checks.

Claude's plugin validator also offered non-warning advice to add a published repository installation line. No publication URL exists, so the README documents the verified local marketplace installation instead of inventing one. No build, test, or validation warning remains outstanding. Expected source diagnostics, such as possible clipping and unsupported projects, remain visible in the JSON report.

## Real reference checks

After the synthetic suite, the user supplied three local reference sets. All original files remained read-only and outside the repository. The first set's WAV, MIDI, and SOLNA version 12 parsed successfully and its 4,077 observations validated against the evidence schema. Its MIDI tempo event and SOLNA tempo both stored 128 BPM. This comparison does not establish tempo from audio.

The second set used an MP3 plus a Logic package. Direct MP3 support remains outside the CLI contract. An available local FFmpeg decoded the MP3 into a temporary 24-bit PCM WAV. That WAV used a WAVE Extensible header, exposing a legitimate rejection in the original adapter. Synthetic PCM24 and float WAVE Extensible tests reproduced it before the format check was fixed. The derived WAV then measured successfully. The Logic package remained inventoried as unsupported.

The third set's WAV measured successfully and its Cubase project remained inventoried as unsupported. Every resulting report passed the canonical schema. No native Logic or Cubase project contents were parsed.

FFmpeg warned that the MP3 container duration was estimated from bitrate. The measurement uses the decoded WAV frame count instead of that estimate, and the conversion warning remains recorded in the private handoff report. The first reference WAV had 1,229 interleaved sample values at full scale. The tool retained a possible-clipping diagnostic without claiming audible distortion or altering the mix. Unsupported Logic/Cubase diagnostics were retained as intended fallback behavior.

## Limits

Committed fixtures remain synthetic. The local real-reference runs are additional smoke checks, not a redistributable corpus or an exhaustive format-compatibility claim. SOLNA coverage is an inspected subset, not complete format compatibility. DAWproject coverage is metadata only, with stored or deflated project.xml entries.

The CLI does not estimate key, tempo, genre, or mood from audio. It does not measure LUFS or true peak, reconstruct nested arrangements, align stems, render music, or generate artwork. The conversational skill supplies explanations and proposals within available evidence. Model response quality across real music requests has not been evaluated end to end.

The normal user profile has not been installed or configured. Nothing has been pushed, published, or submitted as a pull request.
