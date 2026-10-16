# Music Guide MVP design

Approved in chat on 2026-10-16. Build a local Music Intelligence assistant for Claude Code and Codex. Cover analyze, explain, critique, recommend, compare, and create across composition, theory, arrangement, production, mixing, mastering, mood, genre, and artwork direction.

## Architecture

Use one portable skill payload under `skills/music-guide/`, with modular Python adapters, schemas, requirements, and progressively loaded references inside it. Plugin manifests and an installer wrap the same payload. No MCP server or paid API is needed. Standalone copied skills must run without the repository.

A CLI creates JSON evidence. The conversational skill interprets that evidence and the user's intent. Do not pretend the CLI listens semantically or generates finished audio or artwork.

## Data contract

A versioned document contains sources, observations, conflicts, diagnostics, and limitations. Source identities include SHA-256, basename, role, adapter, and status. Observations retain source IDs, method/version, locator, scope, value/unit, provenance (`measured`, `extracted`, `inferred`, `user_declared`), and confidence with a reason. Confidence is an ordinal assessment of the stated claim, not a calibrated probability. Missing facts remain absent. Resolve nothing by universal source ranking. Compare only compatible properties, scopes, units, and time domains. Preserve disagreement and mark unresolved conflicts.

## Supported inputs

Require a readable mixdown for track-specific CLI analysis. General theory, composition, and creative requests need no audio. WAV is the guaranteed MVP audio input; declare exact codecs and precision. Measure duration, channels, sample rate, sample peak and RMS with units. Never label RMS as LUFS or sample peak as true peak. Silence must serialize as valid JSON. Additional audio support is optional only if dependency-backed and tested.

MIDI SMF 0/1 supports tempo, meter, track metadata, and note events in ticks or beats with explicit time semantics. Type 2 and SMPTE may be rejected explicitly. Never infer sound or audible balance from MIDI.

SOLNA supports an inspected subset of `formatVersion`, `content.bpm`, `content.meterId`, `content.loops`, each loop's `scaleRoot`, `scaleType`, `chords` with `root`, `quality`, `bars`, and optional `bassNote`. Local source inspection established current formatVersion 12. Extract stored fields only, validate values, reject unknown future versions, and do not resolve proprietary pattern tables or claim full playback equivalence.

DAWproject reads bounded ZIP `project.xml` metadata only: project version, Transport Tempo/TimeSignature, track names/count. Do not reconstruct arrangement or notes until their nested timing semantics are tested. Do not extract media or follow paths. Proprietary Logic/Cubase/FL/Ableton projects are inventoried unsupported with export guidance.

Optional sources enrich progressively. Corrupt or missing optional files become diagnostics while valid mixdown evidence survives. Unreadable required mixdown is an actionable nonzero CLI error. Bound untrusted JSON/XML/archive inputs and reject nonfinite values.

## Installation and verification

Provide Codex and Claude compatibility manifests, Claude local marketplace metadata, and portable root plugin metadata if supported. A Python installer supports a user-supplied destination, dry-run, idempotent installation, and collision refusal without deletion. Do not change this user's real host configuration during tests. Document plugin install and standalone skill install separately and do not invent a published repository URL.

Test synthetic WAV, MIDI, SOLNA, and DAWproject, invalid/partial data, provenance, conflicts, output schema, and installed-payload execution from an unrelated directory. No personal paths or source audio in fixtures. Verify on macOS here; report other OS and real host discovery unverified. Commit locally on a feature branch. No push, PR, or publishing.

## Alternatives and synthesis

Typed internal records keep observation invariants local. Schema-first dictionaries reduce types but spread keys and validation. Choose small typed records with a JSON Schema checked by tests, plus a static extension-to-adapter mapping rather than a dynamic plugin framework. Both independent design sketches agreed on standalone payload and explicit provenance. DAWproject metadata-only support avoids unverified timing reconstruction.

## Roadmap

Real-world fixture coverage with permission; broader audio codecs and validated LUFS/true peak; tempo/key estimation with uncertainty; deeper DAWproject event timing; additional verified adapters. No automatic proprietary-format parsing claims.
