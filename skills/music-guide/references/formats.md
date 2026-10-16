# Input format contract

| Flag | Accepted content | Bound | Evidence |
| --- | --- | --- | --- |
| `--mixdown` | Required WAV PCM or IEEE float | 4 GiB | Duration, channels, sample rate, sample peak dBFS, RMS dBFS, full-scale sample count |
| `--stem` | Optional WAV PCM or IEEE float | 4 GiB each | Same measurements, independent source scope |
| `--midi` | SMF 0/1 with ticks per quarter note | 32 MiB each | Stored tempo/meter events, track names, paired notes in ticks and beats |
| `--solna` | JSON formatVersion <= 12 with validated inspected fields | 8 MiB each | Stored BPM, meter ID, loop scale IDs and chords |
| `--dawproject` | ZIP with one stored or deflated project.xml | 64 MiB archive, 8 MiB XML, 256 entries | Project version, direct transport tempo/meter, structure track names/count |
| `--project` | Unsupported native project or package to inventory | No parsing | Diagnostic and export guidance |
| `--declarations` | JSON array under declarations schema | 8 MiB | User supplied static facts |

Parser bounds live in `scripts/music_guide/adapters/common.py` and the CLI role limits in `scripts/music_guide/cli.py`. The archive parser reads only `project.xml` and rejects DTD, entities, and external references. MIDI tempo and meter events are reported as `midi.tempo_map` and `midi.meter_map` lists of `{tick, beat, ...}` entries. Only when a file has exactly one `set_tempo` event across all tracks, at tick 0, does it also produce a static `tempo` claim (method `midi-constant-tempo`); a single `time_signature` at tick 0 likewise produces a static `meter` claim (method `midi-constant-meter`). A changing or late tempo is never reduced to a static claim, because the SMF default before the first event is not evidence. Compare declarations against such maps by hand. No source can establish rendered sound except the supplied audio samples, and the CLI does not infer timbre or meaning from them.

DAWproject transport tempo requires an explicit `unit="bpm"`. Other units and missing units are rejected rather than guessed. A missing `Structure` produces no track-count claim. An explicitly empty `Structure` has a count of zero. Automatic conflict detection compares shared static tempo and meter claims with matching scopes and units whose provenance is measured, extracted, or user declared. Inferred estimates never raise automatic conflicts. Tempo values within 0.01 BPM count as equal because MIDI stores whole microseconds per beat (92 BPM reads back as 91.99999 BPM). Meter must match exactly. Other properties and event maps require explicit human comparison.

## Add an adapter

1. Add a module under `scripts/music_guide/adapters/` with a function taking `Path`, source ID, and a staged `Evidence` object. Use `audio.py` or `dawproject.py` as a boundary example.
2. Register its role in `ADAPTERS`, its bound in `LIMITS`, and its argument in `cli.py`. Use the constants in `common.py` as the single source for byte limits.
3. Add observations through `Evidence.add`. Include source, exact locator, units, scope, time domain, method version, provenance, and a confidence reason. Store file-specific inventories separately from shared song assertions.
4. Reject unsupported variants with an actionable `ValueError`. Normalize parser-specific failures at the adapter boundary. The CLI merges staged observations only after parsing succeeds.
5. Test a representative fixture, missing fields, malformed data, unsupported variants, and the copied skill. Update this contract and the roadmap only after those checks pass.

Do not launch a DAW, execute a plugin, decode proprietary data without a tested parser, or follow media paths as a shortcut. Ask for WAV, MIDI, stems, or an explicitly supported export instead.
