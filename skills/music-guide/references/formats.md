# Input format contract

| Flag | Accepted content | Bound | Evidence |
| --- | --- | --- | --- |
| `--mixdown` | Optional WAV PCM or IEEE float | 4 GiB | Duration, channels, sample rate, sample peak dBFS, RMS dBFS, full-scale sample count |
| `--stem` | Optional WAV PCM or IEEE float | 4 GiB each | Same measurements, independent source scope |
| `--midi` | SMF 0/1 with ticks per quarter note | 32 MiB each | Tempo and meter maps, static tempo/meter for constant files, track names, per-track note summaries; paired notes with `--detail full` |
| `--solna` | JSON formatVersion <= 12 with validated inspected fields | 8 MiB each | Stored BPM, meter ID, loop scale IDs and chords |
| `--dawproject` | ZIP with one stored or deflated project.xml | 64 MiB archive, 8 MiB XML, 256 entries | Project version, application, direct transport tempo/meter, track count, and each track's name and kind |
| `--project` | Any project file or package; the format is detected from the extension and content signature | Per format in `formats.py` | Parsed evidence for a supported format; otherwise an inventory with format-specific export guidance |
| `--declarations` | JSON array under declarations schema | 8 MiB | User supplied static facts |
| `--detail` | `summary` (default) or `full` | Not applicable | `full` adds one observation per stored note |

At least one of `--mixdown`, `--stem`, `--midi`, `--solna`, `--project` or `--dawproject` is required; declarations alone are not enough. Without a mixdown the report carries a `no_mixdown` info diagnostic, because the rendered sound is unknown. A supplied mixdown that cannot be read is still fatal.

`--project` accepts these formats. A directory package must be passed as the directory. An extension whose content does not match the format's signature is reported as a source error rather than guessed. An unknown extension is inventoried with generic export guidance.

| Format | Extensions | Support |
| --- | --- | --- |
| DAWproject | `.dawproject` | Parsed |
| Standard MIDI file | `.mid`, `.midi` | Parsed |
| SOLNA | `.solna` | Parsed |
| Pro Tools | `.ptx`, `.ptf`, `.pts` | Inventory; export AAF or stems and MIDI |
| Bitwig Studio | `.bwproject` | Inventory; export DAWproject |
| Reason | `.reason` | Inventory; export stems and MIDI |
| Logic Pro | `.logicx` directory | Inventory; export stems and MIDI |
| GarageBand | `.band` directory | Inventory; export audio |
| Cubase, Nuendo | `.cpr`, `.npr` | Inventory; export DAWproject (Cubase 14+) or MIDI |
| Ableton Live | `.als` | Parsed |
| REAPER | `.rpp` | Parsed |
| FL Studio | `.flp` | Inventory; export stems and MIDI |
| Studio One | `.song` | Inventory; export DAWproject (6.5+) or stems and MIDI |
| Ardour | `.ardour` | Inventory; export stems and MIDI |
| LMMS | `.mmp`, `.mmpz` | Inventory; export audio and MIDI |
| AAF, OMF | `.aaf`, `.omf` | Inventory |
| MusicXML | `.musicxml`, `.mxl` | Inventory; export MIDI |
| MuseScore | `.mscz` | Inventory; export MIDI |

Parser bounds live in `scripts/music_guide/adapters/common.py`; each format's byte limit is its row in `scripts/music_guide/formats.py`. The archive parser reads only `project.xml` and rejects DTD, entities, and external references. MIDI tempo and meter events are reported as `midi.tempo_map` and `midi.meter_map` lists of `{tick, beat, ...}` entries. Only when a file has exactly one `set_tempo` event across all tracks, at tick 0, does it also produce a static `tempo` claim (method `midi-constant-tempo`); a single `time_signature` at tick 0 likewise produces a static `meter` claim (method `midi-constant-meter`). A changing or late tempo is never reduced to a static claim, because the SMF default before the first event is not evidence. Compare declarations against such maps by hand. No source can establish rendered sound except the supplied audio samples, and the CLI does not infer timbre or meaning from them.

DAWproject transport tempo requires an explicit `unit="bpm"`. Other units and missing units are rejected rather than guessed. A missing `Structure` produces no track-count claim. An explicitly empty `Structure` has a count of zero. Automatic conflict detection compares shared static tempo and meter claims with matching scopes and units whose provenance is measured, extracted, or user declared. Inferred estimates never raise automatic conflicts. Tempo values within 0.01 BPM count as equal because MIDI stores whole microseconds per beat (92 BPM reads back as 91.99999 BPM). Meter must match exactly. Other properties and event maps require explicit human comparison.

## Add an adapter

1. Add a module under `scripts/music_guide/adapters/` with a function `adapter(path, source_id, evidence, detail)` that writes to the staged `Evidence` object. Use `dawproject.py` as a boundary example.
2. Register it as one `ProjectFormat` row in `scripts/music_guide/formats.py`: a unique name (also reported as the source adapter), label, extensions, `file` or `directory`, byte limit, signature check, adapter, and export guidance. Setting `adapter` on an existing inventory row upgrades that format.
3. Read containers with the bounded helpers in `common.py` (`_read_zip_member`, `_gunzip_bounded`, `_parse_xml`) and keep embedded media paths as `_media_basename` values only.
4. Record shared facts through `ProjectFacts` in `scripts/music_guide/project.py` so every application emits the same properties: `project.application`, `tempo`, `meter`, `project.tempo_map`, `project.sample_rate`, `project.key`, `project.track_count`, `project.track`, `project.marker`, `project.plugin`, `project.media_reference`, `project.clip_count`, `project.track_notes` and, with `--detail full`, `project.note`. Summarize notes with `notes.summarize_notes`. Use `Evidence.add` directly only for format-specific facts, with an exact locator, units, scope, time domain, provenance, and confidence reason.
5. Reject unsupported variants with an actionable `ValueError`. Normalize parser-specific failures at the adapter boundary. The CLI merges staged observations only after parsing succeeds.
6. Test a representative fixture, missing fields, malformed data, unsupported variants, and the copied skill. Update this contract only after those checks pass.

## Ableton Live sets and REAPER projects

Both formats are read as stored intent only. The adapters never open referenced media, load a plugin, or follow a path, and they keep embedded media paths as basenames. Both emit the shared `project.*` vocabulary: application, tempo, meter, tempo map, markers, tracks, plugins, media references, clip counts and per-track note summaries. An unreadable file becomes a `optional_source_error` and the rest of the report survives.

**Ableton Live (`.als`).** The file is gzip XML, decompressed with a 64 MiB bound. Application comes from the root `Creator`, and `ableton.version` keeps the schema attributes. Tempo comes from `Tempo/Manual` on `MasterTrack` (Live 11 and earlier) or `MainTrack` (Live 12). Tempo automation on that track becomes `project.tempo_map` in beats; a static `tempo` claim exists only when the automation holds one value or is absent, and the Manual value is kept as `ableton.manual_tempo` beside a changing map. Meter is decoded from the master `TimeSignature` integer code (4/4 is 201, 7/8 is 303), with confidence `low` because Ableton does not document the encoding. It falls back to explicit `TimeSignatures/RemoteableTimeSignature` Numerator and Denominator pairs, and meter automation becomes `ableton.meter_map`. Tracks are `AudioTrack`, `MidiTrack`, `GroupTrack`, `ReturnTrack` and the master, named from `EffectiveName` then `UserName`. Locators become beat markers. Devices become `project.plugin` entries named from `PlugName`, `Vst3PluginInfo/Name`, `AuPluginInfo/Name`, or the native device element name. Samples come from `SampleRef/FileRef` (`Name`, then `Path`, then `RelativePath`). A Live 12 `ScaleInformation` root and name becomes `project.key` with confidence `low`, because Live may store a default scale when Scale Mode is unused.

`project.clip_count` counts arrangement clips; session-view slot clips are counted in `ableton.session_clip_count`. MIDI note summaries use `Notes/KeyTracks/KeyTrack/MidiKey` with `Notes/MidiNoteEvent` `Time`, `Duration` and `Velocity`. Arrangement notes are positioned at the clip `CurrentStart` plus the note time relative to the clip start marker (`LoopStart + StartRelative`), limited to the clip's first pass, and reported as `project.track_notes` with confidence `medium`. Session-clip notes have no arrangement position, so they are summarized separately as `ableton.session_clip_notes` and `--detail full` lists arrangement notes only. Loops are not unrolled, muted notes are excluded, and warping and tempo changes are not applied. A clip without readable loop and position fields is skipped and counted in the confidence reason. The sample rate is not stored in a set, so no `project.sample_rate` is produced.

**REAPER (`.rpp`).** The file is a text tree of `<TAG ...>` blocks closed by `>`. The reader follows REAPER's quoting (`"`, `'` or a backtick opens a string that ends at the next matching character, with no escapes), skips `|` text lines and long single-token data lines, and refuses files over 1,500,000 property lines, 64 levels of nesting, binary bytes, unbalanced blocks or an unterminated string. The header gives the application and `reaper.header`. `SAMPLERATE` gives the sample rate and `TEMPO` gives tempo and meter. Points in `TEMPOENVEX` (`PT position bpm ...`, positions in seconds) become `project.tempo_map` in beats by treating each point as a tempo step, so ramps are not interpolated. A static `tempo` claim follows the same rule as Ableton, with the `TEMPO` line kept as `reaper.base_tempo` beside a changing map. An envelope stored as inactive (`ACT 0`) is ignored with a warning, and time-signature changes on the envelope are not decoded. `MARKER` lines become `project.marker` in seconds; two lines sharing an index with the region flag become one `reaper.region`.

`TRACK` blocks give names, and `ISBUS 1` marks a folder; REAPER stores no track type, so every other track is `unknown`. Effects inside any `FXCHAIN`, `TAKEFX` or master chain (`VST`, `AU`, `JS`, `CLAP`, `DX`, `LV2`) become `project.plugin` entries named by their first argument. `ITEM` blocks give `project.clip_count`, and `SOURCE ... FILE` lines give media basenames from every take. Notes come from `SOURCE MIDI` blocks with `HASDATA 1 <ppq> QN`: `E` and `e` lines carry delta ticks and hex status and data bytes, paired by channel and pitch. Only the selected take is read, muted items are skipped, and note positions are the item `POSITION` converted to beats plus ticks over ppq. Source start offsets, item loops and tempo-ignoring sources are not applied, and notes past the item end are dropped. The confidence reason of `project.track_notes` reports these counts and whether a tempo map was integrated.

Parsing uses memory in proportion to the file; a 38 MB XML set with 500,000 notes peaks near 520 MB, and a 26 MB project with 1.2 million MIDI event lines near 565 MB.

Format uncertainty. No public specification exists for either format, and no real `.als` or `.rpp` file was available. The structures above come from the sources listed next, and the tests use fixtures built from them. Unverified points are the Ableton meter integer code, the exact clip start-marker and loop semantics, the location of `RemoteableTimeSignature` outside clips, the REAPER `TAKE SEL` convention, the `ACT` flag meaning, and the sub-fields of `MARKER` and `PT` beyond those used here.

Sources: Ableton `.als`: github.com/offlinemark/dawtool (`dawtool/daw/ableton.py`: root attributes, `Tempo/Manual`, tempo envelopes by `PointeeId`, locators), github.com/jake-g/ableton-parser (`parse_projects.py`: `MasterTrack/DeviceChain/Mixer`, `Tracks/*`, `ScaleInformation`, `PluginDesc` infos, `SampleRef/FileRef`, `MainSequencer`), github.com/andrewcb/alsd (`alsd.py`: `Notes/KeyTracks/KeyTrack`, `MidiKey`, `MidiNoteEvent`, `Loop`, `CurrentStart`), pypi.org/project/daw-schemas (structural schema of Live 12 sets), help.ableton.com (numerator 1 to 99 and denominators 1, 2, 4, 8, 16). REAPER `.rpp`: github.com/Perlence/rpp (lexer quoting rules), github.com/CharlesHolbrow/rppp (`rpp-examples/midi-item.RPP`, `two-vst3s.RPP`: `HASDATA 1 960 QN`, `E` and `e` lines, `FXCHAIN` and `VST` blocks, `TEMPO`, `SAMPLERATE`, `ISBUS`).

Time domains in reports: `static` (one value for the scope), `midi_ticks` (positioned by MIDI ticks), `project_beats` and `project_seconds` (positioned in a project timeline), and `loop_sequence` (ordered inside a stored loop).

Do not launch a DAW, execute a plugin, decode proprietary data without a tested parser, or follow media paths as a shortcut. Ask for WAV, MIDI, stems, or an explicitly supported export instead.
