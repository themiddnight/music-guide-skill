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
| Logic Pro | `.logicx` directory | Parsed: metadata plists per alternative; tracks, regions and MIDI need an export |
| GarageBand | `.band` directory | Parsed when `Alternatives/NNN/MetaData.plist` exists (GarageBand 10.0.3+); older layouts need an audio export |
| Cubase, Nuendo | `.cpr`, `.npr` | Inventory; export DAWproject (Cubase 14+) or MIDI |
| Ableton Live | `.als` | Inventory; export stems and MIDI |
| REAPER | `.rpp` | Inventory; render stems and export MIDI |
| FL Studio | `.flp` | Inventory; export stems and MIDI |
| Studio One | `.song` | Inventory; export DAWproject (6.5+) or stems and MIDI |
| Ardour | `.ardour` | Inventory; export stems and MIDI |
| LMMS | `.mmp`, `.mmpz` | Inventory; export audio and MIDI |
| AAF, OMF | `.aaf`, `.omf` | Inventory |
| MusicXML | `.musicxml`, `.mxl` | Inventory; export MIDI |
| MuseScore | `.mscz` | Inventory; export MIDI |

Parser bounds live in `scripts/music_guide/adapters/common.py`; each format's byte limit is its row in `scripts/music_guide/formats.py`. The archive parser reads only `project.xml` and rejects DTD, entities, and external references. MIDI tempo and meter events are reported as `midi.tempo_map` and `midi.meter_map` lists of `{tick, beat, ...}` entries. Only when a file has exactly one `set_tempo` event across all tracks, at tick 0, does it also produce a static `tempo` claim (method `midi-constant-tempo`); a single `time_signature` at tick 0 likewise produces a static `meter` claim (method `midi-constant-meter`). A changing or late tempo is never reduced to a static claim, because the SMF default before the first event is not evidence. Compare declarations against such maps by hand. No source can establish rendered sound except the supplied audio samples, and the CLI does not infer timbre or meaning from them.

DAWproject transport tempo requires an explicit `unit="bpm"`. Other units and missing units are rejected rather than guessed. A missing `Structure` produces no track-count claim. An explicitly empty `Structure` has a count of zero. Automatic conflict detection compares shared static tempo and meter claims with matching scopes and units whose provenance is measured, extracted, or user declared. Inferred estimates never raise automatic conflicts. Tempo values within 0.01 BPM count as equal because MIDI stores whole microseconds per beat (92 BPM reads back as 91.99999 BPM). Meter must match exactly. Other properties and event maps require explicit human comparison.

## Logic Pro and GarageBand packages

Sources: the observed `.logicx` layout of Logic Pro 12.4 and the [MultimediaWiki GarageBand page](https://wiki.multimedia.cx/index.php/GarageBand). GarageBand 10.0.3 and later uses the same package layout as Logic Pro.

The adapter reads only `Resources/ProjectInformation.plist` and each `Alternatives/NNN/MetaData.plist`, as XML or binary property lists of at most 4 MiB. It never opens `ProjectData` (undocumented binary), `Media/` files, or any path a plist lists. A symbolic link at any step of a read path rejects the whole package. A package without a readable `MetaData.plist` (older GarageBand, iOS GarageBand) is an error with export guidance.

- `project.application` comes from `LastSavedFrom`. Each alternative is its own scope, `alternative:NNN`, named by `VariantNames` in `logic.alternative_name` (`garageband.*` for `.band`). Compare a declaration against it with the same scope, for example `alternative:000`.
- `tempo` (`BeatsPerMinute`), `meter` (`SongSignatureNumerator` and `SongSignatureDenominator`), `project.key` (`SongKey` and `SongGenderKey`, joined as "D minor"), `project.sample_rate` and `project.track_count` (`NumberOfTracks`) are stored settings per alternative. The tempo is the project start tempo; a tempo map in `ProjectData` is not read. Logic stores a key even when the user never set one, so treat `project.key` as stored intent. A key with only one half present is reported as `logic.partial_key` instead.
- `project.media_reference` lists basenames from `AudioFiles`, `UnusedAudioFiles`, `PlaybackFiles`, `SamplerInstrumentsFiles`, `QuicksamplerFiles`, `ImpulsResponsesFiles` and `UltrabeatFiles`; the list name is the end of the locator and `logic.media_list` carries the role and path count. `logic.bundled_media` counts regular files under `Media/` by name only; links are counted but not followed.
- Not extracted: track names and kinds, regions, MIDI notes, plugins, tempo and meter changes, markers. An info diagnostic `project_data_not_read` says so.

## Add an adapter

1. Add a module under `scripts/music_guide/adapters/` with a function `adapter(path, source_id, evidence, detail)` that writes to the staged `Evidence` object. Use `dawproject.py` as a boundary example.
2. Register it as one `ProjectFormat` row in `scripts/music_guide/formats.py`: a unique name (also reported as the source adapter), label, extensions, `file` or `directory`, byte limit, signature check, adapter, and export guidance. Setting `adapter` on an existing inventory row upgrades that format.
3. Read containers with the bounded helpers in `common.py` (`_read_zip_member`, `_gunzip_bounded`, `_parse_xml`) and keep embedded media paths as `_media_basename` values only.
4. Record shared facts through `ProjectFacts` in `scripts/music_guide/project.py` so every application emits the same properties: `project.application`, `tempo`, `meter`, `project.tempo_map`, `project.sample_rate`, `project.key`, `project.track_count`, `project.track`, `project.marker`, `project.plugin`, `project.media_reference`, `project.clip_count`, `project.track_notes` and, with `--detail full`, `project.note`. Summarize notes with `notes.summarize_notes`. Use `Evidence.add` directly only for format-specific facts, with an exact locator, units, scope, time domain, provenance, and confidence reason.
5. Reject unsupported variants with an actionable `ValueError`. Normalize parser-specific failures at the adapter boundary. The CLI merges staged observations only after parsing succeeds.
6. Test a representative fixture, missing fields, malformed data, unsupported variants, and the copied skill. Update this contract only after those checks pass.

Time domains in reports: `static` (one value for the scope), `midi_ticks` (positioned by MIDI ticks), `project_beats` and `project_seconds` (positioned in a project timeline), and `loop_sequence` (ordered inside a stored loop).

Do not launch a DAW, execute a plugin, decode proprietary data without a tested parser, or follow media paths as a shortcut. Ask for WAV, MIDI, stems, or an explicitly supported export instead.
