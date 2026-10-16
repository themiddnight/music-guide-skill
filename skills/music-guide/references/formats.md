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
| Ardour | `.ardour` | Parsed: tempo map, meter, locations, routes, plugins, region counts |
| LMMS | `.mmp`, `.mmpz` | Parsed: tempo, meter, tracks, plugins, note summaries |
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

## LMMS projects

Sources: LMMS 1.2.2 `DataFile.cpp`, `Song.cpp`, `Track.cpp`, `Note.cpp`, `MidiTime.h`, `AutomatableModel.cpp` and `MidiExport.cpp`, the LMMS 1.3 clip renames in `DataFile.cpp`, and the 1.2.2 demo projects.

LMMS parses a file as XML first and falls back to `qUncompress`, so the adapter does the same: an `.mmpz` is a 4-byte big-endian length followed by a zlib stream. Both the file and the inflated XML are limited to 32 MiB, inflation is bounded against zlib bombs, and a declared length over the limit is refused. LMMS writes a bare `<!DOCTYPE lmms-project>`; only that exact line is removed, so any other DTD or entity is still rejected. Detection needs `<?xml`, `<!DOCTYPE lmms-project`, `<lmms-project` or a valid zlib header after the length.

- `project.application` is `creator` plus `creatorversion`. `tempo` is `head/@bpm` and `meter` is `head/@timesig_numerator` over `@timesig_denominator`. When LMMS stores the tempo as a `<bpm>` element (automated or log-scaled) the value is the one at save time, the reason says so, and the warning `tempo_model_is_element` is added.
- Every `track` in the song container and, for beat/bassline tracks, every track in the nested pattern store becomes a `project.track` (type 0 instrument, 1 beat/bassline as group, 2 sample as audio, others unknown); `lmms.track_type` carries the type code, label and container. `project.plugin` lists instrument and effect names from tracks and from the mixer, `project.clip_count` counts the pattern, sample, beat/bassline and automation clips of a track, and `project.media_reference` lists `src` basenames of sample clips and instruments.
- Notes come from `note` elements in MIDI clips (`pattern`, or `midiclip` in 1.3). Ticks convert to quarter notes at 48 ticks per quarter (192 per whole note in every meter). Pitch is `key + 12`, so LMMS key 57 is A4, as in the LMMS MIDI export; base-note and master-pitch transposition are not applied. Velocity is `vol * 127 / 200`. A negative length marks a beat step note and is taken as its magnitude. Positions inside a nested beat/bassline pattern are local to that pattern, not placed on the song timeline.
- Not extracted: automation values, the placement of beat/bassline clips on the timeline, sample rate (LMMS keeps it in its settings), instrument settings, and any embedded `data`.

## Ardour sessions

Sources: Ardour 5.12, 6.9 and 8.12 `session_state.cc`, `tempo.cc` and `libs/temporal/tempo.cc`, `location.cc`, `route.cc`, `playlist.cc`, `plugin_insert.cc` and `temporal/timeline.cc`, and the [Ardour session-file notes](https://git.ardour.org/ardour/manual/raw/branch/master/include/whats-in-a-session.html).

The adapter reads the session XML only (32 MiB limit, no DTD or entities). It never opens the session's audio, MIDI or plugin files. Three `TempoMap` layouts are read. Ardour 4 and older use flat `Tempo` and `Meter` children placed by `start="1|1|0"` (`beats-per-bar` names the meter numerator). Ardour 5 and 6 add whole-note `pulse` positions. Ardour 7 and later use `Tempos` and `Meters` lists with `quarters="beats:ticks"` positions and `npm`/`note-type` tempos. Inactive tempos are skipped. Tempo is converted to quarter notes per minute as `npm * 4 / note-type`.

- `project.application` is `ProgramVersion/@modified-with` (or `created-with`), `project.sample_rate` is `Session/@sample-rate`, and `ardour.session_version` is the file version.
- `project.tempo_map` lists every active tempo start. A single tempo also gives a static `tempo`, a single meter gives a static `meter`, and several meters give `ardour.meter_map`. Ramps are reduced to their starting tempo. A session with several tempos but no readable positions gets the warning `tempo_positions_unreadable` and no tempo claim.
- `Location` marks become `project.marker`; session, loop and range locations become `ardour.session_range`, `ardour.loop_range` and `ardour.range`. Positions are sample counts (divided by the session rate), `a<superclock>` (divided by the tempo map's superclock rate) or `b<ticks>` (1920 ticks per quarter). A location whose rate is missing is skipped with the warning `locations_skipped`, never defaulted.
- Each `Route` is a `project.track`: master, audio, MIDI or bus from `PresentationInfo/@flags` (the monitor section is `unknown`); `ardour.route` keeps the type, flags and active state. `lv2`, `ladspa`, `luaproc`, VST and AudioUnit processors become `project.plugin`; built-in processors do not. `project.clip_count` is the region count of the track's current playlist; unused playlists are not counted. `project.media_reference` lists `Source` names.
- Not extracted: MIDI notes (they live in separate `.mid` source files; the info diagnostic `ardour_notes_not_read` says so), automation, region positions, VCAs, and tempo ramp shapes.

## Add an adapter

1. Add a module under `scripts/music_guide/adapters/` with a function `adapter(path, source_id, evidence, detail)` that writes to the staged `Evidence` object. Use `dawproject.py` as a boundary example.
2. Register it as one `ProjectFormat` row in `scripts/music_guide/formats.py`: a unique name (also reported as the source adapter), label, extensions, `file` or `directory`, byte limit, signature check, adapter, and export guidance. Setting `adapter` on an existing inventory row upgrades that format.
3. Read containers with the bounded helpers in `common.py` (`_read_zip_member`, `_gunzip_bounded`, `_parse_xml`) and keep embedded media paths as `_media_basename` values only.
4. Record shared facts through `ProjectFacts` in `scripts/music_guide/project.py` so every application emits the same properties: `project.application`, `tempo`, `meter`, `project.tempo_map`, `project.sample_rate`, `project.key`, `project.track_count`, `project.track`, `project.marker`, `project.plugin`, `project.media_reference`, `project.clip_count`, `project.track_notes` and, with `--detail full`, `project.note`. Summarize notes with `notes.summarize_notes`. Use `Evidence.add` directly only for format-specific facts, with an exact locator, units, scope, time domain, provenance, and confidence reason.
5. Reject unsupported variants with an actionable `ValueError`. Normalize parser-specific failures at the adapter boundary. The CLI merges staged observations only after parsing succeeds.
6. Test a representative fixture, missing fields, malformed data, unsupported variants, and the copied skill. Update this contract only after those checks pass.

Time domains in reports: `static` (one value for the scope), `midi_ticks` (positioned by MIDI ticks), `project_beats` and `project_seconds` (positioned in a project timeline), and `loop_sequence` (ordered inside a stored loop).

Do not launch a DAW, execute a plugin, decode proprietary data without a tested parser, or follow media paths as a shortcut. Ask for WAV, MIDI, stems, or an explicitly supported export instead.
