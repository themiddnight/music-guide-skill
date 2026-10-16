# Input format contract

| Flag | Accepted content | Bound | Evidence |
| --- | --- | --- | --- |
| `--mixdown` | Optional WAV PCM or IEEE float | 4 GiB | Duration, channels, sample rate, sample peak dBFS, RMS dBFS, full-scale sample count |
| `--stem` | Optional WAV PCM or IEEE float | 4 GiB each | Same measurements, independent source scope |
| `--midi` | SMF 0/1 with ticks per quarter note | 32 MiB each | Tempo and meter maps, static tempo/meter for constant files, track names, per-track note summaries; paired notes with `--detail full` |
| `--solna` | JSON formatVersion <= 12 with validated inspected fields | 8 MiB each | Stored BPM, meter ID, loop scale IDs and chords |
| `--dawproject` | ZIP with one stored or deflated project.xml and an optional metadata.xml | 64 MiB archive, 8 MiB XML, 256 entries | Project version, application, metadata, transport tempo/meter, tempo and meter automation, tracks with plugins, clip counts, notes, markers and media basenames |
| `--project` | Any project file or package; the format is detected from the extension and content signature | Per format in `formats.py` | Parsed evidence for a supported format; otherwise an inventory with format-specific export guidance |
| `--declarations` | JSON array under declarations schema | 8 MiB | User supplied static facts |
| `--detail` | `summary` (default) or `full` | Not applicable | `full` adds one observation per stored note |

At least one of `--mixdown`, `--stem`, `--midi`, `--solna`, `--project` or `--dawproject` is required; declarations alone are not enough. Without a mixdown the report carries a `no_mixdown` info diagnostic, because the rendered sound is unknown. A supplied mixdown that cannot be read is still fatal.

`--project` accepts these formats. A directory package must be passed as the directory. An extension whose content does not match the format's signature is reported as a source error rather than guessed. An unknown extension is inventoried with generic export guidance.

| Format | Extensions | Support |
| --- | --- | --- |
| DAWproject | `.dawproject` | Parsed: metadata, tempo and meter maps, tracks, plugins, clips, notes, markers, media basenames |
| Standard MIDI file | `.mid`, `.midi` | Parsed |
| SOLNA | `.solna` | Parsed |
| Pro Tools | `.ptx`, `.ptf`, `.pts` | Inventory; `.ptx` and `.ptf` must carry the ptformat header; export AAF or stems and MIDI |
| Bitwig Studio | `.bwproject` | Inventory; export DAWproject |
| Reason | `.reason` | Inventory; export stems and MIDI |
| Logic Pro | `.logicx` directory | Inventory; export stems and MIDI |
| GarageBand | `.band` directory | Inventory; export audio |
| Cubase, Nuendo | `.cpr`, `.npr` | Inventory; export DAWproject (Cubase 14+) or MIDI |
| Ableton Live | `.als` | Inventory; export stems and MIDI |
| REAPER | `.rpp` | Inventory; render stems and export MIDI |
| FL Studio | `.flp` | Inventory; export stems and MIDI |
| Studio One | `.song` | Inventory; export DAWproject (6.5+) or stems and MIDI |
| Ardour | `.ardour` | Inventory; export stems and MIDI |
| LMMS | `.mmp`, `.mmpz` | Inventory; export audio and MIDI |
| AAF | `.aaf` | Parsed: application, top-level compositions, sound and picture tracks, source clip counts, markers, media basenames, sample rate |
| OMF | `.omf` | Inventory |
| MusicXML | `.musicxml`, `.mxl` | Parsed: parts, notes, tempo, meter, key, rehearsal marks |
| MuseScore | `.mscz` | Parsed: parts, notes, tempo, meter, key, rehearsal marks |

Parser bounds live in `scripts/music_guide/adapters/common.py`; each format's byte limit is its row in `scripts/music_guide/formats.py`. The DAWproject parser reads only `project.xml` and an optional `metadata.xml`, and rejects DTD, entities, and external references. MIDI tempo and meter events are reported as `midi.tempo_map` and `midi.meter_map` lists of `{tick, beat, ...}` entries. Only when a file has exactly one `set_tempo` event across all tracks, at tick 0, does it also produce a static `tempo` claim (method `midi-constant-tempo`); a single `time_signature` at tick 0 likewise produces a static `meter` claim (method `midi-constant-meter`). A changing or late tempo is never reduced to a static claim, because the SMF default before the first event is not evidence. Compare declarations against such maps by hand. No source can establish rendered sound except the supplied audio samples, and the CLI does not infer timbre or meaning from them.

DAWproject transport tempo requires an explicit `unit="bpm"`. Other units and missing units are rejected rather than guessed. A missing `Structure` produces no track-count claim. An explicitly empty `Structure` has a count of zero. Automatic conflict detection compares shared static tempo and meter claims with matching scopes and units whose provenance is measured, extracted, or user declared. Inferred estimates never raise automatic conflicts. Tempo values within 0.01 BPM count as equal because MIDI stores whole microseconds per beat (92 BPM reads back as 91.99999 BPM). Meter must match exactly. Other properties and event maps require explicit human comparison.

## Add an adapter

1. Add a module under `scripts/music_guide/adapters/` with a function `adapter(path, source_id, evidence, detail)` that writes to the staged `Evidence` object. Use `dawproject.py` as a boundary example.
2. Register it as one `ProjectFormat` row in `scripts/music_guide/formats.py`: a unique name (also reported as the source adapter), label, extensions, `file` or `directory`, byte limit, signature check, adapter, and export guidance. Setting `adapter` on an existing inventory row upgrades that format.
3. Read containers with the bounded helpers in `common.py` (`_read_zip_member`, `_gunzip_bounded`, `_parse_xml`) and keep embedded media paths as `_media_basename` values only.
4. Record shared facts through `ProjectFacts` in `scripts/music_guide/project.py` so every application emits the same properties: `project.application`, `tempo`, `meter`, `project.tempo_map`, `project.sample_rate`, `project.key`, `project.track_count`, `project.track`, `project.marker`, `project.plugin`, `project.media_reference`, `project.clip_count`, `project.track_notes` and, with `--detail full`, `project.note`. Summarize notes with `notes.summarize_notes`. Use `Evidence.add` directly only for format-specific facts, with an exact locator, units, scope, time domain, provenance, and confidence reason.
5. Reject unsupported variants with an actionable `ValueError`. Normalize parser-specific failures at the adapter boundary. The CLI merges staged observations only after parsing succeeds.
6. Test a representative fixture, missing fields, malformed data, unsupported variants, and the copied skill. Update this contract only after those checks pass.

Time domains in reports: `static` (one value for the scope), `midi_ticks` (positioned by MIDI ticks), `project_beats` and `project_seconds` (positioned in a project timeline), and `loop_sequence` (ordered inside a stored loop).

Do not launch a DAW, execute a plugin, decode proprietary data without a tested parser, or follow media paths as a shortcut. Ask for WAV, MIDI, stems, or an explicitly supported export instead.

## Scores, AAF and DAWproject details

**MusicXML** (`.musicxml`, `.mxl`). Partwise and timewise scores, plain or compressed (the rootfile named by `META-INF/container.xml`), up to 16 MiB of XML. The standard public DOCTYPE that exporters write is removed before parsing, so a DOCTYPE with an internal subset or any other external reference is still rejected. Each part is a track named by `part-name`. Beats are quarter notes from `divisions`. Chords share a start, `backup` and `forward` move the cursor, and `tie` chains merge into one note. Written pitch is converted to sounding pitch with `transpose`. Tempo comes from `sound/@tempo`, or from `metronome` marks only when no playback tempo exists (beat unit and dots are converted to quarter-note BPM). Key comes from `fifths` and `mode`, and only from non-transposing parts. Rehearsal marks become markers. Not read: repeats and jumps (never unrolled), grace notes, unpitched notes, `offset` on directions, per-note articulation, and the MusicXML dynamics model beyond a note's `dynamics` attribute. A velocity placeholder of 90 stands in where none is stored, and an info diagnostic says so.

**MuseScore** (`.mscz`). A zip holding one `.mscx` (named in `META-INF/container.xml`, otherwise the only top-level `.mscx`). Notes use MuseScore's concert pitch, `durationType` plus dots, the stored `duration` fraction, and tuplet ratios from `Tuplet` and `endTuplet`. Ties come from `Spanner type="Tie"` with `next` and `prev`. Tempo is the stored beats per second times 60. Not read: repeats and jumps, grace notes, notes marked as not played, measure-repeat contents, and the older pre-3.0 tie encoding. A velocity placeholder of 80 stands in where none is stored. The MuseScore file format is not formally specified, so the layout follows the open-source reader and was tested on synthetic files only.

**AAF** (`.aaf`, binary structured storage). pyaaf2 opens the file read-only inside a child process that is stopped after 60 seconds or 1.5 GiB, because pyaaf2 sizes buffers from counts inside the file. The application is the latest Identification. Tracks are the sound and picture timeline slots of top-level composition mobs, with their edit rate and length in `aaf.slot`. Clip counts are source clips with a non-null source. Markers are comment or descriptive markers in event slots, converted to seconds. Media references are network locator basenames of source mobs. Sample rate comes from sound edit rates of 8 kHz or more and sound descriptors, and is reported only when they all agree. Container inconsistencies pyaaf2 logs become `aaf_container_warning` diagnostics. AAF stores no tempo or meter, so none is reported. AAF XML, OMF and embedded essence are not read.

**DAWproject**. A missing `Structure` still gives no track count. Tempo automation needs `unit="bpm"` and becomes `project.tempo_map` (beats). Time signature automation becomes `dawproject.meter_map`. Clip counts cover arrangement clips only (clips nested inside a clip are content, and scenes are ignored). Notes are placed on the arrangement in quarter-note beats: a clip's `time` and `playStart` anchor its content, notes outside the played window are dropped or cut at the clip end, and `contentTimeUnit="seconds"` content is converted with the stored tempo. Approximations: tempo ramps use the mean of the two segment tempos, tempo before the first automation point is the first value, loops are not unrolled (a diagnostic counts looped clips), and clips that reference shared content by id or time-warped note timelines are skipped with a warning. A missing note velocity becomes the placeholder 100.

**Sources.** W3C MusicXML 4.0 reference (https://www.w3.org/2021/06/musicxml40/), MuseScore 3.6 reader sources (`libmscore/measure.cpp`, `chordrest.cpp`, `note.cpp`, `tempotext.cpp`), pyaaf2 (https://github.com/markreidvfx/pyaaf2, MIT) and the AAF object specification, DAWproject `Project.xsd`, `MetaData.xsd` and the Java `Timeline` and `Clip` sources (https://github.com/bitwig/dawproject), and ptformat (https://github.com/zamaudio/ptformat) for the Pro Tools header.

**Signatures.** Pro Tools `.ptx` and `.ptf` must start with byte `0x03` or hold the 16-character bit string `0010111100101011` at offset 1 of the first 256 bytes, which is the check ptformat applies for Pro Tools 5 to 12. Pro Tools session version is not extracted, because reading it needs the session de-obfuscated and no real session was available to test that. No source confirms a `.bwproject` or `.reason` header (the published Reason signature belongs to the older `.rns`), so those two accept any content.
