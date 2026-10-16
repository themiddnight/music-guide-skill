# Input format contract

| Flag | Accepted content | Bound | Evidence |
| --- | --- | --- | --- |
| `--mixdown` | Required WAV PCM or IEEE float | 4 GiB | Duration, channels, sample rate, sample peak dBFS, RMS dBFS, full-scale sample count |
| `--stem` | Optional WAV PCM or IEEE float | 4 GiB each | Same measurements, independent source scope |
| `--midi` | SMF 0/1 with ticks per quarter note | 32 MiB each | Stored tempo/meter events, track names, paired notes in ticks and beats |
| `--solna` | JSON formatVersion <= 12 with validated inspected fields | 8 MiB each | Stored BPM, meter ID, loop scale IDs and chords |
| `--dawproject` | ZIP with one project.xml | 64 MiB archive, 8 MiB XML, 256 entries | Project version, direct transport tempo/meter, structure track names/count |
| `--project` | Unsupported native project or package to inventory | No parsing | Diagnostic and export guidance |
| `--declarations` | JSON array under declarations schema | 8 MiB | User supplied static facts |

Parser bounds live in `scripts/music_guide/adapters/common.py` and the CLI role limits in `scripts/music_guide/cli.py`. The archive parser reads only `project.xml` and rejects DTD, entities, and external references. MIDI tempo and meter events retain their tick locations; even a single event is not silently converted to a static tempo claim. Static user declarations therefore need human comparison with a MIDI tempo map. No source can establish rendered sound except the supplied audio samples, and the CLI does not infer timbre or meaning from them.
