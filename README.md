# Music Guide

A music assistant for Claude Code and Codex that helps you understand a piece, develop musical ideas, and decide what to try next. Use it for composition, theory, arrangement, production, mixing, mastering, mood, genre, and artwork direction.

Music Guide works across DAWs. A rendered mixdown is the starting point for claims about how a specific track sounds. MIDI, stems, scores and project files from Ableton Live, Logic Pro, Cubase, FL Studio, REAPER, Studio One and other applications add stored intent such as tempo, key, tracks and notes. Project files alone are enough to discuss stored intent. General music questions and new creative briefs need no file.

The plugin combines a conversational skill with local Python tools. The tools extract evidence; your host assistant explains it and proposes changes. Claims retain their source and uncertainty. Missing information stays unknown.

## Install through a marketplace

Requirements:

- Claude Code or Codex with plugin support.
- Python 3.10 or newer for file analysis.
- Network access for installation and Python dependencies. The analysis tools run locally.

The commands below use this repository as a Git-backed marketplace. Its catalog name is `music-guide-local`, including when installed from GitHub. The install identifier is `music-guide@music-guide-local`.

### Claude Code

In a terminal:

```sh
claude plugin marketplace add themiddnight/music-guide-skill
claude plugin install music-guide@music-guide-local
```

Or run these commands inside Claude Code:

```text
/plugin marketplace add themiddnight/music-guide-skill
/plugin install music-guide@music-guide-local
```

Restart Claude Code, then invoke:

```text
/music-guide:music-guide
```

Follow it with your question or file paths. You can also describe the task in natural language and ask the assistant to use Music Guide.

### Codex

In a terminal:

```sh
codex plugin marketplace add themiddnight/music-guide-skill
codex plugin add music-guide@music-guide-local
```

Restart Codex and ask:

```text
Use music-guide to analyze this mixdown: /path/to/song.wav.
Separate measured facts from interpretations and suggest three experiments.
```

### Set up the analysis runtime

Installing the plugin registers the skill. It does not install Python packages. File analysis needs Python 3.10 or newer and the packages in the skill's `requirements.txt`.

If [uv](https://docs.astral.sh/uv/) is installed, the assistant can run the tool with `uv run --no-project --with-requirements`, which builds a cached environment on first use. Otherwise, on the first file-analysis task, ask the assistant:

```text
Set up Music Guide's Python requirements in a virtual environment in my
working folder. Use that environment's interpreter for the analysis.
```

The requirements file is included in the installed skill, so no repository clone is needed. Keep the environment outside the plugin cache so plugin updates do not replace it. If a package is missing, the tool exits with code 3 and names the requirements file. General theory questions do not require the analysis tools.

### Install from a local checkout

To test a checkout or an unpacked release, replace the GitHub source in either marketplace command with its absolute directory:

```sh
codex plugin marketplace add /absolute/path/to/music-guide
codex plugin add music-guide@music-guide-local
```

For Claude Code:

```sh
claude plugin marketplace add /absolute/path/to/music-guide
claude plugin install music-guide@music-guide-local
```

Choose either a Git-backed marketplace or a local checkout for this catalog name. If you already registered `music-guide-local` from another source, keep that source or explicitly remove its marketplace registration before changing it.

## Use Music Guide

Start with the result you want. Provide the relevant file paths and any creative constraints, such as the intended audience, reference sound, or part you want to improve.

| Intent | Example request |
| --- | --- |
| Analyze | "Analyze this mixdown with its MIDI and Ableton set. Explain which findings come from each source." |
| Explain | "Explain how Hirajoshi differs from natural minor, with a short melodic example. I do not have a recording." |
| Critique | "Critique this track against my goal of a restrained verse and a stronger chorus. Mark anything that needs listening." |
| Recommend | "Suggest three mixing experiments for this mixdown. Explain how I can check whether each helps." |
| Compare | "Compare these two WAV mixdowns. Account for their level differences and identify what still needs a listening test." |
| Create | "Propose a contrasting bridge, a bass variation, and three artwork directions. Treat them as new ideas." |

For claims about the sound of a specific track, supply a mixdown as WAV, AIFF, FLAC, MP3 or another format libsndfile decodes. Add a project file, MIDI or stems when you have them. You do not need to export every file type, and a project file without audio still supports questions about tempo, key, structure and parts.

An analysis can include source-backed observations, unresolved conflicts, missing evidence, and prioritized experiments. When the host cannot hear audio, Music Guide says so and avoids claims about audible instruments, mood, or genre. Stored project settings describe intent and may differ from the render.

## Apply it to your workflow

### Develop a composition

Provide a mixdown and, optionally, the project, MIDI, SOLNA or score file. Ask about motif development, register, harmonic rhythm, voice leading, or a contrasting section. Request a short musical proposal you can audition, together with the evidence behind the suggestion. Per-track note summaries show pitch range, pitch-class use and density. Stored notes and chords help explain the composition without proving exactly what is audible.

### Review an arrangement or production decision

State the intended effect, such as a gradual build or a quieter opening. Ask for a few changes to density, entrances, transitions, sound choice, or automation. Treat suggestions requiring listening as hypotheses. Audition one change at a time and report what you hear for a follow-up discussion.

### Compare revisions

Supply a mixdown for each version and describe what changed. Ask Music Guide to analyze each separately, compare compatible measurements, and design a listening test level-matched by integrated loudness. Different loudness values alone do not establish which version sounds better.

### Prepare a mix or mastering review

The report measures integrated, momentary and short-term loudness (LUFS), loudness range, true peak (dBTP), crest factor, band energy, spectral centroid, and stereo correlation. Use them as a first check, then ask for a listening plan for masking and problem moments, which whole-file averages cannot locate. Confirm final delivery numbers with a dedicated meter.

### Develop a visual direction

Provide a creative brief, or ask for artwork ideas informed by an existing track's available evidence. Request palette, imagery, composition, or typography proposals. Keep musical interpretation separate from measured facts. Music Guide produces directions you can take to a designer or image-generation tool; its CLI does not generate artwork.

### Reuse structured evidence

The CLI writes JSON that you can keep with a session, review alongside revisions, or feed into your own reporting workflow. Cite the report name together with an observation ID when comparing reports, because each report starts its own ID sequence. Reuse the same evidence to explore another question or artwork direction without re-reading unchanged source files.

## Run the tools directly

From a repository checkout, create an environment and install the requirements:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r skills/music-guide/requirements.txt
```

Or skip the environment and use uv:

```sh
uv run --no-project --with-requirements skills/music-guide/requirements.txt \
  python skills/music-guide/scripts/analyze.py --mixdown /path/to/song.wav
```

Analyze a mixdown:

```sh
.venv/bin/python skills/music-guide/scripts/analyze.py \
  --mixdown /path/to/song.wav > evidence.json
```

Add whichever optional inputs you have:

```sh
.venv/bin/python skills/music-guide/scripts/analyze.py \
  --mixdown /path/to/song.flac \
  --stem /path/to/drums.wav \
  --midi /path/to/parts.mid \
  --project /path/to/session.als \
  --declarations /path/to/declarations.json > evidence.json
```

At least one input is required; the mixdown is optional. Repeat an input flag for multiple files. `--project` detects the format from the file's extension and content, and accepts directory packages such as `.logicx`. `--midi`, `--solna` and `--dawproject` name those formats explicitly. Stems are measured independently; the tools do not align them or reconstruct the mix. A missing or corrupt optional source produces a diagnostic while the other results remain available. An unreadable mixdown exits with code 2.

The default `--detail summary` reports one note summary per track. `--detail full` also lists every note, which can make the report several megabytes for a dense project.

A declarations file records information you supply rather than facts measured by the tool:

```json
[
  {"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}
]
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Windows has not been tested here.

## Understand the evidence and limits

| Source | Current support |
| --- | --- |
| Audio mixdown or stems | WAV, RF64, W64, AIFF, FLAC, CAF, OGG Vorbis and Opus, MP3. Peak, RMS, crest factor, BS.1770-4 loudness, loudness range, true peak, loudness timeline, band energy, spectral centroid, stereo correlation, pitch-class profile, and inferred tempo and key candidates. Lossy files are flagged. |
| MIDI | SMF 0/1 tempo and meter maps, track names, per-track note summaries, and every note with `--detail full`. Type 2 and SMPTE timing are unsupported. |
| Ableton Live, REAPER, LMMS, MusicXML, MuseScore, DAWproject | Parsed: tempo, meter, tracks and note summaries, plus tempo maps, plugins, clip counts, markers and media file names where the format stores them. |
| FL Studio, Studio One, Ardour, AAF | Parsed with documented gaps: for example FL Studio pattern notes are not placed on the timeline, Studio One instrument notes and Ardour MIDI are not decoded, and AAF stores no tempo. |
| Logic Pro, GarageBand | Package metadata: tempo, meter, key, sample rate, track count and media file names. Regions, notes and tempo changes in the binary `ProjectData` are not read. |
| Cubase, Nuendo | Application version and project metadata. Track, plugin and media names are found by byte patterns at low confidence, verified on one Cubase 11 project. Export DAWproject (Cubase 14+) for complete data. |
| SOLNA | Version 12 and earlier: project tempo, meter ID, loop scales and chords. |
| Pro Tools, Bitwig, Reason, OMF | Inventory only, with format-specific export guidance. |

Most project formats have no public specification. Except for Logic Pro and Cubase, the project parsers were checked only against synthetic files built from public documentation and open-source readers, so real projects may expose gaps. Each observation's confidence reason states what was approximated. The [input contract](skills/music-guide/references/formats.md) lists the details and sources for every format.

Evidence distinguishes `measured`, `extracted`, `inferred`, and `user_declared` claims. Observations include source hashes, method, locator, units, scope, and ordinal confidence with a reason. Confidence is not a calibrated probability. Conflicts remain visible rather than being silently resolved by source priority.

Automatic conflict detection covers static tempo and meter claims with the same scope; tempo values within 0.01 BPM count as equal. A MIDI file with a single tempo event at the start yields a static tempo claim, while a changing tempo stays a map. Inferred audio estimates never raise automatic conflicts.

Silence produces `null` loudness values with a diagnostic. Samples at full scale indicate possible clipping, not certain audible distortion. Tempo and key estimates are candidates: tempo may be half or double, and the key estimate knows only major and minor, so modal or pentatonic music can be mislabelled. The tools do not recognise instruments, mood, or genre, render music, or generate artwork. Project files are never used to open the media they reference.

The local tools do not upload source files. Your host assistant still processes the conversation and any evidence or files you give it under that host's settings.

## Install a standalone skill instead

If you want the skill without marketplace registration, run the bundled installer with an exact destination:

```sh
python3 install.py --dry-run --dest /path/to/skills/music-guide
python3 install.py --dest /path/to/skills/music-guide
```

The installer copies only `skills/music-guide`. It does not install Python packages or change host settings. Repeating an identical install is safe, including after use. Conflicting content is refused without deletion. The copy runs without this checkout.

Codex's legacy user destination is `~/.codex/skills/music-guide`, or `<CODEX_HOME>/skills/music-guide` when configured. Claude Code's user skill directory is `~/.claude/skills/music-guide`. Restart the host after installation and set up a separate Python environment as described above.

## Update or uninstall

For a Git-backed Codex marketplace, refresh the catalog with:

```sh
codex plugin marketplace upgrade music-guide-local
```

Use the host's plugin management to install an available plugin update. Refreshing a marketplace alone does not establish that an installed copy was updated. For a local marketplace, update the checkout first.

For Claude Code:

```sh
claude plugin marketplace update music-guide-local
claude plugin update music-guide@music-guide-local
```

To uninstall the plugin:

```sh
# Codex
codex plugin remove music-guide@music-guide-local

# Claude Code
claude plugin uninstall music-guide@music-guide-local
```

Keep analysis reports and virtual environments separately if you want them to survive plugin changes.

## Verification and further development

Validation includes synthetic fixtures for every format, standard loudness test signals cross-checked against ffmpeg's EBU R128 meter, and three user-provided reference sets (WAV with MIDI and SOLNA, MP3 with a Logic Pro project, WAV with a Cubase project) on macOS. Committed fixtures are synthetic. Other operating systems, real projects for most DAW formats, and end-to-end model response quality remain unverified. See the [verification record](docs/verification.md) for evidence and remaining limits.

Run the behavior tests with the environment that has the requirements installed:

```sh
PYTHONWARNINGS=error .venv/bin/python -m unittest discover -s tests -v
```

Reference material:

- [Input contract and adapter extension guide](skills/music-guide/references/formats.md).
- [Canonical evidence schema](skills/music-guide/schemas/evidence.schema.json).
- [User declarations schema](skills/music-guide/schemas/declarations.schema.json).
- [OpenAI plugin packaging guide](https://developers.openai.com/plugins/build/plugins).
- [DAWproject specification](https://github.com/bitwig/dawproject).

The roadmap prioritizes checks against real projects for each DAW format, Logic Pro regions and notes, Cubase tempo tracks, and placing FL Studio patterns on the timeline. Add or extend an adapter only with a tested parser and an explicit support contract.
