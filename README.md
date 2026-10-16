# Music Guide

A music assistant for Claude Code and Codex that helps you understand a piece, develop musical ideas, and decide what to try next. Use it for composition, theory, arrangement, production, mixing, mastering, mood, genre, and artwork direction.

Music Guide works across DAWs. A mixdown is the starting point for analysis of a specific track. MIDI, stems, SOLNA, and DAWproject files can add context when available. General music questions and new creative briefs need no audio file.

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

Installing the plugin registers the skill. It does not install Python packages. On the first file-analysis task, ask the assistant:

```text
Set up Music Guide's Python requirements in a virtual environment in my
working folder. Use that environment's interpreter for the analysis.
```

The requirements file is included in the installed skill. The assistant can locate it and set up the environment without a separate repository clone. Keep the environment outside the plugin cache so plugin updates do not replace it. General theory questions do not require the analysis tools.

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
| Analyze | "Analyze this WAV with its MIDI and SOLNA files. Explain which findings come from each source." |
| Explain | "Explain how Hirajoshi differs from natural minor, with a short melodic example. I do not have a recording." |
| Critique | "Critique this track against my goal of a restrained verse and a stronger chorus. Mark anything that needs listening." |
| Recommend | "Suggest three mixing experiments for this mixdown. Explain how I can check whether each helps." |
| Compare | "Compare these two WAV mixdowns. Account for their level differences and identify what still needs a listening test." |
| Create | "Propose a contrasting bridge, a bass variation, and three artwork directions. Treat them as new ideas." |

For a specific track, supply a readable WAV mixdown. Add optional sources only when you have them. You do not need to export every file type.

An analysis can include source-backed observations, unresolved conflicts, missing evidence, and prioritized experiments. When the host cannot hear audio, Music Guide says so and avoids claims about audible instruments, mood, or genre. Stored project settings describe intent and may differ from the render.

## Apply it to your workflow

### Develop a composition

Provide a mixdown and optional MIDI or SOLNA file. Ask about motif development, register, harmonic rhythm, voice leading, or a contrasting section. Request a short musical proposal you can audition, together with the evidence behind the suggestion. Stored notes and chords help explain the composition without proving exactly what is audible.

### Review an arrangement or production decision

State the intended effect, such as a gradual build or a quieter opening. Ask for a few changes to density, entrances, transitions, sound choice, or automation. Treat suggestions requiring listening as hypotheses. Audition one change at a time and report what you hear for a follow-up discussion.

### Compare revisions

Supply a WAV for each version and describe what changed. Ask Music Guide to analyze each separately, compare compatible measurements, and design a level-matched listening test. Different RMS values alone do not establish which version sounds better.

### Prepare a mix or mastering review

Use the measured levels and full-scale sample count as a first check. Ask for a metering and listening plan for tonal balance, masking, stereo image, and dynamics. Use a dedicated LUFS and true-peak meter for release decisions; the MVP's RMS and sample peak are different measurements.

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

Analyze a mixdown:

```sh
.venv/bin/python skills/music-guide/scripts/analyze.py \
  --mixdown /path/to/song.wav > evidence.json
```

Add whichever optional inputs you have:

```sh
.venv/bin/python skills/music-guide/scripts/analyze.py \
  --mixdown /path/to/song.wav \
  --stem /path/to/drums.wav \
  --midi /path/to/parts.mid \
  --solna /path/to/sketch.solna \
  --dawproject /path/to/session.dawproject \
  --declarations /path/to/declarations.json > evidence.json
```

Repeat an optional flag for multiple files. Stems are measured independently; the tools do not align them or reconstruct the mix. A missing or corrupt optional source produces a diagnostic while the valid mixdown results remain available. An unreadable required mixdown exits nonzero.

Use `--project /path/to/session.logicx` or another native project path to inventory an unsupported source and receive export guidance. This does not parse the native project.

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
| WAV mixdown or stems | PCM and IEEE float, including WAVE Extensible. Duration, channels, sample rate, sample peak, RMS, and full-scale sample count. |
| MIDI | SMF 0/1 track names, tempo and meter events, and paired notes in ticks and beats. Type 2 and SMPTE timing are unsupported. |
| SOLNA | Inspected version 12 and earlier files with the same validated subset of project tempo, meter ID, loop scales, and chord fields. No pattern decoding or rendering. |
| DAWproject | Bounded metadata from `project.xml`, including transport tempo and meter, track count, and names. No nested arrangement reconstruction or media loading. |
| Native Logic, Cubase, FL Studio, or Ableton project | Unsupported inventory only. Export a WAV mixdown and optional standard files from the DAW. |
| MP3 and other audio codecs | No direct CLI support. Export WAV or convert a copy locally, keeping the conversion's provenance and warnings. |

Evidence distinguishes `measured`, `extracted`, `inferred`, and `user_declared` claims. Observations include source hashes, method, locator, units, scope, and ordinal confidence with a reason. Confidence is not a calibrated probability. Conflicts remain visible rather than being silently resolved by source priority.

Automatic conflict detection covers compatible static tempo and meter claims. MIDI events retain tick locations, including a single event. Compare declarations with a MIDI tempo map explicitly rather than treating the map as one static fact.

Silence produces `null` dBFS values with a diagnostic. Samples at full scale indicate possible clipping, not certain audible distortion. The tools do not estimate key, tempo, mood, or genre from audio, measure LUFS or true peak, render music, or generate finished artwork.

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

Validation includes synthetic fixtures and three user-provided reference sets on macOS, plus native Codex and Claude Code installation in isolated profiles. Committed fixtures are synthetic. Other operating systems, broader project coverage, and end-to-end model response quality remain unverified. See the [verification record](docs/verification.md) for evidence and remaining limits.

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

The roadmap prioritizes broader verified codecs, dedicated LUFS and true-peak measurement, uncertain key and tempo estimation, and deeper DAWproject timing. Add a native-format adapter only with a tested parser and an explicit support contract.
