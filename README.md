# Music Guide

Music Guide is a local skill for Claude Code and Codex. Its Python command reads a WAV mixdown plus optional MIDI, SOLNA, or DAWproject metadata and writes JSON evidence. The conversational skill uses that evidence to answer music questions and develop ideas. It does not listen semantically, render music, or generate artwork.

## Requirements

Python 3.10 or newer. Install the payload's dependencies in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r skills/music-guide/requirements.txt
.venv/bin/python skills/music-guide/scripts/analyze.py --mixdown song.wav > evidence.json
```

WAV PCM and IEEE float are guaranteed. Decoding uses libsndfile through `soundfile`; supported integer and float WAV subtypes are checked. No other audio codec is promised. Sample peak and RMS use decoded sample values. Silence produces `null` dBFS values and a diagnostic. Samples at or above full scale are only a potential clipping signal.

Optional inputs, including independently measured WAV stems:

```sh
.venv/bin/python skills/music-guide/scripts/analyze.py --mixdown song.wav \
  --stem drums.wav --midi parts.mid --solna sketch.solna --dawproject session.dawproject \
  --declarations declarations.json > evidence.json
```

Stems are measured under separate source scopes; no alignment or mix reconstruction is assumed. `declarations.json` is an optional array such as `[ {"property":"tempo","value":120,"unit":"BPM","scope":"project"} ]`. User declarations remain marked `user_declared` and can conflict with extracted project metadata. Repeat an optional flag for more files. `--project native.logicx` records a native project as unsupported and recommends an export. A missing or corrupt optional file yields a warning diagnostic; a bad required mixdown exits nonzero.

The [evidence schema](skills/music-guide/schemas/evidence.schema.json) and [declarations schema](skills/music-guide/schemas/declarations.schema.json) documents the stable evidence shape. Source hashes are SHA-256; observations identify source, method, locator, unit, scope, time domain, provenance, and ordinal confidence with a reason. Conflicts retain all compatible static claims. MIDI tempo events retain tick time, including a single event; they are not silently promoted to static facts or collapsed into conflicts. Compare declarations with MIDI event maps manually.

## Installation

The repository root contains Codex, Claude Code, and portable plugin metadata. For local Codex plugin registration, use `codex plugin marketplace add /absolute/path/to/this/repository` then `codex plugin add music-guide@music-guide-local`; exact host discovery depends on the installed Codex version. For Claude Code, add this directory as a local marketplace with `claude plugin marketplace add /absolute/path/to/this/repository`, then install `music-guide@music-guide-local`. Neither route requires a published repository URL.

For a standalone copy of the skill, choose an exact destination directory supported by your host:

```sh
python3 install.py --dry-run --dest /path/to/skills/music-guide
python3 install.py --dest /path/to/skills/music-guide
python3 -m pip install -r /path/to/skills/music-guide/requirements.txt
python3 /path/to/skills/music-guide/scripts/analyze.py --mixdown song.wav
```

The installer copies only `skills/music-guide`. Repeating it with identical content is safe; different existing content is refused without deletion. It does not alter host configuration or install Python dependencies. A copied skill runs without this repository.

For Codex's legacy user skill directory, the destination is `~/.codex/skills/music-guide`, or `<CODEX_HOME>/skills/music-guide` when configured. Claude Code's user skill directory is `~/.claude/skills/music-guide`. Restart the host after installation. Keep dependencies in a virtual environment and use that environment's Python when running the CLI. On Windows, its interpreter is `.venv\Scripts\python.exe`; Windows has not been tested here.

## Use the conversational skill

Ask Codex to use `music-guide`, or invoke `/music-guide:music-guide` in Claude Code after plugin installation. For example:

- "Analyze this mixdown and explain which findings come from audio versus my MIDI."
- "Explain why a D minor progression can feel unresolved. I do not have a recording."
- "Critique this arrangement against my goal of a quieter first verse. Mark anything that needs listening."
- "Recommend three mixing experiments from this mixdown, with a way to check each one."
- "Compare these two mixdowns and account for their level difference."
- "Create a contrasting bridge and three artwork directions. Treat them as proposals."

The CLI extracts evidence. The host model supplies the musical explanation and proposals. If the host cannot hear audio, the skill states that limit and avoids claims about audible instruments, mood, or genre.

The [format contract](skills/music-guide/references/formats.md) lists accepted inputs and parser limits. Bounds are defined in `skills/music-guide/scripts/music_guide/adapters/common.py` and role preflight limits in `skills/music-guide/scripts/music_guide/cli.py`.

## Supported metadata and limits

MIDI SMF 0/1 records stored tempo and meter events, track names, paired notes, ticks and beats. Type 2 and SMPTE timing are explicitly unsupported. SOLNA extraction is limited to inspected version 12 and earlier with the same validated shape: project BPM/meter ID, loop scales, and chord fields. It does not decode patterns or emulate rendering. DAWproject reads bounded `project.xml` metadata for version, transport settings, and track count/names; it does not reconstruct nested event timing or media. Proprietary Logic, Cubase, FL Studio, and Ableton sessions need WAV and optional standard exports.

Current validation uses generated fixtures on macOS. Real source projects, other operating systems, and host discovery beyond local CLI checks need separate verification. Future work includes real-world fixtures with permission, broader codecs, dedicated LUFS/true-peak measurement, uncertain key/tempo estimation, and deeper DAWproject event timing.

See the [input contract and adapter extension guide](skills/music-guide/references/formats.md) for parser limits and how to add support. Packaging follows the [OpenAI plugin packaging guide](https://developers.openai.com/plugins/build/plugins), with a portable root manifest and Codex compatibility metadata. The [DAWproject specification](https://github.com/bitwig/dawproject) defines the source container and XML fields.

## Development checks

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m json.tool skills/music-guide/schemas/evidence.schema.json >/dev/null
```
