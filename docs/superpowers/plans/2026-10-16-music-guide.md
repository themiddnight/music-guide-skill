# Music Guide implementation plan

**Execution:** pstack:poteto-mode, as required by the approved project instructions.
**Goal:** Deliver one installable local music assistance skill with a working evidence CLI.
**Architecture:** One self-contained Python skill payload with thin host manifests and installer. Adapter outputs share a versioned evidence document.
**Tech stack:** Python 3.10+, small declared dependencies, unittest or pytest, JSON Schema validation.
**Spec:** ../specs/2026-10-16-music-guide-design.md

## Global constraints

Keep content in English, no personal paths, no network during analysis, no host configuration mutations during verification. Require mixdown for specific-track analysis. Never invent audio semantics from metadata. Preserve optional-input diagnostics.

## Review focus

Malformed inputs must not become confident facts. Silence must produce valid JSON. Tempo changes must not be treated as conflicts with themselves. Installed skill execution must not depend on checkout paths. Installer collisions must preserve files.

## Tasks

- [ ] Build typed evidence records and canonical JSON schemas. Tests assert source links, provenance, confidence, and conflicts.
- [ ] Implement audio, MIDI, SOLNA, and bounded DAWproject metadata adapters. Tests use generated fixtures and invalid inputs.
- [ ] Implement CLI orchestration, optional-source diagnostics, declarations, and conservative conflict detection. Drive CLI from an unrelated directory.
- [ ] Write skill routing and references for every requested intent and domain, including missing-audio and conflict behavior.
- [ ] Add host manifests, safe installer, requirements/config, and README installation, usage, limits, and roadmap.
- [ ] Run test suite, schema/package validation, and copied-skill CLI checks. Independently review, fix accepted findings, rerun affected checks, and commit locally.
