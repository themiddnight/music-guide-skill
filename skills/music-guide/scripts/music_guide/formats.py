"""Registry of stored-intent formats. Adding a format is one row here plus its adapter module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .adapters import dawproject, midi, solna
from .adapters import flstudio, studioone
from .adapters.common import MAX_JSON_BYTES, MAX_MIDI_BYTES, MAX_ZIP_BYTES
from .model import Detail, Evidence

Adapter = Callable[[Path, str, Evidence, Detail], None]
SNIFF_BYTES = 64


@dataclass(frozen=True)
class ProjectFormat:
    name: str
    label: str
    extensions: tuple[str, ...]
    kind: Literal["file", "directory"]
    # Largest file that is parsed (when there is an adapter) or hashed (inventory only).
    byte_limit: int
    detect: Callable[[Path], bool]
    adapter: Adapter | None
    guidance: str


def _any(path: Path) -> bool:
    return True


def _signature(*prefixes: bytes, text: bool = False) -> Callable[[Path], bool]:
    def detect(path: Path) -> bool:
        with path.open("rb") as stream:
            head = stream.read(SNIFF_BYTES)
        if text:
            head = head.removeprefix(b"\xef\xbb\xbf").lstrip()
        return head.startswith(prefixes)
    return detect


ZIP = b"PK\x03\x04"
COMPOUND_FILE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
GENERIC_GUIDANCE = "Export a mixdown and, if available, MIDI or DAWproject metadata."

FORMATS: tuple[ProjectFormat, ...] = (
    ProjectFormat("dawproject", "DAWproject", (".dawproject",), "file", MAX_ZIP_BYTES, _signature(ZIP),
                  dawproject.dawproject, ""),
    ProjectFormat("midi", "Standard MIDI file", (".mid", ".midi"), "file", MAX_MIDI_BYTES, _signature(b"MThd"),
                  midi.midi, ""),
    ProjectFormat("solna", "SOLNA project", (".solna",), "file", MAX_JSON_BYTES, _signature(b"{", text=True),
                  solna.solna, ""),
    ProjectFormat("pro-tools-session", "Pro Tools session", (".ptx", ".ptf", ".pts"), "file", MAX_ZIP_BYTES, _any, None,
                  "Export the session as AAF, or bounce a mixdown and stems; export MIDI tracks as a MIDI file."),
    ProjectFormat("bitwig-project", "Bitwig Studio project", (".bwproject",), "file", MAX_ZIP_BYTES, _any, None,
                  "In Bitwig Studio use File > Export > DAWproject and pass the .dawproject file."),
    ProjectFormat("reason-song", "Reason song", (".reason",), "file", MAX_ZIP_BYTES, _any, None,
                  "Export a mixdown and stems, and export the sequencer as a MIDI file."),
    ProjectFormat("logic-package", "Logic Pro project", (".logicx",), "directory", MAX_ZIP_BYTES, _any, None,
                  "Export a mixdown and stems, and the MIDI regions as a MIDI file."),
    ProjectFormat("garageband-package", "GarageBand project", (".band",), "directory", MAX_ZIP_BYTES, _any, None,
                  "Export the song to disk as WAV or AIFF. GarageBand cannot export MIDI; Logic Pro can open the project to do so."),
    ProjectFormat("cubase-project", "Cubase or Nuendo project", (".cpr", ".npr"), "file", MAX_ZIP_BYTES, _any, None,
                  "Export a DAWproject (Cubase 14 or later) or a MIDI file, plus a mixdown."),
    ProjectFormat("ableton-als", "Ableton Live set", (".als",), "file", MAX_ZIP_BYTES, _signature(b"\x1f\x8b"), None,
                  "Export a mixdown and stems with File > Export Audio/Video, and MIDI clips as MIDI files."),
    ProjectFormat("reaper-rpp", "REAPER project", (".rpp",), "file", MAX_ZIP_BYTES,
                  _signature(b"<REAPER_PROJECT", text=True), None,
                  "Render a mixdown and stems, and export MIDI items as MIDI files."),
    ProjectFormat("fl-studio-flp", "FL Studio project", (".flp",), "file", MAX_ZIP_BYTES, flstudio.detect, flstudio.flstudio,
                  "Export a WAV mixdown and stems, and use File > Export > MIDI file."),
    ProjectFormat("studio-one-song", "Studio One song", (".song",), "file", MAX_ZIP_BYTES, studioone.detect,
                  studioone.studioone,
                  "Export a DAWproject (Studio One 6.5 or later), or a mixdown, stems and MIDI."),
    ProjectFormat("ardour-session", "Ardour session", (".ardour",), "file", MAX_ZIP_BYTES, _signature(b"<", text=True), None,
                  "Export a mixdown and stems with Session > Export, and MIDI regions as MIDI files."),
    ProjectFormat("lmms-project", "LMMS project", (".mmp", ".mmpz"), "file", MAX_ZIP_BYTES, _any, None,
                  "Export a WAV mixdown with File > Export and the tracks with File > Export MIDI."),
    ProjectFormat("aaf", "AAF interchange file", (".aaf",), "file", MAX_ZIP_BYTES,
                  _signature(COMPOUND_FILE, b"<", text=True), None,
                  "Export a mixdown and stems, or a DAWproject or MIDI file if the source application supports it."),
    ProjectFormat("omf", "OMF interchange file", (".omf",), "file", MAX_ZIP_BYTES, _any, None,
                  "Export a mixdown and stems, or a DAWproject or MIDI file if the source application supports it."),
    ProjectFormat("musicxml", "MusicXML score", (".musicxml", ".mxl"), "file", MAX_ZIP_BYTES,
                  _signature(ZIP, b"<", text=True), None,
                  "Export the score as a MIDI file, and as audio for the rendered sound."),
    ProjectFormat("musescore-mscz", "MuseScore score", (".mscz",), "file", MAX_ZIP_BYTES, _signature(ZIP), None,
                  "Export the score as a MIDI file, and as audio for the rendered sound."),
)
BY_NAME = {item.name: item for item in FORMATS}
assert len(BY_NAME) == len(FORMATS), "format names must be unique"


def resolve(path: Path) -> ProjectFormat | None:
    """Pick the format by extension and content signature; None means the extension is unknown."""
    candidates = [item for item in FORMATS if path.suffix.lower() in item.extensions]
    if not candidates:
        return None
    for item in candidates:
        if item.kind == "directory" and path.is_dir():
            return item
        if item.kind == "file" and path.is_file() and item.detect(path):
            return item
    labels = " or ".join(f"{item.label} {item.kind}" for item in candidates)
    raise ValueError(f"source does not match the {labels} it is named as")
