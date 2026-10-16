from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element

from ..model import Detail, Evidence
from ..project import ProjectFacts, TrackKind
from .common import MAX_XML_BYTES, _child, _finite_number, _local, _parse_xml, _read_zip_member

CHANNEL_ROLES: dict[str, TrackKind] = {"master": "master", "effect": "return", "submix": "group"}
CONTENT_TYPES: list[tuple[str, TrackKind]] = [("tracks", "folder"), ("notes", "instrument"), ("audio", "audio")]


def _track_kind(track: Element) -> TrackKind:
    channel = _child(track, "Channel")
    role = channel.attrib.get("role", "") if channel is not None else ""
    if role in CHANNEL_ROLES:
        return CHANNEL_ROLES[role]
    content = set(track.attrib.get("contentType", "").split())
    return next((kind for token, kind in CONTENT_TYPES if token in content), "unknown")


def dawproject(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _parse_xml(_read_zip_member(path, "project.xml", MAX_XML_BYTES))
    if _local(root.tag) != "Project":
        raise ValueError("DAWproject project.xml root must be Project")
    facts = ProjectFacts(evidence, source_id, "dawproject-xml", detail)
    if "version" in root.attrib:
        evidence.add(source_id, "dawproject.version", root.attrib["version"], "version", "project",
                     "static", "Project/@version", "dawproject-xml", "extracted", "high",
                     "Stored DAWproject version.")
    application = _child(root, "Application")
    if application is not None and application.attrib.get("name", "").strip():
        name = f"{application.attrib['name']} {application.attrib.get('version', '')}".strip()
        facts.application(name, "Application/@name")
    transport = _child(root, "Transport")
    if transport is not None:
        tempo = _child(transport, "Tempo")
        if tempo is not None and "value" in tempo.attrib:
            if tempo.attrib.get("unit") != "bpm":
                raise ValueError("DAWproject Transport/Tempo unit must be bpm")
            facts.tempo(_finite_number(float(tempo.attrib["value"]), "Tempo value", positive=True),
                        "Transport/Tempo/@value", "Stored transport tempo; automation is not reconstructed.")
        meter = _child(transport, "TimeSignature")
        if meter is not None and {"numerator", "denominator"} <= meter.attrib.keys():
            facts.meter(int(meter.attrib["numerator"]), int(meter.attrib["denominator"]),
                        "Transport/TimeSignature", "Stored transport meter; automation is not reconstructed.")
    structure = _child(root, "Structure")
    if structure is None:
        return
    tracks = [item for item in structure.iter() if _local(item.tag) == "Track"]
    facts.track_count(len(tracks), "Structure/Track")
    for index, track in enumerate(tracks):
        facts.track(index, track.attrib.get("name"), _track_kind(track), f"Structure/Track[{index}]")
