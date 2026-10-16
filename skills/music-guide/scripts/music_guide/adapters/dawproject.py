"""Read only bounded DAWproject transport and structure metadata."""

from __future__ import annotations

import zipfile
from pathlib import Path

from defusedxml import ElementTree as ET

from ..model import Evidence
from .common import MAX_XML_BYTES, MAX_ZIP_BYTES, MAX_ZIP_ENTRIES, _finite_number


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in parent if _local(item.tag) == name), None)


def dawproject(path: Path, source_id: str, evidence: Evidence) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES or sum(item.file_size for item in entries) > MAX_ZIP_BYTES:
            raise ValueError("DAWproject archive exceeds metadata safety limits")
        files = [item for item in entries if item.filename == "project.xml"]
        if len(files) != 1 or files[0].file_size > MAX_XML_BYTES:
            raise ValueError("DAWproject must contain one bounded project.xml")
        with archive.open(files[0]) as stream:
            raw = stream.read(MAX_XML_BYTES + 1)
        if len(raw) > MAX_XML_BYTES:
            raise ValueError("DAWproject project.xml exceeds size limit")
    root = ET.fromstring(raw, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    if _local(root.tag) != "Project":
        raise ValueError("DAWproject project.xml root must be Project")
    if "version" in root.attrib:
        evidence.add(source_id, "dawproject.version", root.attrib["version"], "version", "project",
                     "static", "Project/@version", "dawproject-xml", "extracted", "high",
                     "Stored DAWproject version.")
    transport = _child(root, "Transport")
    if transport is not None:
        tempo = _child(transport, "Tempo")
        if tempo is not None and "value" in tempo.attrib:
            value = _finite_number(float(tempo.attrib["value"]), "Tempo value", positive=True)
            evidence.add(source_id, "tempo", value, "BPM", "project", "static",
                         "Transport/Tempo/@value", "dawproject-xml", "extracted", "medium",
                         "Stored transport tempo; automation is not reconstructed.")
        meter = _child(transport, "TimeSignature")
        if meter is not None and {"numerator", "denominator"} <= meter.attrib.keys():
            numerator = int(meter.attrib["numerator"])
            denominator = int(meter.attrib["denominator"])
            if numerator <= 0 or denominator <= 0:
                raise ValueError("invalid TimeSignature")
            evidence.add(source_id, "meter", f"{numerator}/{denominator}", "signature", "project",
                         "static", "Transport/TimeSignature", "dawproject-xml", "extracted", "medium",
                         "Stored transport meter; automation is not reconstructed.")
    structure = _child(root, "Structure")
    tracks = [item for item in structure.iter() if _local(item.tag) == "Track"] if structure is not None else []
    evidence.add(source_id, "dawproject.track_count", len(tracks), "count", "project", "static",
                 "Structure/Track", "dawproject-xml", "extracted", "high", "Count of stored Track elements.")
    for index, track in enumerate(tracks):
        if "name" in track.attrib:
            evidence.add(source_id, "dawproject.track_name", track.attrib["name"], "text",
                         f"project_track:{index}", "static", f"Structure/Track[{index}]/@name",
                         "dawproject-xml", "extracted", "high", "Stored track name.")
