from __future__ import annotations

import json
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "music-guide"
SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())


class ProjectCliCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SKILL / "scripts" / "analyze.py"), *map(str, args)],
                              cwd=self.dir, text=True, capture_output=True)

    def evidence(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        return data

    def project(self, path, *extra):
        return self.evidence("--project", path, *extra)

    def failed(self, path):
        data = self.project(path)
        source = data["sources"][0]
        self.assertEqual(source["status"], "error")
        self.assertEqual(data["observations"], [])
        message = next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error")
        return message

    @staticmethod
    def values(data, property, scope=None):
        return [item["value"] for item in data["observations"]
                if item["property"] == property and (scope is None or item["scope"] == scope)]


class LogicPackageTest(ProjectCliCase):
    def make_package(self, name="song.logicx", alternatives=((0, "Main"),), meta=None, binary=True, **info):
        package = self.dir / name
        (package / "Resources").mkdir(parents=True)
        fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
        names = {str(number): label for number, label in alternatives}
        (package / "Resources" / "ProjectInformation.plist").write_bytes(plistlib.dumps(
            {"LastSavedFrom": "Logic Pro 12.4 (6707)", "VariantNames": names, **info}, fmt=fmt))
        for number, _ in alternatives:
            folder = package / "Alternatives" / f"{number:03d}"
            folder.mkdir(parents=True)
            (folder / "ProjectData").write_bytes(b"\x00binary that must never be parsed")
            (folder / "MetaData.plist").write_bytes(plistlib.dumps(self.meta(number, meta), fmt=fmt))
        return package

    @staticmethod
    def meta(number, overrides=None):
        data = {"BeatsPerMinute": 92.0 + number, "SongKey": "D", "SongGenderKey": "minor",
                "SongSignatureNumerator": 4, "SongSignatureDenominator": 4, "SampleRate": 48000,
                "NumberOfTracks": 34, "isTimeCodeBased": False, "HasARAPlugins": False,
                "AudioFiles": ["Audio Files/vox.wav", "${CONTENT_iOS:Logic Pro Library.bundle}/Apple Loops/lead.caf",
                               "Audio Files/vox.wav"],
                "UnusedAudioFiles": ["Audio Files/old take.aif"],
                "SamplerInstrumentsFiles": ["${CONTENT_iOS:Logic Pro Library.bundle}/Samples/Guitar/50D.aif"]}
        data.update(overrides or {})
        return data

    def test_each_alternative_is_its_own_scope_with_stored_intent(self):
        package = self.make_package(alternatives=((0, "Main"), (1, "Alt mix")))
        media = package / "Media" / "Audio Files"
        media.mkdir(parents=True)
        (media / "vox.wav").write_bytes(b"RIFF")
        (media / "vox.wav").chmod(0)
        self.addCleanup(lambda: (media / "vox.wav").chmod(stat.S_IRUSR | stat.S_IWUSR))
        (package / "Alternatives" / "000" / "ProjectData").chmod(0)
        self.addCleanup(lambda: (package / "Alternatives" / "000" / "ProjectData").chmod(stat.S_IRUSR | stat.S_IWUSR))
        data = self.project(package)
        source = data["sources"][0]
        self.assertEqual((source["adapter"], source["status"], source["sha256"]), ("logic-package", "ok", None))
        self.assertEqual(self.values(data, "project.application"), ["Logic Pro 12.4 (6707)"])
        for number, bpm in (("000", 92.0), ("001", 93.0)):
            scope = f"alternative:{number}"
            self.assertEqual(self.values(data, "tempo", scope), [bpm])
            self.assertEqual(self.values(data, "meter", scope), ["4/4"])
            self.assertEqual(self.values(data, "project.key", scope), ["D minor"])
            self.assertEqual(self.values(data, "project.sample_rate", scope), [48000])
            self.assertEqual(self.values(data, "project.track_count", scope), [34])
        self.assertEqual(self.values(data, "logic.alternative_name"), ["Main", "Alt mix"])
        tempo = next(item for item in data["observations"] if item["property"] == "tempo")
        self.assertEqual((tempo["provenance"], tempo["confidence"]), ("extracted", "medium"))
        self.assertIn("tempo map in ProjectData is not read", tempo["confidenceReason"])
        self.assertEqual(self.values(data, "project.media_reference", "alternative:000"), [
            {"basename": "lead.caf", "count": 1}, {"basename": "vox.wav", "count": 2},
            {"basename": "old take.aif", "count": 1}, {"basename": "50D.aif", "count": 1}])
        locators = {item["locator"].rsplit(":", 1)[1] for item in data["observations"]
                    if item["property"] == "project.media_reference"}
        self.assertEqual(locators, {"AudioFiles", "UnusedAudioFiles", "SamplerInstrumentsFiles"})
        roles = {item["list"]: (item["role"], item["count"]) for item in self.values(data, "logic.media_list", "alternative:000")}
        self.assertEqual(roles["UnusedAudioFiles"], ("unused audio", 1))
        self.assertEqual(self.values(data, "logic.bundled_media"), [{"fileCount": 1, "linksNotFollowed": 0, "truncated": False}] * 1)
        self.assertEqual(self.values(data, "logic.has_ara_plugins", "alternative:000"), [False])
        self.assertTrue(any(item["code"] == "project_data_not_read" for item in data["diagnostics"]))
        self.assertFalse(data["conflicts"])
        self.assertEqual(self.values(data, "project.media_reference", "project"), [])

    def test_xml_plists_and_missing_optional_fields_are_not_defaulted(self):
        package = self.make_package(binary=False)
        meta_path = package / "Alternatives" / "000" / "MetaData.plist"
        meta = plistlib.loads(meta_path.read_bytes())
        for key in ("SongGenderKey", "NumberOfTracks", "SampleRate", "SongSignatureDenominator"):
            meta.pop(key, None)
        meta_path.write_bytes(plistlib.dumps(meta, fmt=plistlib.FMT_XML))
        data = self.project(package)
        self.assertEqual(self.values(data, "tempo"), [92.0])
        for absent in ("meter", "project.key", "project.sample_rate", "project.track_count"):
            self.assertEqual(self.values(data, absent), [], absent)
        self.assertEqual(self.values(data, "logic.partial_key"), ["D"])

    def test_tempo_conflicts_with_a_declaration_for_the_same_alternative(self):
        package = self.make_package()
        declarations = self.dir / "declarations.json"
        declarations.write_text(json.dumps([{"property": "tempo", "value": 120, "unit": "BPM", "scope": "alternative:000"}]))
        data = self.evidence("--project", package, "--declarations", declarations)
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])
        self.assertEqual(data["conflicts"][0]["scope"], "alternative:000")
        declarations.write_text(json.dumps([{"property": "tempo", "value": 92, "unit": "BPM", "scope": "alternative:000"}]))
        self.assertFalse(self.evidence("--project", package, "--declarations", declarations)["conflicts"])

    def test_garageband_package_shares_the_layout_and_explains_older_ones(self):
        package = self.make_package("beat.band", LastSavedFrom="GarageBand 10.4.12 (5640)")
        data = self.project(package)
        self.assertEqual(data["sources"][0]["adapter"], "garageband-package")
        self.assertEqual(self.values(data, "project.application"), ["GarageBand 10.4.12 (5640)"])
        self.assertEqual(self.values(data, "tempo", "alternative:000"), [92.0])
        self.assertTrue(self.values(data, "garageband.alternative_name"))
        self.assertFalse(self.values(data, "logic.alternative_name"))
        legacy = self.dir / "old.band"
        legacy.mkdir()
        (legacy / "projectData").write_bytes(b"x")
        message = self.failed(legacy)
        self.assertIn("projectData", message)
        self.assertIn("WAV or AIFF", message)
        empty = self.dir / "empty.logicx"
        empty.mkdir()
        self.assertIn("Alternatives/NNN/MetaData.plist", self.failed(empty))

    def test_malformed_oversize_and_unsafe_plists_are_actionable_errors(self):
        cases = {
            "garbage": b"not a plist at all",
            "truncated binary": plistlib.dumps({"BeatsPerMinute": 1.0}, fmt=plistlib.FMT_BINARY)[:-12],
            "array root": plistlib.dumps([1, 2], fmt=plistlib.FMT_XML),
            "entity": b'<?xml version="1.0"?><!DOCTYPE plist [<!ENTITY a "aaaa">]><plist version="1.0"><dict>'
                      b'<key>SongKey</key><string>&a;</string></dict></plist>',
            "oversize": b"<?xml version=\"1.0\"?><plist version=\"1.0\"><dict><key>x</key><string>"
                        + b"a" * (4 * 1024 * 1024) + b"</string></dict></plist>",
        }
        for label, raw in cases.items():
            with self.subTest(label=label):
                package = self.make_package(f"{label.replace(' ', '-')}.logicx")
                (package / "Alternatives" / "000" / "MetaData.plist").write_bytes(raw)
                self.assertIn("MetaData.plist", self.failed(package))
        for label, bad in (("tempo text", {"BeatsPerMinute": "fast"}), ("negative tempo", {"BeatsPerMinute": -4.0}),
                           ("signature", {"SongSignatureNumerator": 0}), ("media", {"AudioFiles": "a.wav"}),
                           ("media item", {"AudioFiles": [1]}), ("tracks", {"NumberOfTracks": True})):
            with self.subTest(label=label):
                package = self.make_package(f"{label.replace(' ', '-')}.logicx", meta=bad)
                self.failed(package)
        package = self.make_package("names.logicx")
        (package / "Resources" / "ProjectInformation.plist").write_bytes(plistlib.dumps({"VariantNames": [1]}))
        self.assertIn("VariantNames", self.failed(package))

    def test_symbolic_links_inside_the_package_are_never_followed(self):
        outside = self.dir / "outside.plist"
        outside.write_bytes(plistlib.dumps(self.meta(0)))
        linked_meta = self.make_package("meta.logicx")
        (linked_meta / "Alternatives" / "000" / "MetaData.plist").unlink()
        os.symlink(outside, linked_meta / "Alternatives" / "000" / "MetaData.plist")
        self.assertIn("symbolic link", self.failed(linked_meta))
        linked_info = self.make_package("info.logicx")
        (linked_info / "Resources" / "ProjectInformation.plist").unlink()
        os.symlink(outside, linked_info / "Resources" / "ProjectInformation.plist")
        self.assertIn("symbolic link", self.failed(linked_info))
        linked_alt = self.make_package("alt.logicx")
        real = self.dir / "real-alt"
        (linked_alt / "Alternatives" / "000").rename(real)
        os.symlink(real, linked_alt / "Alternatives" / "000")
        self.assertIn("symbolic link", self.failed(linked_alt))
        linked_dir = self.make_package("dir.logicx")
        os.symlink(self.dir, linked_dir / "Media")
        self.assertIn("symbolic link", self.failed(linked_dir))
        inner = self.make_package("inner.logicx")
        media = inner / "Media"
        media.mkdir()
        os.symlink(self.dir, media / "elsewhere")
        (media / "kick.wav").write_bytes(b"x")
        data = self.project(inner)
        self.assertEqual(self.values(data, "logic.bundled_media"), [{"fileCount": 1, "linksNotFollowed": 1, "truncated": False}])
