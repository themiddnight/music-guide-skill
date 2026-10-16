"""Untrusted meter maps cannot allocate a metrical grid outside project bounds."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import run_cli
from music_guide import rhythm
import json


class MeterMapBoundsTest(unittest.TestCase):
    def test_hostile_musicxml_keeps_notes_without_allocating_giant_grid(self):
        xml = """<score-partwise><part-list><score-part id="P1"><part-name>Piano</part-name></score-part>
        </part-list><part id="P1"><measure number="1"><attributes><divisions>1</divisions>
        <time><beats>536870912</beats><beat-type>4</beat-type></time></attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note></measure>
        <measure number="2"><attributes><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
        <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note></measure>
        </part></score-partwise>"""
        original = rhythm.Meter.weights
        def bounded(meter, subdivision):
            self.assertLessEqual(meter.numerator, 999, 'hostile meter reached allocation')
            return original(meter, subdivision)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hostile.musicxml'
            path.write_text(xml)
            with mock.patch.object(rhythm.Meter, 'weights', bounded):
                result = run_cli('--project', path, cwd=path.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        summary = next(item for item in report['observations'] if item['property'] == 'project.track_rhythm')
        self.assertIsNone(summary['value']['syncopation'])
        self.assertIsNone(summary['value']['onsetsPerBar'])
        self.assertIn('project.track_notes', {item['property'] for item in report['observations']})
        self.assertIn('implausible_value', {item['code'] for item in report['diagnostics']})


if __name__ == '__main__':
    unittest.main()
