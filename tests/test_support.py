"""The in-process CLI runner must fail the way a child interpreter would, or the suite stops proving anything."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import run_cli  # noqa: E402
from music_guide import cli  # noqa: E402


class InProcessRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()

    def test_a_warning_in_the_analysis_fails_the_run_like_pythonwarnings_error(self):
        def noisy(*_args):
            warnings.warn("stray warning", DeprecationWarning)
            raise AssertionError("the warning above should have been raised first")
        with mock.patch.object(cli, "build", noisy):
            result = run_cli("--midi", self.dir / "a.mid", cwd=self.dir)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("DeprecationWarning: stray warning", result.stderr)
        self.assertNotIn("AssertionError", result.stderr)

    def test_an_escaping_exception_is_a_traceback_and_status_1(self):
        with mock.patch.object(cli, "build", side_effect=RuntimeError("boom")):
            result = run_cli("--midi", self.dir / "a.mid", cwd=self.dir)
        self.assertEqual(result.returncode, 1)
        self.assertIn("RuntimeError: boom", result.stderr)

    def test_argparse_exit_is_reported_not_raised(self):
        result = run_cli(cwd=self.dir)
        self.assertEqual((result.returncode, result.stdout), (2, ""))
        self.assertIn("usage: analyze.py", result.stderr)

    def test_state_is_restored_after_every_run(self):
        before = (list(sys.argv), os.getcwd(), sys.stderr, list(warnings.filters))
        run_cli(cwd=self.dir)
        self.assertEqual((sys.argv, os.getcwd(), sys.stderr, warnings.filters), before)

    def test_output_written_straight_to_descriptor_2_is_captured_in_order(self):
        def chatty(*_args):
            sys.stderr.write("python first\n")
            os.write(2, b"native second\n")
            sys.stderr.write("python third\n")
            raise ValueError("last")
        with mock.patch.object(cli, "build", chatty):
            result = run_cli("--midi", self.dir / "a.mid", cwd=self.dir)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "python first\nnative second\npython third\nmusic-guide: last\n")

    def test_the_working_directory_is_the_one_asked_for(self):
        seen = []

        def probe(*_args):
            seen.append(os.getcwd())
            raise ValueError("stop here")
        with mock.patch.object(cli, "build", probe):
            run_cli("--midi", "a.mid", cwd=self.dir)
        self.assertEqual([Path(item).resolve() for item in seen], [self.dir])


if __name__ == "__main__":
    unittest.main()
