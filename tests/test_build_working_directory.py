# SPDX-License-Identifier: Apache-2.0
"""Run the real build script through its filesystem preparation phase."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class BuildWorkingDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository with spaces"
        self.repository.mkdir()
        for name in ("build.sh", "CMakeLists.txt"):
            shutil.copyfile(SOURCE_ROOT / name, self.repository / name)
        self.caller = self.root / "caller with spaces"
        self.caller.mkdir()
        self.owned = self.repository / "build" / "previous-build.txt"
        self.unrelated = self.caller / "build" / "valuable.txt"
        for path in (self.owned, self.unrelated):
            path.parent.mkdir()
            path.write_text("preserve unless this is the repository clean build")

    def run_build(self, cwd, *arguments):
        environment = os.environ.copy()
        # Stop at the native installation-path validation, before compilation.
        environment["TRITON_HOME_PATH"] = str(self.root / "missing-triton")
        return subprocess.run(
            ["bash", str(self.repository / "build.sh"), *arguments],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def assert_reaches_installation_check(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("is not a directory", result.stdout)

    def test_external_clean_preserves_callers_build(self):
        result = self.run_build(self.caller)
        self.assert_reaches_installation_check(result)
        self.assertTrue(self.unrelated.is_file())
        self.assertFalse(self.owned.exists())

    def test_repository_subdirectory_uses_repository_build(self):
        nested = self.repository / "example"
        nested.mkdir()
        nested_file = nested / "build" / "valuable.txt"
        nested_file.parent.mkdir()
        nested_file.write_text("unrelated subdirectory build")
        result = self.run_build(nested)
        self.assert_reaches_installation_check(result)
        self.assertTrue(nested_file.is_file())
        self.assertFalse(self.owned.exists())

    def test_external_incremental_keeps_existing_repository_build(self):
        result = self.run_build(self.caller, "--incremental")
        self.assert_reaches_installation_check(result)
        self.assertTrue(self.unrelated.is_file())
        self.assertTrue(self.owned.is_file())

    def test_external_incremental_creates_repository_build_only(self):
        shutil.rmtree(self.owned.parent)
        shutil.rmtree(self.unrelated.parent)
        result = self.run_build(self.caller, "--incremental")
        self.assert_reaches_installation_check(result)
        self.assertTrue(self.repository.joinpath("build").is_dir())
        self.assertFalse(self.caller.joinpath("build").exists())

    def test_root_clean_still_removes_previous_build(self):
        result = self.run_build(self.repository)
        self.assert_reaches_installation_check(result)
        self.assertFalse(self.owned.exists())
        self.assertTrue(self.unrelated.is_file())

    def test_root_incremental_still_keeps_previous_build(self):
        result = self.run_build(self.repository, "--incremental")
        self.assert_reaches_installation_check(result)
        self.assertTrue(self.owned.is_file())

    def test_help_does_not_modify_either_build_directory(self):
        result = self.run_build(self.caller, "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertTrue(self.owned.is_file())
        self.assertTrue(self.unrelated.is_file())

    def test_invalid_abi_preserves_existing_build(self):
        for value in ("invalid", ""):
            with self.subTest(value=value):
                result = self.run_build(self.caller, "--cxx11-abi", value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid --cxx11-abi value", result.stderr)
                self.assertTrue(self.owned.is_file())
                self.assertTrue(self.unrelated.is_file())

    def test_missing_abi_preserves_existing_build(self):
        result = self.run_build(self.caller, "--cxx11-abi")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid --cxx11-abi value", result.stderr)
        self.assertTrue(self.owned.is_file())
        self.assertTrue(self.unrelated.is_file())

    def test_missing_source_marker_rejects_before_cleanup(self):
        self.repository.joinpath("CMakeLists.txt").unlink()
        result = self.run_build(self.caller)
        self.assertEqual(result.returncode, 1)
        self.assertIn("CMakeLists.txt", result.stderr)
        self.assertTrue(self.owned.is_file())
        self.assertTrue(self.unrelated.is_file())

    def test_failed_build_directory_creation_stops_before_cleanup(self):
        shutil.rmtree(self.owned.parent)
        self.owned.parent.write_text("existing file must survive")
        repository_file = self.repository / "valuable.txt"
        repository_file.write_text("not a build artifact")
        result = self.run_build(self.repository)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(repository_file.is_file())
        self.assertTrue(self.owned.parent.is_file())
        self.assertTrue(self.repository.joinpath("CMakeLists.txt").is_file())

    def test_symlink_build_directory_does_not_clear_its_target(self):
        shutil.rmtree(self.owned.parent)
        self.owned.parent.symlink_to(self.unrelated.parent, target_is_directory=True)
        result = self.run_build(self.repository)
        self.assertEqual(result.returncode, 1)
        self.assertIn("symbolic link", result.stderr)
        self.assertTrue(self.unrelated.is_file())


if __name__ == "__main__":
    unittest.main()
