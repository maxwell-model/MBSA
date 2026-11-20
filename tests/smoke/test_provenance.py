# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free provenance integrity and deliberate-corruption tests."""

import copy
import hashlib
import json
import os
import subprocess
import unittest

from tests.smoke.test_packaging import ROOT, check_optimized_manifest


MANIFEST_PATH = "mbsa/backends/optimized/UPSTREAM.json"
BASE = "07ebee24d561190437bae8740fabeea0f506d0b7"


class ProvenanceContracts(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "python" / MANIFEST_PATH).read_bytes())

    def validate(self, manifest=None, overrides=None):
        manifest = self.manifest if manifest is None else manifest
        overrides = {} if overrides is None else overrides

        def read_bytes(name):
            if name == MANIFEST_PATH:
                return json.dumps(manifest).encode()
            return overrides[name] if name in overrides else (ROOT / "python" / name).read_bytes()

        check_optimized_manifest(read_bytes)

    def test_actual_manifest(self):
        self.validate()

    def test_local_corruption_is_rejected(self):
        target = self.manifest["files"][1]["path"].removeprefix("python/")
        with self.assertRaises(AssertionError):
            self.validate(overrides={target: b"corrupted source"})

    def test_false_local_hash_is_rejected(self):
        self.manifest["files"][0]["local_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            self.validate()

    def test_false_copied_blob_is_rejected(self):
        record = next(r for r in self.manifest["files"] if r["relationship"] == "copied")
        record["upstream_git_blob"] = "0" * 40
        with self.assertRaises(AssertionError):
            self.validate()

    def test_malformed_revision_is_rejected(self):
        self.manifest["commit"] = "not-a-revision"
        with self.assertRaises(AssertionError):
            self.validate()

    def test_license_scope_is_required(self):
        for value in (None, "historical-repository-grant"):
            with self.subTest(value=value):
                manifest = copy.deepcopy(self.manifest)
                if value is None:
                    del manifest["license_scope"]
                else:
                    manifest["license_scope"] = value
                with self.assertRaises(AssertionError):
                    self.validate(manifest)

    def test_repository_is_data_not_a_private_url_guard(self):
        self.manifest["repository"] = "https://example.org/source/project"
        self.validate()

    def test_repository_credentials_and_malformed_urls_are_rejected(self):
        for value in ("file:///tmp/source", "https://user:secret@example.org/repo",
                      "https://example.org/repo?token=value", "https://example.org/repo#fragment",
                      "https://example.org", "https://example.org/re po"):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                manifest = copy.deepcopy(self.manifest)
                manifest["repository"] = value
                self.validate(manifest)

    def test_unsafe_source_paths_are_rejected(self):
        for value in ("../source.py", "/source.py", "a/../../source.py", "a\\source.py"):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][1]["upstream_path"] = value
                self.validate(manifest)

    def test_duplicate_target_is_rejected(self):
        self.manifest["files"].append(copy.deepcopy(self.manifest["files"][0]))
        with self.assertRaises(AssertionError):
            self.validate()

    def test_unverified_intermediate_cannot_be_promoted_implicitly(self):
        record = next(r for r in self.manifest["files"] if "intermediate_source" in r)
        record["intermediate_source"]["status"] = "verified"
        with self.assertRaises(AssertionError):
            self.validate()

    def test_earlier_origin_gaps_are_not_implicitly_resolved(self):
        self.manifest["earlier_origin_gaps"] = []
        with self.assertRaises(AssertionError):
            self.validate()

    def test_customer_preimages_match_available_git_objects(self):
        env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
        available = subprocess.run(
            ["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=ROOT, env=env,
            capture_output=True,
        )
        if available.returncode:
            self.skipTest("Customer-base Git objects unavailable; archive cannot verify preimages")
        records = [r for r in self.manifest["files"] if "reference_role" in r]
        self.assertEqual(len(records), 5)
        for record in records:
            with self.subTest(path=record["path"]):
                self.assertEqual(record["commit"], BASE)
                blob = subprocess.check_output(
                    ["git", "rev-parse", f"{BASE}:{record['upstream_path']}"], cwd=ROOT, env=env,
                ).decode().strip()
                self.assertEqual(blob, record["upstream_git_blob"])
                source = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT, env=env)
                header = b"blob " + str(len(source)).encode() + b"\0"
                self.assertEqual(hashlib.sha1(header + source).hexdigest(), blob)

    def test_derived_host_files_retain_customer_attribution(self):
        for name in ("index.py", "index_base.py"):
            source = (ROOT / "python/mbsa/backends/optimized" / name).read_text()
            self.assertIn("# Copyright 2026 Maxwell Team.", source.splitlines()[:5])
            self.assertIn("NVIDIA CORPORATION", source)
            self.assertIn("SPDX-License-Identifier: Apache-2.0", source)


if __name__ == "__main__":
    unittest.main()
