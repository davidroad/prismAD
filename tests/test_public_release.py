"""Synthetic rejection/acceptance checks for the public-file boundary."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "public_release_guard",
    Path(__file__).resolve().parents[1] / "scripts/check_public_release.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)

class PublicReleaseTests(unittest.TestCase):
    def test_code_documentation_allowed(self):
        guard.inspect("docs/example.md", b"Methods and input interface.")

    def test_genomic_and_model_containers_rejected(self):
        for name in ["input.pgen", "input.vcf.gz", "model.pt", "records.parquet"]:
            with self.subTest(path=name), self.assertRaises(ValueError):
                guard.inspect(name, b"synthetic")

    def test_tables_and_unapproved_json_rejected(self):
        for name in ["participants.csv", "summary.tsv", "example.json"]:
            with self.subTest(path=name), self.assertRaises(ValueError):
                guard.inspect(name, b"{}")

    def test_changed_or_new_images_rejected(self):
        for name in ["assets/new.png", "assets/figure1a.png"]:
            with self.subTest(path=name), self.assertRaises(ValueError):
                guard.inspect(name, b"unreviewed image")

    def test_identifier_and_secret_patterns_rejected(self):
        examples = ["NACC" + "123456", "ghp_" + "a" * 36]
        for text in examples:
            with self.assertRaises(ValueError):
                guard.inspect("example.md", text.encode())

    def test_disguised_LFS_pointer_rejected(self):
        pointer = b"version " + b"https://git-lfs.github.com/spec/v1" + bytes([10])
        with self.assertRaises(ValueError):
            guard.inspect("example.md", pointer)

    def test_participant_payload_in_configuration_rejected(self):
        with self.assertRaises(ValueError):
            guard.inspect("config/input.example.json",
                          b'{"subject_id": ["synthetic"]}')

if __name__ == "__main__":
    unittest.main()
