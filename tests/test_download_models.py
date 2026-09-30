import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import download_models as models


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        # Keep tiny test artifacts; no user files or existing models are removed.
        self.lora_dir = Path(__file__).resolve().parents[2] / "installer-cache" / f"model-test-{uuid4().hex}"
        self.lora_dir.mkdir(parents=True)
        self.data = b"known-good-test-lora"
        self.digest = hashlib.sha256(self.data).hexdigest()

    def test_existing_verified_lora_is_not_downloaded(self):
        target = self.lora_dir / "adapter.safetensors"
        target.write_bytes(self.data)
        with patch.object(models, "LORA_DIR", self.lora_dir), patch.object(models, "hf_hub_download") as upstream:
            models.download_lora("repo", "revision", target.name, "drive-id", len(self.data), self.digest)
        upstream.assert_not_called()

    def test_google_backup_after_upstream_failure(self):
        def fake_drive_download(**kwargs):
            Path(kwargs["output"]).write_bytes(self.data)
            return kwargs["output"]

        fake_gdown = SimpleNamespace(download=fake_drive_download)
        with patch.object(models, "LORA_DIR", self.lora_dir), \
             patch.object(models, "hf_hub_download", side_effect=RuntimeError("upstream unavailable")), \
             patch.dict(sys.modules, {"gdown": fake_gdown}):
            models.download_lora("repo", "revision", "adapter.safetensors", "drive-id",
                                 len(self.data), self.digest)
        self.assertEqual((self.lora_dir / "adapter.safetensors").read_bytes(), self.data)

    def test_bad_backup_is_not_installed(self):
        def fake_drive_download(**kwargs):
            Path(kwargs["output"]).write_bytes(b"wrong")
            return kwargs["output"]

        fake_gdown = SimpleNamespace(download=fake_drive_download)
        with patch.object(models, "LORA_DIR", self.lora_dir), \
             patch.object(models, "hf_hub_download", side_effect=RuntimeError("upstream unavailable")), \
             patch.dict(sys.modules, {"gdown": fake_gdown}):
            with self.assertRaisesRegex(RuntimeError, "checksum failed"):
                models.download_lora("repo", "revision", "adapter.safetensors", "drive-id",
                                     len(self.data), self.digest)
        self.assertFalse((self.lora_dir / "adapter.safetensors").exists())

    def test_low_vram_adapter_has_no_unverified_google_backup(self):
        with patch.object(models, "LORA_DIR", self.lora_dir), \
             patch.object(models, "hf_hub_download", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "No verified backup"):
                models.download_lora(*models.LOW_TURBO_LORA)


if __name__ == "__main__":
    unittest.main()
