import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import download_models as models


class DownloadModelsTests(unittest.TestCase):
    def test_current_profile_retains_twelve_gb_mode(self):
        from unittest.mock import Mock
        profile = Mock()
        profile.is_file.return_value = True
        profile.read_text.return_value = "low-vram-12gb\n"
        with patch.object(models, "PROFILE_FILE", profile), \
             patch.object(sys, "argv", ["download_models.py", "--profile", "current"]), \
             patch.object(models, "snapshot_download"), \
             patch.object(models, "download_lora") as download, \
             patch.object(Path, "mkdir"), patch.object(Path, "is_file", return_value=True), \
             patch.object(models.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
            self.assertEqual(models.main(), 0)
        self.assertEqual(download.call_args_list[-1].args, models.LOW_TURBO_LORA)
        profile.write_text.assert_called_once_with("low-vram-12gb\n", encoding="utf-8")

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
