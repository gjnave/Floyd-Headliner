from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "Qwen-Image-2.1"
LORA_DIR = MODELS_DIR / "loras"

BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
BFS_REPO = "Alissonerdx/BFS-Best-Face-Swap"
BFS_REVISION = "0ca3913ade4b4ada458d60c232354e8586c4c181"
BFS_FILE = "bfs_head_v1.1_alternative_qwen_2.1.safetensors"
TURBO_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_REVISION = "bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13"
TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
LOW_TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
PROFILE_FILE = MODELS_DIR / "install-profile.txt"

# Public Google Drive copies are used only if a pinned upstream LoRA is unavailable.
# Both Drive copies were downloaded and matched the working local model SHA-256 hashes.
BFS_LORA = (BFS_REPO, BFS_REVISION, BFS_FILE, "18ZKkYWDGzIWrFrrlYJrK--K7_b1wJZdG", 318821008,
            "c5332bbc2f826856e7a09bce9740e97d39514b849368212e1f2c3c2c2d81c217")
STANDARD_TURBO_LORA = (TURBO_REPO, TURBO_REVISION, TURBO_FILE, "1VceHvGZXdu4sO2GJW5njn6ALC1EMW4XM", 1359147904,
                       "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803")
# Publisher's rank-128 cut uses the same six-step schedule as the rank-256 adapter.
LOW_TURBO_LORA = (TURBO_REPO, TURBO_REVISION, LOW_TURBO_FILE, None, 679604800,
                  "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b")


def gib(value: int) -> float:
    return value / (1024**3)


def validate_lora(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def download_lora(repo: str, revision: str, filename: str, drive_id: str | None,
                  size: int, sha256: str) -> None:
    target = LORA_DIR / filename
    if target.exists():
        if validate_lora(target, size, sha256):
            print(f"Verified existing LoRA: {filename}")
            return
        raise RuntimeError(f"Existing LoRA is incomplete or has the wrong checksum; preserved: {target}")

    staging = LORA_DIR / ".downloads"
    staging.mkdir(parents=True, exist_ok=True)
    primary_path = staging / (filename + ".huggingface.part")
    try:
        print(f"Trying pinned Hugging Face revision for {filename}...")
        cached = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
        if primary_path.exists():
            raise RuntimeError(f"Previous staging file must be checked manually: {primary_path}")
        shutil.copyfile(cached, primary_path)
        if not validate_lora(primary_path, size, sha256):
            raise RuntimeError("Hugging Face LoRA did not match the expected size and SHA-256")
        primary_path.rename(target)
        print(f"Installed from Hugging Face: {filename}")
        return
    except Exception as upstream_error:
        print(f"Pinned Hugging Face source unavailable: {upstream_error}")
        if drive_id is None:
            raise RuntimeError(f"No verified backup exists for {filename}; existing files were preserved.") from upstream_error

    import gdown

    backup_path = staging / (filename + ".google-drive.part")
    print(f"Trying Google Drive backup for {filename}...")
    try:
        gdown.download(id=drive_id, output=str(backup_path), quiet=False, resume=True)
    except Exception as backup_error:
        raise RuntimeError(f"Google Drive backup failed for {filename}: {backup_error}") from backup_error
    if not validate_lora(backup_path, size, sha256):
        raise RuntimeError(f"Google Drive backup checksum failed; download preserved for inspection: {backup_path}")
    if target.exists():
        raise RuntimeError(f"LoRA appeared while downloading; refusing to overwrite: {target}")
    backup_path.rename(target)
    print(f"Installed verified Google Drive backup: {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Floyd Headliner models")
    parser.add_argument("--profile", choices=("standard", "low-vram", "low-vram-12gb", "current"), default="current")
    args = parser.parse_args()
    profile = args.profile
    if profile == "current":
        profile = PROFILE_FILE.read_text(encoding="utf-8").strip() if PROFILE_FILE.is_file() else "standard"
    if profile not in {"standard", "low-vram", "low-vram-12gb"}:
        raise RuntimeError(f"Invalid installed model profile: {profile!r}")
    selected_loras = (BFS_LORA, LOW_TURBO_LORA if profile != "standard" else STANDARD_TURBO_LORA)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(APP_DIR).free
    print(f"Free disk space: {gib(free):.1f} GiB")
    if free < 58 * 1024**3 and not (BASE_MODEL_DIR / "model_index.json").is_file():
        raise RuntimeError(
            "At least 58 GiB of free disk space is required for the model, adapters, and download staging."
        )

    print("\n[1/3] Qwen Image 2.1 base model (about 35 GB; resumable)")
    snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        local_dir=BASE_MODEL_DIR,
        max_workers=4,
    )

    for index, lora in enumerate(selected_loras, start=2):
        print(f"\n[{index}/3] {lora[2]}")
        download_lora(*lora)

    required = [BASE_MODEL_DIR / "model_index.json", *(LORA_DIR / lora[2] for lora in selected_loras)]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Download finished with missing files: " + ", ".join(map(str, missing)))

    PROFILE_FILE.write_text(profile + "\n", encoding="utf-8")
    print(f"\nAll {profile} model files are installed. Existing files in the other profile were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
