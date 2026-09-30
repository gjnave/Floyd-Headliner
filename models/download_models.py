from __future__ import annotations

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


def gib(value: int) -> float:
    return value / (1024**3)


def main() -> int:
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

    print("\n[2/3] BFS v1.1 alternative LoRA")
    hf_hub_download(
        repo_id=BFS_REPO,
        filename=BFS_FILE,
        revision=BFS_REVISION,
        local_dir=LORA_DIR,
    )

    print("\n[3/3] Viggle six-step LoRA")
    hf_hub_download(
        repo_id=TURBO_REPO,
        filename=TURBO_FILE,
        revision=TURBO_REVISION,
        local_dir=LORA_DIR,
    )

    required = [BASE_MODEL_DIR / "model_index.json", LORA_DIR / BFS_FILE, LORA_DIR / TURBO_FILE]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Download finished with missing files: " + ", ".join(map(str, missing)))

    print("\nAll model files are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
