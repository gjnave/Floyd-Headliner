"""Download and verify the app-local fast-core weights, without ComfyUI.

The standard Headliner installation already supplies the BFS and Viggle LoRAs.
This downloads only the INT8 diffusion model, INT8 text encoder, and VAE.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "fast-core"
QWEN_REPO = "Comfy-Org/Qwen-Image-2.1"
QWEN_REVISION = "eee52e917df0f00d910bc1f04f8a4d825dcef255"
VISION_REPO = "Comfy-Org/Qwen3-VL"
VISION_REVISION = "056c1bacea04322faeab9d7c9edb8c3f24460023"

# (repository, immutable revision, repository-relative file, byte length, SHA-256)
WEIGHTS = (
    (QWEN_REPO, QWEN_REVISION,
     "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7256783064,
     "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d"),
    (VISION_REPO, VISION_REVISION,
     "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9350798360,
     "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"),
    (QWEN_REPO, QWEN_REVISION,
     "vae/qwen_image_2.1_vae_bf16.safetensors", 675509688,
     "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
)


def verified(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def main() -> int:
    missing = [(repo, revision, name, size, sha)
               for repo, revision, name, size, sha in WEIGHTS
               if not (MODEL_DIR / name).is_file()]
    required = sum(size for _, _, _, size, _ in missing)
    if required and shutil.disk_usage(APP_DIR).free < required + 2 * 1024**3:
        raise RuntimeError(
            f"Fast-core model downloads need about {required / 1024**3 + 2:.1f} GiB free. "
            "Existing model and output files have not been removed."
        )

    for index, (repo, revision, name, size, sha) in enumerate(WEIGHTS, 1):
        target = MODEL_DIR / name
        print(f"[{index}/{len(WEIGHTS)}] Checking {name}...", flush=True)
        if target.is_file():
            if not verified(target, size, sha):
                raise RuntimeError(
                    f"Existing fast-core model is incomplete or has the wrong hash: {target}. "
                    "It was left unchanged."
                )
            print("Verified existing file.", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading from {repo} at pinned revision {revision}...", flush=True)
        downloaded = Path(hf_hub_download(
            repo_id=repo, revision=revision, filename=name, local_dir=MODEL_DIR,
        ))
        if downloaded.resolve() != target.resolve() or not verified(target, size, sha):
            raise RuntimeError(f"Downloaded fast-core model failed verification: {target}")
        print("SHA-256 verified.", flush=True)

    print("Fast-core model files are ready. No ComfyUI installation is required.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
