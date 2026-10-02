"""App-local Qwen inference using the bundled upstream core modules.

This imports model loading, Qwen conditioning and sampling components from
vendor/comfy_core. It does not need a ComfyUI install or start its server.
The process stays alive to retain loaded models between requests. JSON lines on
stdout form a private protocol with app.py; diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


MODEL = "qwen_image_2.1_int8_convrot.safetensors"
ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
BFS = "bfs_head_v1.1_alternative_qwen_2.1.safetensors"
TURBO = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"

DEFAULT_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, "
    "environment, and background. remove the head from <image1> completely "
    "and replace it with the head from <image2>, strictly preserving the hair, "
    "eye color, nose structure from <image2>. copy the direction of the eye, "
    "head rotation, micro expressions from <image1>, high quality, sharp details, 4k"
)


def required_base_files(model_root: Path) -> list[Path]:
    return [
        model_root / "diffusion_models" / MODEL,
        model_root / "text_encoders" / ENCODER,
        model_root / "vae" / VAE,
    ]


def required_files(model_root: Path, lora_root: Path) -> list[Path]:
    return required_base_files(model_root) + [
        lora_root / BFS,
        lora_root / TURBO,
    ]


def validate_root(root: Path, model_root: Path, lora_root: Path) -> None:
    if not (root / "nodes.py").is_file() or not (root / "comfy_extras" / "nodes_qwen.py").is_file():
        raise RuntimeError(f"Bundled inference core not found at {root}")
    missing = [str(path) for path in required_base_files(model_root) if not path.is_file()]
    if missing:
        raise RuntimeError("Missing workflow model files:\n" + "\n".join(missing))


def load_image_tensor(path: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        if source.width * source.height > 32_000_000 or max(source.size) > 8192:
            raise ValueError("Reference exceeds 32 megapixels or 8192 pixels on one side.")
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).unsqueeze(0)


class Engine:
    def __init__(self, root: Path, model_root: Path, lora_root: Path):
        validate_root(root, model_root, lora_root)
        os.chdir(root)
        sys.path.insert(0, str(root))
        import torch
        import nodes
        import folder_paths
        from comfy_extras.nodes_qwen import QwenImage21Cache, TextEncodeQwenImage21

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the app's Python environment.")
        for kind in ("diffusion_models", "text_encoders", "vae"):
            folder_paths.add_model_folder_path(kind, str(model_root / kind), is_default=True)
        folder_paths.add_model_folder_path("loras", str(lora_root), is_default=True)
        self.torch = torch
        self.nodes = nodes
        self.cache_node = QwenImage21Cache
        self.text_node = TextEncodeQwenImage21
        self.base_model = self.model = self.clip = self.vae = None
        self.lora_strengths = None
        self.load_seconds = None
        self.conditioning_cache = None

    def load(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        nodes = self.nodes
        model = nodes.UNETLoader().load_unet(MODEL, "default")[0]
        clip = nodes.CLIPLoader().load_clip(ENCODER, "qwen_image", "default")[0]
        vae = nodes.VAELoader().load_vae(VAE)[0]
        self.base_model, self.clip, self.vae = model, clip, vae
        # Plain image editing starts from the untouched model. Likeness transfer
        # applies its adapters only when that operation is requested.
        self.model = self.cache_node.execute(model, "auto", "default")[0]
        self.lora_strengths = (0.0, 0.0)
        self.load_seconds = time.perf_counter() - started

    def set_lora_strengths(self, bfs_strength: float, turbo_strength: float) -> None:
        strengths = (bfs_strength, turbo_strength)
        if strengths == self.lora_strengths:
            return
        model = self.base_model
        loader = self.nodes.LoraLoader()
        if bfs_strength:
            model, _ = loader.load_lora(model, None, BFS, bfs_strength, 0.0)
        if turbo_strength:
            model, _ = loader.load_lora(model, None, TURBO, turbo_strength, 0.0)
        model = self.cache_node.execute(model, "auto", "default")[0]
        self.model = model
        self.lora_strengths = strengths

    def generate(self, request: dict) -> dict:
        import numpy as np
        from PIL import Image

        request_started = time.perf_counter()
        body_path = str(request["body_path"])
        head_path = str(request["head_path"])
        output_dir = Path(request["output_dir"]).resolve()
        prompt = str(request.get("prompt") or DEFAULT_PROMPT).strip()
        extra = str(request.get("extra_prompt") or "").strip()
        if extra:
            prompt = f"{prompt}, {extra}"
        steps = int(request.get("steps", 6))
        resolution = int(request.get("resolution", 1024))
        upscale = int(request.get("upscale", 2))
        bfs_strength = float(request.get("bfs_strength", 1.0))
        turbo_strength = float(request.get("turbo_strength", 1.0))
        if not 1 <= steps <= 60 or not 256 <= resolution <= 2048 or upscale not in (1, 2):
            raise ValueError("Invalid steps, reference resolution, or output scale.")
        if not 0 <= bfs_strength <= 1.5 or not 0 <= turbo_strength <= 1.25:
            raise ValueError("Invalid LoRA strength.")
        seed = int(request.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**63)
        if seed >= 2**64:
            raise ValueError("Seed must be below 2^64.")

        self.load()
        self.set_lora_strengths(bfs_strength, turbo_strength)
        started = time.perf_counter()
        def image_key(path):
            stat = Path(path).stat()
            return (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)

        cache_key = (image_key(body_path), image_key(head_path), prompt, resolution)
        cache_hit = self.conditioning_cache is not None and self.conditioning_cache[0] == cache_key
        if cache_hit:
            _, positive, negative, cached_latent = self.conditioning_cache
            latent = {**cached_latent, "samples": cached_latent["samples"].clone()}
        else:
            body = load_image_tensor(body_path)
            head = load_image_tensor(head_path)
            with self.torch.no_grad():
                positive, negative, latent = self.text_node.execute(
                    self.clip, prompt, "", vae=self.vae, resolution=resolution,
                    images={"image_1": body, "image_2": head},
                )
            self.torch.cuda.synchronize()
            self.conditioning_cache = (cache_key, positive, negative, latent)
        conditioning_seconds = time.perf_counter() - started
        with self.torch.no_grad():
            sampled = self.nodes.KSampler().sample(
                model=self.model, seed=seed, steps=steps, cfg=1.0,
                sampler_name="euler_ancestral", scheduler="simple",
                positive=positive, negative=negative, latent_image=latent, denoise=0.95,
            )[0]
        self.torch.cuda.synchronize()
        sampling_seconds = time.perf_counter() - started - conditioning_seconds
        with self.torch.no_grad():
            pixels = self.nodes.VAEDecode().decode(self.vae, sampled)[0][0]
        self.torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started - conditioning_seconds - sampling_seconds
        image = Image.fromarray((pixels.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
        if upscale == 2:
            image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"headliner-fast-core-{int(time.time() * 1000)}-seed-{seed}.png"
        image.save(output)
        self.torch.cuda.synchronize()
        return {
            "output": str(output), "seed": seed,
            "working_size": [image.width // upscale, image.height // upscale],
            "loader_setup_seconds": round(self.load_seconds, 2),
            "conditioning_cache_hit": cache_hit,
            "conditioning_seconds": round(conditioning_seconds, 2),
            "sampling_seconds": round(sampling_seconds, 2),
            "decode_seconds": round(decode_seconds, 2),
            "save_seconds": round(time.perf_counter() - started - conditioning_seconds - sampling_seconds - decode_seconds, 2),
            "total_seconds_excluding_load": round(time.perf_counter() - started, 2),
            "total_seconds": round(time.perf_counter() - request_started, 2),
        }

    def generate_inpaint(self, request: dict) -> dict:
        import numpy as np
        from PIL import Image, PngImagePlugin

        request_started = time.perf_counter()
        source_path = str(request["image_path"])
        output_dir = Path(request["output_dir"]).resolve()
        prompt = str(request.get("prompt") or "").strip()
        steps = int(request.get("steps", 40))
        resolution = int(request.get("resolution", 1024))
        seed = int(request.get("seed", -1))
        if not prompt:
            raise ValueError("Type what you want to change before generating.")
        if not 1 <= steps <= 60 or not 256 <= resolution <= 2048:
            raise ValueError("Invalid edit steps or working resolution.")
        if seed < 0:
            seed = random.randrange(2**63)
        if seed >= 2**63:
            raise ValueError("Seed must be below 2^63.")

        with Image.open(source_path) as source:
            original_size = source.size
        self.load()
        self.set_lora_strengths(0.0, 0.0)
        started = time.perf_counter()
        stat = Path(source_path).stat()
        cache_key = ("inpaint", str(Path(source_path).resolve()), stat.st_size,
                     stat.st_mtime_ns, prompt, resolution)
        cache_hit = self.conditioning_cache is not None and self.conditioning_cache[0] == cache_key
        if cache_hit:
            _, positive, negative, cached_latent = self.conditioning_cache
            latent = {**cached_latent, "samples": cached_latent["samples"].clone()}
        else:
            image = load_image_tensor(source_path)
            with self.torch.no_grad():
                positive, negative, latent = self.text_node.execute(
                    self.clip, prompt, "", vae=self.vae, resolution=resolution,
                    images={"image_1": image},
                )
            self.torch.cuda.synchronize()
            self.conditioning_cache = (cache_key, positive, negative, latent)
        conditioning_seconds = time.perf_counter() - started
        with self.torch.no_grad():
            sampled = self.nodes.KSampler().sample(
                model=self.model, seed=seed, steps=steps, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=positive, negative=negative, latent_image=latent, denoise=1.0,
            )[0]
            pixels = self.nodes.VAEDecode().decode(self.vae, sampled)[0][0]
        self.torch.cuda.synchronize()
        working_size = (pixels.shape[1], pixels.shape[0])
        result = Image.fromarray((pixels.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"floyd-inpaint-{int(time.time() * 1000)}-seed-{seed}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Floyd Headliner", "App-local Qwen Image 2.1 core edit; LoRAs disabled")
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        result.save(output, pnginfo=metadata)
        return {
            "output": str(output), "seed": seed,
            "working_size": list(working_size),
            "conditioning_cache_hit": cache_hit,
            "conditioning_seconds": round(conditioning_seconds, 2),
            "total_seconds": round(time.perf_counter() - request_started, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    # ComfyUI parses sys.argv at import; keep this worker's flags out of it.
    sys.argv = [sys.argv[0], "--models-directory", str(args.model_root.resolve())]
    try:
        engine = Engine(args.core_root.resolve(), args.model_root.resolve(), args.lora_root.resolve())
    except Exception as error:
        protocol.write(json.dumps({"ready": False, "error": str(error)}) + "\n")
        protocol.flush()
        traceback.print_exc()
        return 1
    protocol.write(json.dumps({"ready": True}) + "\n")
    protocol.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "stop":
                break
            if request.get("mode") == "inpaint":
                response = {"ok": True, **engine.generate_inpaint(request)}
            else:
                response = {"ok": True, **engine.generate(request)}
        except Exception as error:
            response = {"ok": False, "error": str(error)}
            traceback.print_exc()
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
