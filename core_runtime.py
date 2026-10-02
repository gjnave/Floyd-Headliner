"""App-local inference core for the existing Headliner interface.

The required upstream Python modules live under vendor/comfy_core. The worker
uses the app's private environment and never needs a ComfyUI installation or
server. Model weights live in the app's models directory or an explicit path.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from core_worker import required_files


HERE = Path(__file__).resolve().parent
CORE_ROOT = Path(os.environ.get("FLOYD_CORE_ROOT", str(HERE / "vendor" / "comfy_core")))
_LOCAL_MODELS = HERE / "models" / "fast-core"
LORA_ROOT = HERE / "models" / "loras"
MODEL_ROOT = Path(os.environ.get("FLOYD_CORE_MODEL_DIR", str(_LOCAL_MODELS)))
WORKER_PYTHON = Path(os.environ.get("FLOYD_CORE_PYTHON", sys.executable))


def available() -> bool:
    return (
        WORKER_PYTHON.is_file()
        and (CORE_ROOT / "nodes.py").is_file()
        and all(path.is_file() for path in required_files(MODEL_ROOT, LORA_ROOT))
    )


def _readline_with_timeout(stream, seconds: int) -> str:
    result: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: result.put(stream.readline()), daemon=True).start()
    try:
        return result.get(timeout=seconds)
    except queue.Empty as error:
        raise TimeoutError(f"Fast-core worker did not respond within {seconds} seconds.") from error


class Worker:
    def __init__(self):
        self.process = None
        self.log = None
        self.lock = threading.RLock()

    def start(self, output_dir: Path) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not available():
            raise RuntimeError(
                "Fast core needs its bundled inference modules and Qwen/BFS/Viggle "
                "model files. Choose Standalone in Advanced settings if they are not installed."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log = (output_dir / "fast-core-worker.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(WORKER_PYTHON), "-u", str(HERE / "core_worker.py"),
             "--core-root", str(CORE_ROOT), "--model-root", str(MODEL_ROOT),
             "--lora-root", str(LORA_ROOT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            line = _readline_with_timeout(self.process.stdout, 60)
            ready = json.loads(line) if line else {}
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "Fast-core worker failed to start."))
        except Exception:
            self.stop()
            raise

    def request(self, payload: dict, output_dir: Path) -> dict:
        with self.lock:
            self.start(output_dir)
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                line = _readline_with_timeout(self.process.stdout, 300)
                answer = json.loads(line) if line else {}
            except (OSError, ValueError, TimeoutError) as error:
                self.stop()
                raise RuntimeError(
                    f"Fast-core worker stopped or timed out. See {output_dir / 'fast-core-worker.log'}"
                ) from error
            if not answer.get("ok"):
                raise RuntimeError(answer.get("error", "Fast-core generation failed."))
            return answer

    def stop(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            if self.log is not None:
                self.log.close()
                self.log = None


WORKER = Worker()
atexit.register(WORKER.stop)


def release_models() -> str:
    WORKER.stop()
    return "Fast-core models released from GPU memory. Saved files remain unchanged."


def _cached_input(image: Image.Image, output_dir: Path) -> Path:
    """Stable filenames preserve the worker's image/prompt conditioning cache."""
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    folder = output_dir / "fast-core-input-cache"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest.hexdigest()}.png"
    if not path.is_file() or path.stat().st_size == 0:
        image.save(path, format="PNG")
    return path


def run_swap(runtime, body_image, head_image, prompt, seed, randomize_seed,
             bfs_strength, turbo_strength, steps, working_megapixels,
             output_upscale, keep_on_gpu, extra_prompt, selected_only,
             feather, mask_mode):
    if body_image is None or head_image is None:
        raise ValueError("Add both the body reference and head reference.")
    if not prompt or "<image1>" not in prompt or "<image2>" not in prompt:
        raise ValueError("The likeness instruction must include <image1> and <image2>.")
    if mask_mode not in ("Edit painted area", "Protect painted area (experimental)"):
        raise ValueError("Choose a valid painted-area mode.")
    if not 0 <= float(bfs_strength) <= 1.5 or not 0 <= float(turbo_strength) <= 1.25:
        raise ValueError("Choose valid BFS and Viggle LoRA strengths.")

    started = time.perf_counter()
    selected_only = bool(selected_only or runtime.has_painted_mask(body_image))
    original, mask, crop = runtime.prepare_selection(body_image, selected_only)
    head = ImageOps.exif_transpose(head_image).convert("RGB")
    effective_prompt = runtime.compose_prompt(prompt, extra_prompt)
    protect = selected_only and mask_mode == "Protect painted area (experimental)"
    if protect:
        body, protected = runtime.protected_reference(original, mask)
        side_crop = runtime.unprotected_side_crop(protected)
        if side_crop:
            crop = side_crop
            body = original.crop(crop)
        else:
            crop = (0, 0, original.width, original.height)
        mask = ImageOps.invert(protected)
        if mask.crop(crop).getbbox() is None:
            raise ValueError("Leave the target head unpainted; the working area is protected.")
        if side_crop:
            effective_prompt += (
                "\n<image1> is the unprotected person cropped from the group photo. "
                "Transfer the reference likeness to that visible person's head. "
                "Keep their pose, clothing, and surroundings."
            )
        else:
            effective_prompt += (
                "\nThe flat gray covered regions in <image1> hide protected heads. "
                "Transfer the reference likeness only to the remaining visible head. "
                "Do not create a new head in the covered regions."
            )
    else:
        body = original.crop(crop) if crop else original

    if not randomize_seed and (seed is None or not 0 <= int(seed) < 2**63):
        raise ValueError("Seed must be between 0 and 9223372036854775807.")
    used_seed = random.randrange(2**63) if randomize_seed else int(seed)
    megapixels = float(working_megapixels)
    if not 0.25 <= megapixels <= 2.0:
        raise ValueError("Working megapixels must be between 0.25 and 2.0.")
    resolution = round((megapixels ** 0.5) * 1024 / 32) * 32

    # The two backends use separate large model sets. Avoid holding both.
    if runtime._PIPELINE is not None:
        runtime.release_standalone_models()
    output_dir = runtime.OUTPUT_DIR
    response = WORKER.request({
        "body_path": str(_cached_input(body, output_dir)),
        "head_path": str(_cached_input(head, output_dir)),
        "prompt": effective_prompt,
        "seed": used_seed,
        "steps": int(steps),
        "resolution": resolution,
        "upscale": 1 if selected_only else int(output_upscale),
        "bfs_strength": float(bfs_strength),
        "turbo_strength": float(turbo_strength),
        "output_dir": str(output_dir),
    }, output_dir)
    with Image.open(response["output"]) as generated:
        result = generated.convert("RGB")
    output_path = Path(response["output"])
    if selected_only:
        result = runtime.composite_selection(original, result, mask, crop, feather)
        output_path = output_dir / (
            f"floyd-core-mask-{datetime.now():%Y%m%d-%H%M%S-%f}-seed-{used_seed}.png"
        )
        result.save(output_path, format="PNG")
    if not keep_on_gpu:
        WORKER.stop()
    status = (
        f"Saved `{output_path}`  \n"
        f"Seed: `{used_seed}` | Working canvas: "
        f"`{response['working_size'][0]} x {response['working_size'][1]}` | "
        f"Output: `{result.width} x {result.height}`  \n"
        f"Time: `{time.perf_counter() - started:.1f} s` | Fast app-local core "
        f"(no ComfyUI install or server) | Reference/prompt cache: "
        f"`{'reused' if response['conditioning_cache_hit'] else 'encoded'}`"
    )
    if selected_only:
        status += "  \nPainted-area result restored to original dimensions."
    return result, str(output_path), status, used_seed
