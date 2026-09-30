"""Manual two-reference CUDA smoke/latency comparison; does not save images."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reference(color: str, hair: str) -> Image.Image:
    image = Image.new("RGB", (512, 512), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((145, 85, 365, 345), fill="#d5a072")
    draw.pieslice((140, 75, 370, 280), 180, 360, fill=hair)
    draw.ellipse((205, 210, 225, 225), fill="black")
    draw.ellipse((290, 210, 310, 225), fill="black")
    draw.arc((220, 255, 295, 295), 5, 175, fill="black", width=5)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=("standard", "low-vram"))
    parser.add_argument("--size", type=int, default=704)
    parser.add_argument("--vram-cap-gib", type=float)
    args = parser.parse_args()
    os.environ["FLOYD_HEADLINER_PROFILE"] = args.profile

    import torch
    import app

    if args.vram_cap_gib:
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(args.vram_cap_gib / total_gib, 0)

    body = reference("#a0b8a0", "#3c2d20")
    head = reference("#9ab4d8", "#c9a75b")
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipe = app._load_pipeline()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    app.configure_speed_mode(pipe, True, args.size, args.size)
    pipe.set_adapters(["bfs", "turbo"], adapter_weights=[1.0, 1.0])
    generation_seconds = []
    for seed in (42, 43):
        generate_started = time.perf_counter()
        output = pipe(
            prompt=app.DEFAULT_PROMPT,
            image=[body, head],
            width=args.size,
            height=args.size,
            num_inference_steps=6,
            true_cfg_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize()
        generation_seconds.append(round(time.perf_counter() - generate_started, 2))
    print("BENCHMARK " + json.dumps({
        "profile": args.profile,
        "canvas": args.size,
        "result": list(output.size),
        "load_seconds": round(load_seconds, 2),
        "generation_seconds_first_and_cached": generation_seconds,
        "max_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "max_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "offload_mode": app._OFFLOAD_MODE,
        "pytorch_vram_cap_gib": args.vram_cap_gib,
    }), flush=True)


if __name__ == "__main__":
    main()
