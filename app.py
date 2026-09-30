from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, ImageChops, ImageFilter


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "Qwen-Image-2.1"
LORA_DIR = MODELS_DIR / "loras"
OUTPUT_DIR = APP_DIR / "outputs"

BASE_MODEL_MARKER = BASE_MODEL_DIR / "model_index.json"
BFS_LORA = LORA_DIR / "bfs_head_v1.1_alternative_qwen_2.1.safetensors"
TURBO_LORA = LORA_DIR / "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
LOW_TURBO_LORA = LORA_DIR / "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
PROFILE_FILE = MODELS_DIR / "install-profile.txt"

DEFAULT_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, "
    "environment, and background. remove the head from <image1> completely and "
    "replace it with the head from <image2>, strictly preserving the hair, eye "
    "color, nose structure from <image2>. copy the direction of the eye, head "
    "rotation, micro expressions from <image1>, high quality, sharp details, 4k"
)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_PIPELINE = None
_PIPELINE_LOCK = threading.RLock()
_OFFLOAD_MODE = "not loaded"
_RESIDENT_ALLOWED = False
APP_CSS = """
:root {
    --ggf-bg: #07101f;
    --ggf-panel: #111c2f;
    --ggf-line: #2a3b56;
    --ggf-ink: #e8eef9;
    --ggf-muted: #9db0cc;
    --ggf-gold: #f5b942;
}
body, .gradio-container {
    background: var(--ggf-bg) !important;
    color: var(--ggf-ink) !important;
}
.gradio-container {
    --body-background-fill: var(--ggf-bg);
    --body-text-color: var(--ggf-ink);
    --block-background-fill: var(--ggf-panel);
    --block-border-color: var(--ggf-line);
    --input-background-fill: #091425;
    --input-border-color: #3a4f70;
    --button-primary-background-fill: var(--ggf-gold);
    --button-primary-text-color: #111827;
}
.app-shell { max-width: 1320px; margin: auto; padding: 18px 12px 36px; }
.ggf-hero {
    background: linear-gradient(135deg, #0b1220, #172554);
    border: 1px solid #334a6b;
    border-radius: 18px;
    padding: 25px 28px;
    margin-bottom: 14px;
}
.ggf-hero .ggf-kicker {
    color: var(--ggf-gold);
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .14em;
}
.ggf-hero h1 { color: var(--ggf-gold); margin: 7px 0; font-size: clamp(1.8rem, 3vw, 2.6rem); }
.ggf-hero p { color: #dbeafe; margin: 0 0 8px; }
.ggf-hero a, .ggf-footer a { color: #8fc2ff; font-weight: 700; }
.ggf-hero .ggf-links { display: flex; gap: 10px 18px; flex-wrap: wrap; }
.block.ggf-guide, .block.warning {
    background: var(--ggf-panel);
    border: 1px solid var(--ggf-line);
    border-radius: 12px;
    padding: 10px 16px;
}
.block.warning { border-left: 4px solid var(--ggf-gold); }
.prose.ggf-guide, .prose.warning { background: transparent; border: 0; padding: 0; }
[data-testid="block-label"] {
    background: var(--ggf-panel) !important;
    border-color: var(--ggf-line) !important;
    color: var(--ggf-ink) !important;
}
.gradio-container [data-testid="block-info"] {
    color: #cdd9ed !important;
}
.gr-accordion .label-wrap, .gr-accordion .label-wrap span {
    color: var(--ggf-ink) !important;
}
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: #091425 !important;
    color: var(--ggf-ink) !important;
    caret-color: var(--ggf-gold);
}
.gradio-container textarea::placeholder,
.gradio-container input[type="text"]::placeholder {
    color: var(--ggf-muted) !important;
    opacity: 1;
}
.ggf-status code {
    background: #172a43 !important;
    color: var(--ggf-ink) !important;
    border: 1px solid #36516f;
    border-radius: 4px;
    overflow-wrap: anywhere;
    white-space: break-spaces;
}
.ggf-status { overflow-wrap: anywhere; }
.ggf-download tr.file,
.ggf-download tr.file:hover {
    background: #172a43 !important;
    color: var(--ggf-ink) !important;
}
.ggf-download tr.file .filename { color: var(--ggf-ink) !important; }
.ggf-download tr.file .download-link { color: #9dccff !important; }
[data-testid="toast-body"] {
    background: #111c2f !important;
    color: #e8eef9 !important;
    border: 1px solid #b95d66 !important;
    box-shadow: 0 16px 36px rgba(0, 0, 0, .45) !important;
}
[data-testid="toast-body"] .toast-header,
[data-testid="toast-body"] .toast-message-text,
[data-testid="toast-body"] .toast-title,
[data-testid="toast-body"] button {
    color: #e8eef9 !important;
}
[data-testid="toast-body"] .toast-message-text {
    overflow-wrap: anywhere;
}
#ggf-swap-button, #ggf-inpaint-button { background: var(--ggf-gold) !important; color: #111827 !important; font-weight: 800 !important; }
.tab-nav button { color: var(--ggf-ink) !important; }
.tab-nav button.selected { color: var(--ggf-gold) !important; }
.ggf-footer { color: var(--ggf-muted); text-align: center; margin-top: 20px; font-size: .88rem; }
@media (max-width: 680px) { .app-shell { padding: 10px 6px 24px; } .ggf-hero { padding: 18px; } }
"""
APP_JS = """
() => {
    document.addEventListener('keydown', (event) => {
        if (!event.ctrlKey || event.key !== 'Enter' || event.repeat) return;
        const button = ['ggf-inpaint-button', 'ggf-swap-button']
            .map(id => document.getElementById(id))
            .find(button => button && button.getClientRects().length > 0);
        if (!button || button.disabled) return;
        event.preventDefault();
        event.stopPropagation();
        button.click();
    }, true);
}
"""


def calculate_working_size(image: Image.Image, megapixels: float = 1.0) -> tuple[int, int]:
    """Match the workflow's active latent sizing: preserve ratio and round to 32."""
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("The base image has invalid dimensions.")
    if model_profile() == "low-vram-12gb":
        megapixels = min(float(megapixels), 0.5)
    target_area = max(0.25, float(megapixels)) * 1_048_576
    ratio = width / height
    scaled_width = max(32, round((target_area * ratio) ** 0.5 / 32) * 32)
    scaled_height = max(32, round((target_area / ratio) ** 0.5 / 32) * 32)
    if model_profile() == "low-vram-12gb":
        scaled_width = max(32, int((target_area * ratio) ** 0.5 / 32) * 32)
        scaled_height = max(32, int((target_area / ratio) ** 0.5 / 32) * 32)
    return scaled_width, scaled_height


def compose_prompt(prompt: str, extra_prompt: str | None = None) -> str:
    """Append an optional user instruction after the unchanged head-swap prompt."""
    main = prompt.strip()
    extra = (extra_prompt or "").strip()
    return f"{main}\n{extra}" if extra else main


def has_painted_mask(body_input) -> bool:
    """Keep the edit checkbox in sync with visible brush strokes."""
    if not isinstance(body_input, dict) or body_input.get("background") is None:
        return False
    for layer in body_input.get("layers") or []:
        if layer is not None and layer.mode == "RGBA" and layer.getchannel("A").getbbox():
            return True
    return False


def painted_mask_from_editor_payload(editor, raw_input):
    """Read editor changes without making a transient PNG decode a Gradio error."""
    if raw_input is None:
        return False
    for attempt in range(3):
        try:
            payload = editor.data_model.model_validate(raw_input)
            return has_painted_mask(editor.preprocess(payload))
        except (OSError, SyntaxError):
            if attempt < 2:
                time.sleep(0.15 * (attempt + 1))
    import gradio as gr

    gr.Warning("The body image is still being prepared. Wait a moment, then paint again or re-upload it before generating.")
    return gr.skip()


def prepare_selection(body_input, selected_only=False):
    """Use the editor background, never the painted composite, as model input."""
    background = body_input.get("background") if isinstance(body_input, dict) else body_input
    if background is None:
        raise ValueError("Add the body/base image (Image 1).")
    body = ImageOps.exif_transpose(background).convert("RGB")
    if not selected_only:
        return body, None, None
    mask = Image.new("L", body.size, 0)
    for layer in body_input.get("layers", []) if isinstance(body_input, dict) else []:
        if layer is None:
            continue
        if layer.size != body.size or layer.mode != "RGBA":
            raise ValueError("Selection does not match the body image. Upload it again and repaint.")
        mask = ImageChops.lighter(mask, layer.getchannel("A"))
    mask = mask.point(lambda value: 255 if value else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("Paint over the intended head first, or turn off 'Edit painted area only'.")
    left, top, right, bottom = box
    padding = max(32, round(max(right - left, bottom - top) * 0.5))
    crop = (max(0, left - padding), max(0, top - padding),
            min(body.width, right + padding), min(body.height, bottom + padding))
    return body, mask, crop


def composite_selection(body, generated, mask, crop, feather):
    """Blend inward only: every unpainted original pixel remains exact."""
    patch = generated.resize((crop[2] - crop[0], crop[3] - crop[1]), Image.Resampling.LANCZOS)
    blend = mask
    if float(feather) > 0:
        blend = ImageChops.multiply(mask, mask.filter(ImageFilter.GaussianBlur(float(feather))))
    result = body.copy()
    result.paste(patch, crop[:2], blend.crop(crop))
    return result


def run_swap_ui(*args):
    result, path, status, used_seed = run_swap(*args)
    original, _, _ = prepare_selection(args[0])
    original = original.resize(result.size, Image.Resampling.LANCZOS)
    return (original, result), path, status, str(used_seed)


def model_profile() -> str:
    import os

    value = os.environ.get("FLOYD_HEADLINER_PROFILE")
    if value is None and PROFILE_FILE.is_file():
        value = PROFILE_FILE.read_text(encoding="utf-8").strip()
    value = value or "standard"
    if value not in {"standard", "low-vram", "low-vram-12gb"}:
        raise RuntimeError(f"Invalid model profile {value!r}.")
    return value


def selected_turbo_lora() -> Path:
    return LOW_TURBO_LORA if model_profile() != "standard" else TURBO_LORA


def missing_model_files() -> list[Path]:
    required = [BASE_MODEL_MARKER, BFS_LORA, selected_turbo_lora()]
    return [path for path in required if not path.is_file()]


def split_fused_mlp_lora(state_dict: dict):
    """Map Comfy's fused [gate; up] LoRA to Diffusers gate/proj modules."""
    converted = {}
    for key, value in state_dict.items():
        if ".img_mlp.gate_up.lora_A.weight" in key:
            converted[key.replace(".gate_up.", ".gate_layer.")] = value.clone()
            converted[key.replace(".gate_up.", ".proj.")] = value.clone()
        elif ".img_mlp.gate_up.lora_B.weight" in key:
            if value.shape[0] % 2:
                raise ValueError(f"Cannot split fused BFS LoRA tensor with shape {tuple(value.shape)}: {key}")
            gate, projection = value.chunk(2, dim=0)
            converted[key.replace(".gate_up.", ".gate_layer.")] = gate.contiguous()
            converted[key.replace(".gate_up.", ".proj.")] = projection.contiguous()
        else:
            converted[key] = value
    return converted


def global_free_vram_bytes(torch_module) -> int:
    """Prefer NVIDIA's global reading on Windows; torch may report per-context free VRAM."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        free_mib = int(output.splitlines()[0].strip())
        return free_mib * 1024**2
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        free_vram, _ = torch_module.cuda.mem_get_info(0)
        return int(free_vram)


def _load_pipeline():
    global _PIPELINE, _OFFLOAD_MODE, _RESIDENT_ALLOWED
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        missing = missing_model_files()
        if missing:
            names = "\n".join(f"- {path}" for path in missing)
            raise RuntimeError(
                "Required model files are missing. Run 1-INSTALL-Floyd-Headliner.bat first:\n" + names
            )

        import torch
        from diffusers import QwenImage21Pipeline
        from fast_pipeline import FastQwenImage21Pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This app requires a supported NVIDIA GPU.")

        low_vram = model_profile() != "standard"
        load_options = dict(dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
        if low_vram:
            from diffusers.quantizers import PipelineQuantizationConfig

            load_options["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
        pipe = FastQwenImage21Pipeline.from_pretrained(BASE_MODEL_DIR, **load_options)
        pipe.twelve_gb_mode = model_profile() == "low-vram-12gb"
        if pipe.twelve_gb_mode:
            # Quantized loading initially places both large components on CUDA.
            # Free the encoder before attaching the two transformer adapters.
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # Diffusers converts the Comfy/ai-toolkit prefix. Qwen's standalone
        # model keeps the fused [gate; up] MLP as two linear layers, so split
        # that adapter pair before PEFT attaches it.
        bfs_state, _ = QwenImage21Pipeline.lora_state_dict(
            BFS_LORA,
            return_lora_metadata=True,
        )
        bfs_state = split_fused_mlp_lora(bfs_state)
        pipe.load_lora_weights(bfs_state, adapter_name="bfs", low_cpu_mem_usage=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Already found a `peft_config` attribute.*",
                category=UserWarning,
            )
            pipe.load_lora_weights(selected_turbo_lora(), adapter_name="turbo", low_cpu_mem_usage=True)
        pipe.set_adapters(["bfs", "turbo"], adapter_weights=[1.0, 1.0])

        free_vram = global_free_vram_bytes(torch)
        if hasattr(pipe.vae, "enable_tiling"):
            if free_vram >= 20 * 1024**3:
                pipe.vae.enable_tiling(
                    tile_sample_min_height=512, tile_sample_min_width=512,
                    tile_sample_stride_height=448, tile_sample_stride_width=448,
                )
            else:
                pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        # Keep only the swap transformer/VAE resident on an otherwise free
        # 24 GB card. The larger text encoder always returns to system RAM.
        _RESIDENT_ALLOWED = not low_vram and free_vram >= 22 * 1024**3
        if low_vram:
            pipe.enable_model_cpu_offload(gpu_id=0)
            _OFFLOAD_MODE = "low-VRAM NF4 + component CPU offload"
            if pipe.twelve_gb_mode:
                _OFFLOAD_MODE = "experimental 12 GB: NF4 + CPU offload, 512px references"
        elif _RESIDENT_ALLOWED:
            pipe.enable_fast_residency()
            _OFFLOAD_MODE = "fast GPU residency"
        elif free_vram >= 20 * 1024**3:
            pipe.enable_model_cpu_offload(gpu_id=0)
            _OFFLOAD_MODE = "component CPU offload"
        else:
            pipe.enable_sequential_cpu_offload(gpu_id=0)
            _OFFLOAD_MODE = "sequential CPU offload"
        pipe.set_progress_bar_config(desc="Floyd Headliner")
        print(f"Offload mode: {_OFFLOAD_MODE} ({free_vram / 1024**3:.1f} GiB VRAM free at load)")
        _PIPELINE = pipe
        return _PIPELINE


def configure_speed_mode(pipe, keep_on_gpu, width, height):
    global _OFFLOAD_MODE
    # Higher resolution needs extra activation space; retain the offload path.
    use_residency = bool(keep_on_gpu) and _RESIDENT_ALLOWED and width * height <= 1.15 * 1_048_576
    if use_residency:
        pipe.enable_fast_residency()
        _OFFLOAD_MODE = "fast GPU residency"
    elif pipe.fast_resident:
        pipe.disable_fast_residency()
        pipe.enable_model_cpu_offload(gpu_id=0)
        _OFFLOAD_MODE = "component CPU offload"


def release_models():
    """Release in-memory models/caches only. Never remove any files."""
    global _PIPELINE, _OFFLOAD_MODE, _RESIDENT_ALLOWED
    import torch

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            _PIPELINE.disable_fast_residency()
            _PIPELINE.clear_conditioning_cache()
        _PIPELINE = None
        _RESIDENT_ALLOWED = False
        _OFFLOAD_MODE = "not loaded"
        gc.collect()
        torch.cuda.empty_cache()
    return "Models released from memory. Model files and saved images are unchanged. The next swap reloads the models."


def run_swap(
    body_image: Image.Image | None,
    head_image: Image.Image | None,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    bfs_strength: float,
    turbo_strength: float,
    steps: int,
    working_megapixels: float,
    output_upscale: int,
    keep_on_gpu: bool = True,
    extra_prompt: str = "",
    selected_only: bool = False,
    feather: float = 8,
):
    if body_image is None:
        raise ValueError("Add the body/base image (Image 1).")
    if head_image is None:
        raise ValueError("Add the head reference (Image 2).")
    if not prompt or not prompt.strip():
        raise ValueError("The edit prompt cannot be empty.")
    if "<image1>" not in prompt or "<image2>" not in prompt:
        raise ValueError("The head-swap instruction must keep both <image1> and <image2> references.")
    effective_prompt = compose_prompt(prompt, extra_prompt)

    import torch

    original, mask, crop = prepare_selection(body_image, selected_only)
    body = original.crop(crop) if crop else original
    head = ImageOps.exif_transpose(head_image).convert("RGB")
    width, height = calculate_working_size(body, working_megapixels)
    used_seed = random.randint(0, 2**63 - 1) if randomize_seed else int(seed)

    with _PIPELINE_LOCK:
        started = time.perf_counter()
        pipe = _load_pipeline()
        configure_speed_mode(pipe, keep_on_gpu, width, height)
        pipe.set_adapters(
            ["bfs", "turbo"],
            adapter_weights=[float(bfs_strength), float(turbo_strength)],
        )
        generator = torch.Generator(device="cuda").manual_seed(used_seed)
        result = pipe(
            prompt=effective_prompt,
            image=[body, head],
            width=width,
            height=height,
            num_inference_steps=int(steps),
            true_cfg_scale=1.0,
            generator=generator,
        ).images[0]

    if selected_only:
        result = composite_selection(original, result, mask, crop, feather)
    elif int(output_upscale) > 1:
        result = result.resize(
            (result.width * int(output_upscale), result.height * int(output_upscale)),
            Image.Resampling.LANCZOS,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = OUTPUT_DIR / f"bfs-swap-{stamp}-seed-{used_seed}.png"
    result.save(output_path, format="PNG")
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started

    status = (
        f"Saved `{output_path}`  \n"
        f"Seed: `{used_seed}` | Working canvas: `{width} x {height}` | "
        f"Output: `{result.width} x {result.height}`  \n"
        f"Time: `{elapsed:.1f} s` | {_OFFLOAD_MODE} | "
        f"Reference/prompt cache: `{'reused' if pipe.prompt_cache_hit else 'encoded'}`"
    )
    if selected_only:
        status += "  \nPainted-area edit: original dimensions retained; unpainted pixels unchanged."
    return result, str(output_path), status, used_seed


def build_likeness_ui(gr):
    gr.Markdown(
        "**1. BODY REFERENCE:** keeps pose, clothing, lighting, and background.  \n"
        "**2. HEAD REFERENCE:** supplies the face, head, hair, and identity. Both images are required.",
        elem_classes="ggf-guide",
    )
    selected_only = gr.Checkbox(label="Edit painted area only", value=False)
    with gr.Row():
        body = gr.ImageEditor(
            type="pil", format="png", image_mode="RGBA", transforms=(), layers=False,
            brush=gr.Brush(colors=["#f5b942"], color_mode="fixed"), height=420,
            label="1 — BODY REFERENCE (required): pose, clothes, scene",
        )
        head = gr.Image(type="pil", height=420, label="2 — HEAD REFERENCE (required): face, hair, identity")
    extra_prompt = gr.Textbox(
        label="Extra prompt", value="", lines=2,
        placeholder="For example: remove the hat; keep the head reference's hair.",
        info="Added to the end of the likeness instruction in Advanced settings.",
    )
    generate = gr.Button("Transfer likeness · Ctrl+Enter", variant="primary", elem_id="ggf-swap-button")
    output = gr.ImageSlider(
        type="pil", format="png", interactive=False,
        label="BEFORE / AFTER — drag the divider (original left, result right)", buttons=["fullscreen"],
    )
    gr.Markdown(
        "For multiple people, paint over **only the intended head**, including hair or hats to remove. "
        "Painting enables **Edit painted area only** automatically; clearing the paint turns it off. "
        "The app edits a crop and blends it back; unpainted pixels stay unchanged. "
        "Paint enough room for the replacement hair. Uncheck the box to edit the whole image.",
        elem_classes="ggf-guide",
    )
    feather = gr.Slider(0, 32, value=8, step=1, label="Selection edge softness (original-image pixels)")
    body.change(
        fn=lambda raw: painted_mask_from_editor_payload(body, raw),
        inputs=[body], outputs=[selected_only], preprocess=False,
        trigger_mode="always_last", show_progress="hidden",
    )
    with gr.Accordion("Advanced likeness settings", open=False):
        prompt = gr.Textbox(label="Likeness instruction (keep <image1> and <image2>)", value=DEFAULT_PROMPT, lines=5)
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            randomize = gr.Checkbox(label="Randomize seed", value=True)
            steps = gr.Slider(4, 40, value=6, step=1, label="Steps")
        with gr.Row():
            bfs_strength = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="BFS identity LoRA strength")
            turbo_strength = gr.Slider(0.0, 1.25, value=1.0, step=0.05, label="Viggle turbo LoRA strength")
            small = model_profile() == "low-vram-12gb"
            working_mp = gr.Slider(0.25 if small else 0.5, 0.5 if small else 2.0,
                                   value=0.5 if small else 1.0, step=0.05 if small else 0.1,
                                   label="Working megapixels (12 GB test)" if small else "Working megapixels")
            upscale = gr.Radio([1, 2], value=2, label="Lanczos output upscale")
        if model_profile() != "standard":
            keep_on_gpu = gr.Checkbox(
                False,
                label="Low-VRAM mode: automatic component offload (GPU residency disabled)",
                interactive=False,
            )
        else:
            keep_on_gpu = gr.Checkbox(True, label="Keep model on GPU for faster repeat transfers (24 GB GPU, up to about 1 MP)")
        gr.Markdown("Unchanged references and prompt are cached in RAM. Changing either image or the prompt "
                    "automatically re-encodes it. Fast mode retains GPU memory; release models before using another GPU app.")
    status = gr.Markdown(elem_classes="ggf-status")
    saved_file = gr.File(label="Saved PNG", elem_classes="ggf-download")
    used_seed = gr.Textbox(label="Used seed", interactive=False)
    send_to_inpaint = gr.Button("Open result in Inpaint")
    release = gr.Button("Release models / free GPU memory")
    release.click(fn=release_models, inputs=[], outputs=[status], concurrency_id="gpu")
    generate.click(
        fn=run_swap_ui,
        inputs=[body, head, prompt, seed, randomize, bfs_strength, turbo_strength, steps,
                working_mp, upscale, keep_on_gpu, extra_prompt, selected_only, feather],
        outputs=[output, saved_file, status, used_seed], concurrency_id="gpu",
    )
    return output, send_to_inpaint


def build_ui():
    import gradio as gr
    from inpaint import build_inpaint_ui, image_data_url

    with gr.Blocks(title="Floyd Headliner · Get Going Fast") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                '<header class="ggf-hero">'
                '<div class="ggf-kicker">GET GOING FAST · LOCAL AI</div>'
                '<h1>Floyd Headliner</h1>'
                '<p>Likeness Transfer &amp; Inpaint · Draw your next idea.</p>'
                '<nav class="ggf-links" aria-label="Get Going Fast links">'
                '<a href="https://getgoingfast.pro" target="_blank" rel="noopener noreferrer">GetGoingFast.pro ↗</a>'
                '<a href="https://www.youtube.com/@theaihobbyguy" target="_blank" rel="noopener noreferrer">TheAIHobbyGuy on YouTube ↗</a>'
                '</nav></header>'
            )
            gr.Markdown("Use only images you have the right and consent to edit. Do not use the app for impersonation or deception.",
                        elem_classes="warning")
            with gr.Tabs() as tabs:
                with gr.Tab("Likeness Transfer", id="likeness"):
                    likeness_output, send_to_inpaint = build_likeness_ui(gr)
                with gr.Tab("Inpaint", id="inpaint"):
                    inpaint_canvas = build_inpaint_ui(gr, sys.modules[__name__])

            def open_inpaint(pair):
                if pair is None or pair[1] is None:
                    raise ValueError("Generate a likeness result first.")
                return {"load_image": image_data_url(pair[1])}, gr.update(selected="inpaint")

            send_to_inpaint.click(open_inpaint, inputs=[likeness_output], outputs=[inpaint_canvas, tabs], api_name=False)
            gr.HTML(
                '<footer class="ggf-footer">Runs on your computer by default. Interface packaged by '
                '<a href="https://getgoingfast.pro" target="_blank" rel="noopener noreferrer">Get Going Fast</a>. '
                'Qwen Image 2.1, BFS, and Viggle Turbo remain third-party models.</footer>'
            )
    return demo


def self_check() -> int:
    import torch
    import diffusers
    import gradio
    import transformers
    from diffusers import QwenImage21Pipeline
    from fast_pipeline import FastQwenImage21Pipeline

    assert QwenImage21Pipeline is not None
    assert issubclass(FastQwenImage21Pipeline, QwenImage21Pipeline)
    assert tuple(map(int, diffusers.__version__.split(".")[:2])) >= (0, 41)
    print(f"Python app: OK")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model profile: {model_profile()}")
    print(f"Diffusers: {diffusers.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Gradio: {gradio.__version__}")
    if model_profile() != "standard":
        import bitsandbytes

        print(f"BitsAndBytes: {bitsandbytes.__version__}")
    missing = missing_model_files()
    if missing:
        print("Model install: INCOMPLETE")
        for path in missing:
            print(f"  Missing: {path}")
        return 2
    print("Model install: OK")
    return 0


def existing_local_app(port: int) -> bool:
    """Recognize our own Gradio server before trying to bind its port again."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=1.5) as response:
            config = json.load(response)
        return "Floyd Headliner" in config.get("title", "")
    except (OSError, ValueError, urllib.error.URLError):
        return False


def local_port_in_use(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Floyd Headliner")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    if args.host in ("127.0.0.1", "localhost"):
        ports = (args.port,) if args.port is not None else range(7860, 7870)
        for port in ports:
            if existing_local_app(port):
                url = f"http://127.0.0.1:{port}"
                print(f"Floyd Headliner is already running at {url}")
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
        if args.port is not None and local_port_in_use(args.port):
            print(f"Port {args.port} is in use by another service. Start without --port to choose a free port.")
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        show_error=True,
        css=APP_CSS,
        js=APP_JS,
        allowed_paths=[str(OUTPUT_DIR)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
