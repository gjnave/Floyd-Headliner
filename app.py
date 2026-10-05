from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import random
import re
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
from network_settings import read_settings, save_settings, verify_login


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "Qwen-Image-2.1"
LORA_DIR = MODELS_DIR / "loras"
OUTPUT_DIR = APP_DIR / "outputs"
ACTIVE_MODE = "local"
ACTIVE_URL = ""
ACTIVE_AUTH_ENABLED = False

FAST_CORE_LABEL = "Fast core (app-local; experimental)"
STANDALONE_LABEL = "Standalone (Diffusers)"

BASE_MODEL_MARKER = BASE_MODEL_DIR / "model_index.json"
BFS_LORA = LORA_DIR / "bfs_head_v1.1_alternative_qwen_2.1.safetensors"
TURBO_LORA = LORA_DIR / "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
LOW_TURBO_LORA = LORA_DIR / "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
PROFILE_FILE = MODELS_DIR / "install-profile.txt"
VERSION_FILE = APP_DIR / "VERSION"
UPDATE_VERSION_URL = "https://codeberg.org/Cognibuild/Floyd-Headliner/raw/branch/main/VERSION"

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
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, .8fr) 150px;
    align-items: center;
    gap: 20px;
}
.ggf-hero .ggf-kicker {
    color: var(--ggf-gold);
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .14em;
}
.ggf-hero h1 { color: var(--ggf-gold); margin: 7px 0; font-size: clamp(1.8rem, 3vw, 2.6rem); }
.ggf-hero p { color: #dbeafe; margin: 0 0 8px; }
.ggf-profile {
    display: inline-block;
    margin: 3px 0 12px;
    padding: 5px 10px;
    border: 1px solid var(--ggf-gold);
    border-radius: 999px;
    background: #342b16;
    color: #ffe4a3;
    font-size: .84rem;
    font-weight: 800;
}
.ggf-hero a, .ggf-footer a { color: #8fc2ff; font-weight: 700; }
.ggf-hero .ggf-links { display: flex; gap: 10px 18px; flex-wrap: wrap; }
.ggf-hero-slogan {
    color: #f4f7fc;
    font-family: "Segoe UI Variable Display", "Segoe UI", Arial, sans-serif;
    font-size: clamp(1.3rem, 2.2vw, 1.9rem);
    font-weight: 750;
    letter-spacing: -.035em;
    line-height: 1.18;
    text-align: center;
}
.ggf-hero-slogan strong { color: var(--ggf-gold); font-weight: 800; }
.ggf-hero-logo { display: block; width: 150px; height: 150px; object-fit: cover; }
.ggf-update-row { align-items: center; }
.ggf-update-status { color: var(--ggf-muted); }
#ggf-check-updates { background: #172842 !important; color: var(--ggf-ink) !important; border: 1px solid #45607f !important; }
#ggf-update-button[disabled] { background: #172842 !important; color: #8fa2bd !important; border: 1px solid #304763 !important; }
#ggf-update-button:not([disabled]) { background: var(--ggf-gold) !important; color: #111827 !important; font-weight: 800 !important; animation: ggf-update-glow 1.7s ease-in-out infinite alternate; }
@keyframes ggf-update-glow { from { box-shadow: 0 0 5px rgba(245,185,66,.45); } to { box-shadow: 0 0 20px rgba(245,185,66,1); } }
.ggf-image-lightbox { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center; padding: 18px; background: rgba(2, 7, 17, .94); cursor: zoom-out; }
.ggf-image-lightbox img { display: block; max-width: 96vw; max-height: 96vh; object-fit: contain; box-shadow: 0 18px 70px rgba(0, 0, 0, .7); cursor: zoom-out; }
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
/* Gradio 6's dropdown menu otherwise keeps its white Base-theme surface. */
#ggf-network-mode .options {
    background: var(--ggf-panel) !important;
    color: var(--ggf-ink) !important;
    border: 1px solid var(--ggf-line) !important;
}
#ggf-network-mode .options .item {
    color: var(--ggf-ink) !important;
}
#ggf-network-mode .options .item:hover,
#ggf-network-mode .options .item.active {
    background: #172a43 !important;
}
#ggf-network-password input[type="password"] {
    background: #091425 !important;
    color: var(--ggf-ink) !important;
    caret-color: var(--ggf-gold);
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
#ggf-swap-turbo, #ggf-inpaint-turbo { background: #172842 !important; color: var(--ggf-gold) !important; border: 1px solid var(--ggf-gold) !important; font-weight: 700 !important; }
.tab-nav button { color: var(--ggf-ink) !important; }
.tab-nav button.selected { color: var(--ggf-gold) !important; }
.ggf-footer { color: var(--ggf-muted); text-align: center; margin-top: 20px; font-size: .88rem; }
@media (max-width: 900px) { .ggf-hero { grid-template-columns: minmax(0, 1fr) 110px; } .ggf-hero-slogan { grid-column: 1 / -1; grid-row: 2; } .ggf-hero-logo { width: 110px; height: 110px; } }
@media (max-width: 680px) {
    /* Gradio adds 32px outer gutters; reclaim them inside our own column. */
    .app-shell {
        padding: 8px 6px 24px;
        margin-left: -20px !important;
        margin-right: -20px !important;
        width: calc(100% + 40px) !important;
    }
    #ggf-hero-block .html-container { padding: 0 !important; }
    .ggf-hero {
        display: block;
        padding: 14px 16px;
        margin-bottom: 10px;
        border-radius: 14px;
    }
    .ggf-hero .ggf-kicker { font-size: .68rem; letter-spacing: .1em; }
    .ggf-hero h1 { font-size: clamp(1.55rem, 7vw, 2rem); line-height: 1.1; margin: 5px 0; }
    .ggf-hero p { font-size: .9rem; margin-bottom: 5px; }
    .ggf-profile { font-size: .76rem; padding: 3px 8px; margin: 4px 0 8px; }
    .ggf-hero .ggf-links { gap: 4px 14px; }
    .ggf-hero .ggf-links a { font-size: .82rem; overflow-wrap: anywhere; }
    .ggf-hero-slogan, .ggf-hero-logo { display: none; }
    .ggf-update-row { display: flex; flex-wrap: nowrap; gap: 8px; }
    .ggf-update-row > * { min-width: 0 !important; flex: 1 1 0 !important; }
    .ggf-update-row button { min-width: 0 !important; width: 100%; font-size: .8rem; }
    .ggf-update-status { font-size: .8rem; margin: 0 0 8px; }
    #ggf-reference-row { flex-direction: column; }
    #ggf-reference-row > * {
        width: 100% !important;
        min-width: 0 !important;
        flex: 0 0 auto !important;
    }
}
"""
APP_JS = """
() => {
    const outputSelector = '#ggf-likeness-output, #ggf-inpaint-output';
    let outputPress = null;
    let lightbox = null;
    const closeLightbox = () => {
        lightbox?.remove();
        lightbox = null;
    };
    document.addEventListener('pointerdown', (event) => {
        const root = event.target.closest?.(outputSelector);
        outputPress = event.button === 0 && root && event.target.closest?.('.image-container') &&
            !event.target.closest?.('button, [data-testid="slider"]')
            ? { root, x: event.clientX, y: event.clientY } : null;
    }, true);
    document.addEventListener('pointerup', (event) => {
        const press = outputPress;
        outputPress = null;
        if (!press || lightbox || event.button !== 0 ||
            event.target.closest?.(outputSelector) !== press.root ||
            Math.hypot(event.clientX - press.x, event.clientY - press.y) > 8) return;
        const images = press.root.querySelectorAll('img[data-testid="imageslider-image"]');
        const result = images[images.length - 1];
        if (!result?.src) return;
        lightbox = document.createElement('div');
        lightbox.className = 'ggf-image-lightbox';
        lightbox.setAttribute('role', 'dialog');
        lightbox.setAttribute('aria-label', 'Generated image enlarged; click to close');
        const enlarged = document.createElement('img');
        enlarged.src = result.currentSrc || result.src;
        enlarged.alt = 'Generated image';
        enlarged.draggable = false;
        lightbox.append(enlarged);
        lightbox.addEventListener('click', closeLightbox);
        document.body.append(lightbox);
    }, true);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeLightbox();
    }, true);
    const imageInputIds = ['ggf-body-reference', 'ggf-head-reference', 'ggf-inpaint-source'];
    const imageInputSelector = imageInputIds.map(id => `#${id}`).join(', ');
    let pasteTargetId = null;
    document.addEventListener('pointerdown', (event) => {
        pasteTargetId = event.target.closest?.(imageInputSelector)?.id || null;
    }, true);
    document.addEventListener('paste', (event) => {
        if (event.target.closest?.('textarea, input:not([type="file"]), [contenteditable="true"]')) return;
        const targetId = event.target.closest?.(imageInputSelector)?.id || pasteTargetId;
        const target = targetId && document.getElementById(targetId);
        if (!target || target.getClientRects().length === 0) return;
        const item = Array.from(event.clipboardData?.items || [])
            .find(item => item.kind === 'file' && item.type.startsWith('image/'));
        const image = item?.getAsFile();
        const input = target.querySelector('input[type="file"]');
        if (!image || !input) return;
        const subtype = image.type.split('/')[1]?.split('+')[0] || 'png';
        const extension = subtype === 'jpeg' ? 'jpg' : subtype;
        const transfer = new DataTransfer();
        transfer.items.add(new File([image], image.name || `pasted-image.${extension}`,
                                    { type: image.type || 'image/png' }));
        input.files = transfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
        event.preventDefault();
        event.stopImmediatePropagation();
    }, true);
    // Gradio's occupied image surface can consume a drop as a clear action.
    // Route the dropped file through its existing upload input instead.
    document.addEventListener('dragover', (event) => {
        if (!event.target.closest?.('#ggf-head-reference')) return;
        if (!Array.from(event.dataTransfer?.items || []).some(item => item.kind === 'file')) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
    }, true);
    document.addEventListener('drop', (event) => {
        const head = event.target.closest?.('#ggf-head-reference');
        if (!head) return;
        const file = Array.from(event.dataTransfer?.files || []).find(file => file.type.startsWith('image/'));
        if (!file) return;
        const input = head.querySelector('input[type="file"]');
        if (!input) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }, true);
    document.addEventListener('keydown', (event) => {
        if (!event.ctrlKey || event.key !== 'Enter' || event.repeat) return;
        const button = ['ggf-inpaint-turbo', 'ggf-swap-turbo',
                        'ggf-inpaint-button', 'ggf-swap-button']
            .map(id => document.getElementById(id))
            .find(button => button && !button.disabled && button.getClientRects().length > 0);
        if (!button) return;
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
    minimum = 0.25 if model_profile() != "standard" else 0.125
    target_area = max(minimum, float(megapixels)) * 1_048_576
    ratio = width / height
    scaled_width = max(32, round((target_area * ratio) ** 0.5 / 32) * 32)
    scaled_height = max(32, round((target_area / ratio) ** 0.5 / 32) * 32)
    if model_profile() == "low-vram-12gb":
        scaled_width = max(32, int((target_area * ratio) ** 0.5 / 32) * 32)
        scaled_height = max(32, int((target_area / ratio) ** 0.5 / 32) * 32)
    return scaled_width, scaled_height


def turbo_megapixels(megapixels: float) -> float:
    """Quarter the pixel budget to halve each working dimension."""
    return max(0.125, float(megapixels) / 4)


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


def protect_editor_preprocessing(editor):
    """Retry incomplete cache reads at the component boundary, including Generate."""
    original_preprocess = editor.preprocess

    def preprocess(payload):
        for attempt in range(5):
            try:
                from upload_safety import validate_component_images
                validate_component_images(payload)
                value = original_preprocess(payload)
                if payload is not None and (not value or value.get("background") is None):
                    raise ValueError(
                        "The body image upload is not ready or has expired. "
                        "Upload it again, wait for its preview, then generate."
                    )
                return value
            except ValueError as error:
                import gradio as gr
                raise gr.Error(str(error)) from error
            except (OSError, SyntaxError) as error:
                if attempt == 4:
                    import gradio as gr
                    raise gr.Error(
                        "The body image or painted mask could not be read completely. "
                        "Wait for the upload to finish and try again. If it repeats, "
                        "re-upload the original body image and repaint the selection. "
                        "Generation has not started; no models need reinstalling."
                    ) from error
                time.sleep(0.15 * 2**attempt)

    editor.preprocess = preprocess
    return editor


def protect_head_preprocessing(component):
    """Reject an oversized cached head image before Gradio decodes it."""
    original_preprocess = component.preprocess

    def preprocess(payload):
        from upload_safety import validate_component_images
        import gradio as gr

        try:
            validate_component_images(payload)
            return original_preprocess(payload)
        except (ValueError, Image.DecompressionBombError) as error:
            raise gr.Error(str(error)) from error

    component.preprocess = preprocess
    return component


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


def protected_reference(original, painted):
    """Hide painted heads before conditioning; return the exact restoration mask."""
    if painted is None or painted.getbbox() is None:
        raise ValueError("Paint over the whole head you want to protect, including hair.")
    protected = painted.filter(ImageFilter.MaxFilter(17))
    reference = original.copy()
    reference.paste((127, 127, 127), mask=protected)
    return reference, protected


def unprotected_side_crop(protected):
    """Isolate the other side when a protected person is clearly at one edge."""
    width, height = protected.size
    left, _, right, _ = protected.getbbox()
    gap = max(8, round(width * .02))
    left_end = max(0, left - gap)
    right_start = min(width, right + gap)
    minimum = max(64, round(width * .35))
    if width - right_start >= minimum and width - right_start >= left_end * 1.25:
        return (right_start, 0, width, height)
    if left_end >= minimum and left_end >= (width - right_start) * 1.25:
        return (0, 0, left_end, height)
    return None


def run_swap_ui(*args, turbo_preview: bool = False):
    result, path, status, used_seed = run_swap(*args, turbo_preview=turbo_preview)
    original, _, _ = prepare_selection(args[0])
    original = original.resize(result.size, Image.Resampling.LANCZOS)
    return (original, result), path, status, str(used_seed)


def result_as_body_input(result):
    """Reuse the session's last result without carrying a painted mask forward."""
    if result is None:
        raise ValueError("Generate a likeness result first.")
    return {"background": result.copy(), "layers": [], "composite": result.copy()}, False


def retain_uploaded_head(image):
    """Reassert a completed upload after Gradio swaps an occupied image surface."""
    return image.copy() if image is not None else None


def model_profile() -> str:
    import os

    value = os.environ.get("FLOYD_HEADLINER_PROFILE")
    if value is None and PROFILE_FILE.is_file():
        value = PROFILE_FILE.read_text(encoding="utf-8").strip()
    value = value or "standard"
    if value not in {"standard", "low-vram", "low-vram-12gb"}:
        raise RuntimeError(f"Invalid model profile {value!r}.")
    return value


def running_profile_label() -> str:
    return {
        "standard": "Standard · 24 GB",
        "low-vram": "Low VRAM · 16 GB target",
        "low-vram-12gb": "Experimental · 12 GB target",
    }[model_profile()]


def hero_logo_html() -> str:
    logo = APP_DIR / "assets" / "ggf-brain-logo.png"
    if not logo.is_file():
        return ""
    encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
    return f'<img class="ggf-hero-logo" src="data:image/png;base64,{encoded}" alt="Get Going Fast logo">'


def parse_release_version(value: str) -> tuple[int, int, int, int]:
    value = value.strip()
    if not re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}\.\d+", value):
        raise ValueError("Invalid Floyd Headliner version marker.")
    return tuple(int(part) for part in value.split("."))


def check_for_updates() -> tuple[str, bool]:
    """Read only: compare the installed marker with Codeberg's release marker."""
    request = urllib.request.Request(
        UPDATE_VERSION_URL,
        headers={"User-Agent": "Floyd-Headliner-Update-Check", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            latest = response.read(64).decode("utf-8").strip()
        latest_number = parse_release_version(latest)
    except (OSError, TimeoutError, UnicodeError, ValueError) as exc:
        return f"Could not check Codeberg for updates ({type(exc).__name__}). The app still works offline.", False
    try:
        installed = VERSION_FILE.read_text(encoding="utf-8").strip()
        installed_number = parse_release_version(installed)
    except (OSError, ValueError):
        return (f"Update available ({latest}). This installation has no version marker; "
                "close the app and run 3-UPDATE-Floyd-Headliner.bat."), True
    if latest_number > installed_number:
        return (f"Update available: {installed} → {latest}. "
                "Close the app and run 3-UPDATE-Floyd-Headliner.bat."), True
    return f"Up to date (installed {installed}).", False


def selected_turbo_lora() -> Path:
    return LOW_TURBO_LORA if model_profile() != "standard" else TURBO_LORA


def available_startup_profiles() -> list[str]:
    """Inspect installed weights without downloading or loading them into RAM."""
    def present(path):
        return path.is_file() and path.stat().st_size > 0

    if not present(BASE_MODEL_MARKER) or not present(BFS_LORA):
        return []
    for component in ("transformer", "text_encoder", "vae"):
        folder = BASE_MODEL_DIR / component
        indexes = list(folder.glob("*.safetensors.index.json"))
        if indexes:
            try:
                for index in indexes:
                    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    if not shards or not all(present(folder / shard) for shard in shards):
                        return []
            except (OSError, ValueError, KeyError, TypeError):
                return []
        elif not any(present(path) for path in folder.glob("*.safetensors")):
            return []
    profiles = []
    if present(TURBO_LORA):
        profiles.append("standard")
    if present(LOW_TURBO_LORA):
        profiles.extend(("low-vram", "low-vram-12gb"))
    return profiles


def select_startup_profile(requested=None) -> bool:
    profiles = available_startup_profiles()
    labels = {"standard": "Standard (24 GB)", "low-vram": "Low VRAM (16 GB target)",
              "low-vram-12gb": "Experimental 12 GB (reduced detail)"}
    if not profiles:
        print("No complete model set found. Run the installer to finish the model downloads.")
        return False
    try:
        current = model_profile()
    except RuntimeError:
        current = "standard"
    selected = requested or os.environ.get("FLOYD_HEADLINER_PROFILE")
    if selected is not None and selected not in profiles:
        print(f"Models for {selected!r} are not installed. Run the installer for that mode first.")
        return False
    if selected is None:
        default = current if current in profiles else profiles[0]
        # Options 2/3 are two runtime modes of ONE downloaded model variant.
        if "standard" not in profiles or len(profiles) == 1:
            selected = default
        elif not sys.stdin.isatty():
            selected = default
        else:
            print("\nInstalled model modes (no downloads or reinstall required):")
            for number, profile in enumerate(profiles, 1):
                print(f"  {number}. {labels[profile]}" + (" [last selected]" if profile == default else ""))
            print("Close this app before starting it again to change modes.")
            while selected is None:
                try:
                    answer = input(f"Choose 1-{len(profiles)}; Enter keeps {labels[default]}: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nStartup cancelled.")
                    return False
                if not answer:
                    selected = default
                elif answer.isdigit() and 1 <= int(answer) <= len(profiles):
                    selected = profiles[int(answer) - 1]
                else:
                    print("Enter one of the listed numbers.")
    if selected != "standard":
        import importlib.util
        if importlib.util.find_spec("bitsandbytes") is None:
            print("Low-VRAM models are present but bitsandbytes is missing. Run the installer once for this mode.")
            return False
    PROFILE_FILE.write_text(selected + "\n", encoding="utf-8")
    os.environ["FLOYD_HEADLINER_PROFILE"] = selected
    print(f"Starting Floyd Headliner: {labels[selected]}")
    return True


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


def release_standalone_models():
    """Release the original in-memory pipeline without removing files."""
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


def release_core_models():
    from core_runtime import release_models as release_fast_core

    return release_fast_core()


def release_models():
    """Release both inference engines without removing model or output files."""
    release_core_models()
    return release_standalone_models()


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
    mask_mode: str = "Edit painted area",
    backend: str = STANDALONE_LABEL,
    turbo_preview: bool = False,
):
    if backend == FAST_CORE_LABEL:
        from core_runtime import run_swap as run_fast_core

        return run_fast_core(
            sys.modules[__name__], body_image, head_image, prompt, seed,
            randomize_seed, bfs_strength, turbo_strength, steps,
            working_megapixels, output_upscale, keep_on_gpu, extra_prompt,
            selected_only, feather, mask_mode, turbo_preview,
        )
    if backend != STANDALONE_LABEL:
        raise ValueError("Choose a valid likeness-transfer engine.")
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

    # A stray mobile touch can paint a tiny mark. Only an explicitly enabled
    # mask may constrain generation; erasing it disables mask mode.
    selected_only = bool(selected_only and has_painted_mask(body_image))
    original, mask, crop = prepare_selection(body_image, selected_only)
    if mask_mode not in ("Edit painted area", "Protect painted area (experimental)"):
        raise ValueError("Choose a valid painted-area mode.")
    if mask_mode == "Protect painted area (experimental)" and not selected_only:
        raise ValueError("Paint over the head to protect before generating.")
    protect = selected_only and mask_mode == "Protect painted area (experimental)"
    if protect:
        body, protected = protected_reference(original, mask)
        side_crop = unprotected_side_crop(protected)
        if side_crop:
            crop = side_crop
            body = original.crop(crop)
        else:
            crop = (0, 0, original.width, original.height)
        mask = ImageOps.invert(protected)
        if mask.crop(crop).getbbox() is None:
            raise ValueError("Leave the target head unpainted; the entire working area is protected.")
        if side_crop:
            effective_prompt += ("\n<image1> is the unprotected person cropped from the group photo. "
                                 "Transfer the reference likeness to that visible person's head. "
                                 "Keep their pose, clothing, and surroundings.")
        else:
            effective_prompt += ("\nThe flat gray covered regions in <image1> hide protected heads. "
                                 "Transfer the reference likeness only to the remaining visible head. "
                                 "Do not create a new head in the covered regions.")
    else:
        body = original.crop(crop) if crop else original
    head = ImageOps.exif_transpose(head_image).convert("RGB")
    width, height = calculate_working_size(body, working_megapixels)
    used_seed = random.randint(0, 2**63 - 1) if randomize_seed else int(seed)

    with _PIPELINE_LOCK:
        started = time.perf_counter()
        pipe = _load_pipeline()
        configure_speed_mode(pipe, keep_on_gpu, width, height)
        pipe.enable_lora()
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
        if turbo_preview:
            result = result.resize(calculate_working_size(original, working_megapixels), Image.Resampling.LANCZOS)
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
    if protect:
        status += ("  \nProtect experiment: " +
                   ("model focused on the unprotected side; " if side_crop else "painted heads hidden from the model; ") +
                   "protected pixels restored from the original.")
    elif selected_only:
        status += ("  \nPainted-area edit: mask restored before half-size preview resampling."
                   if turbo_preview else
                   "  \nPainted-area edit: original dimensions retained; unpainted pixels unchanged.")
    return result, str(output_path), status, used_seed


def build_likeness_ui(gr):
    gr.Markdown(
        "**1. BODY REFERENCE:** keeps pose, clothing, lighting, and background.  \n"
        "**2. HEAD REFERENCE:** supplies the face, head, hair, and identity. Both images are required.",
        elem_classes="ggf-guide",
    )
    selected_only = gr.Checkbox(label="Use painted mask (optional; check after painting)", value=False)
    mask_mode = gr.Radio(["Edit painted area", "Protect painted area (experimental)"],
                        value="Edit painted area", label="Painted area",
                        info="Protect: cover the entire head you want kept, including hair. For two people, the model focuses on the opposite side.")
    with gr.Row(elem_id="ggf-reference-row"):
        body = gr.ImageEditor(
            type="pil", format="png", image_mode="RGBA", transforms=(), layers=False,
            brush=gr.Brush(colors=["#f5b942"], color_mode="fixed"), height=420,
            label="1 — BODY REFERENCE (required): pose, clothes, scene",
            elem_id="ggf-body-reference",
        )
        head = gr.Image(type="pil", height=420, label="2 — HEAD REFERENCE (required): face, hair, identity",
                        elem_id="ggf-head-reference")
    protect_editor_preprocessing(body)
    protect_head_preprocessing(head)
    body.upload(lambda: False, outputs=[selected_only], queue=False,
                show_progress="hidden", api_name=False)
    body.clear(lambda: False, outputs=[selected_only], queue=False,
               show_progress="hidden", api_name=False)
    head.upload(retain_uploaded_head, inputs=[head], outputs=[head],
                queue=False, show_progress="hidden", api_name=False)
    extra_prompt = gr.Textbox(
        label="Extra prompt — appended to Advanced likeness instruction", value="", lines=2,
        placeholder="For example: remove the hat; keep the head reference's hair.",
    )
    turbo_generate = gr.Button("Turbo preview · half-size · Ctrl+Enter", variant="secondary",
                               elem_id="ggf-swap-turbo")
    generate = gr.Button("Transfer likeness", variant="primary", elem_id="ggf-swap-button")
    output = gr.ImageSlider(
        type="pil", format="png", interactive=False,
        label="BEFORE / AFTER — drag the divider (original left, result right)", buttons=["fullscreen"],
        elem_id="ggf-likeness-output",
    )
    last_result = gr.State(None)
    with gr.Row():
        reuse_result = gr.Button("Use result as new body reference", interactive=False)
        send_to_inpaint = gr.Button("Edit result in Inpaint", interactive=False)
    reuse_result.click(fn=lambda: None, outputs=[body], queue=False, api_name=False).then(
        fn=result_as_body_input, inputs=[last_result], outputs=[body, selected_only],
        api_name=False,
    )
    gr.Markdown(
        "**Edit painted area:** paint the intended head; only that area changes. "
        "**Protect painted area (experimental):** paint the heads to keep. When they are on one side, "
        "the model works on the other side; protected pixels are restored. Everything else can change. "
        "Leave one target head visible. Check 'Use painted mask' after painting; "
        "stray touches are ignored while it is unchecked. Empty masks are ignored when generating.",
        elem_classes="ggf-guide",
    )
    feather = gr.Slider(0, 32, value=8, step=1, label="Selection edge softness (original-image pixels)")
    # Do not read the editor from a change listener. That uploads the entire
    # background/composite again on each edit, and preprocess() consumes Gradio's
    # one-use blob cache. Validate masks only in the generation request instead.
    with gr.Accordion("Advanced likeness settings", open=False):
        prompt = gr.Textbox(label="Likeness instruction (keep <image1> and <image2>)", value=DEFAULT_PROMPT, lines=5)
        from core_runtime import available as fast_core_available

        fast_ready = fast_core_available() and model_profile() == "standard"
        backend = gr.Radio(
            [FAST_CORE_LABEL, STANDALONE_LABEL] if fast_ready else [STANDALONE_LABEL],
            value=FAST_CORE_LABEL if fast_ready else STANDALONE_LABEL,
            label="Likeness-transfer engine",
            info="Fast core uses bundled inference modules and app-local model files; "
                 "no ComfyUI installation or server. Standard-profile Inpaint also uses this core.",
        )
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
    release = gr.Button("Release models / free GPU memory")
    release.click(fn=release_models, inputs=[], outputs=[status], concurrency_id="gpu")
    def generate_transfer(*args):
        result = run_swap_ui(*args)
        return (*result, result[0][1], gr.update(interactive=True), gr.update(interactive=True))

    def generate_turbo_transfer(*args):
        settings = list(args)
        settings[8] = turbo_megapixels(settings[8])
        settings[9] = 1
        result = run_swap_ui(*settings, turbo_preview=True)
        return (*result, result[0][1], gr.update(interactive=True), gr.update(interactive=True))

    transfer_inputs = [body, head, prompt, seed, randomize, bfs_strength, turbo_strength, steps,
                       working_mp, upscale, keep_on_gpu, extra_prompt, selected_only, feather,
                       mask_mode, backend]
    transfer_outputs = [output, saved_file, status, used_seed, last_result, reuse_result, send_to_inpaint]
    generate.click(fn=generate_transfer, inputs=transfer_inputs, outputs=transfer_outputs,
                   concurrency_id="gpu")
    turbo_generate.click(fn=generate_turbo_transfer, inputs=transfer_inputs,
                         outputs=transfer_outputs, concurrency_id="gpu", api_name="turbo_transfer")
    return body, selected_only, last_result, send_to_inpaint


def build_ui():
    import gradio as gr
    from upload_safety import install_upload_fix
    install_upload_fix()
    from inpaint import build_inpaint_ui

    with gr.Blocks(title="Floyd Headliner · Get Going Fast") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                '<header class="ggf-hero">'
                '<div class="ggf-hero-copy">'
                '<div class="ggf-kicker">GET GOING FAST · LOCAL AI</div>'
                '<h1>Floyd Headliner</h1>'
                '<p>Likeness Transfer &amp; Inpaint · Edit with a prompt.</p>'
                f'<div class="ggf-profile" role="status">Running mode: {running_profile_label()}</div>'
                '<nav class="ggf-links" aria-label="Get Going Fast links">'
                '<a href="https://getgoingfast.pro" target="_blank" rel="noopener noreferrer">GetGoingFast.pro ↗</a>'
                '<a href="https://www.youtube.com/@theaihobbyguy" target="_blank" rel="noopener noreferrer">TheAIHobbyGuy on YouTube ↗</a>'
                '</nav></div>'
                '<div class="ggf-hero-slogan">Your Time Is Limited.<br><strong>Get Going Fast.</strong></div>'
                f'{hero_logo_html()}'
                '</header>',
                elem_id="ggf-hero-block",
            )
            with gr.Row(elem_classes="ggf-update-row"):
                check_updates = gr.Button("Check for updates", variant="secondary", size="sm", scale=0,
                                          min_width=180, elem_id="ggf-check-updates")
                update_button = gr.Button("Update", interactive=False, size="sm", scale=0,
                                          min_width=160, elem_id="ggf-update-button")
            update_status = gr.Markdown("Updates not checked yet.", elem_classes="ggf-update-status")

            def check_updates_ui():
                message, available = check_for_updates()
                if available:
                    gr.Info("Update available. Close Floyd Headliner, then run 3-UPDATE-Floyd-Headliner.bat.")
                return message, gr.update(interactive=available)

            def show_update_instructions():
                message = "Close Floyd Headliner, run 3-UPDATE-Floyd-Headliner.bat, then start the app again."
                gr.Info(message)
                return message

            check_updates.click(check_updates_ui, outputs=[update_status, update_button], queue=False, api_name=False)
            update_button.click(show_update_instructions, outputs=[update_status], queue=False, api_name=False)
            gr.Markdown("Use only images you have the right and consent to edit. Do not use the app for impersonation or deception.",
                        elem_classes="warning")
            with gr.Tabs() as tabs:
                with gr.Tab("Likeness Transfer", id="likeness"):
                    body_input, body_mask, likeness_result, send_to_inpaint = build_likeness_ui(gr)
                with gr.Tab("Inpaint", id="inpaint"):
                    (inpaint_input, inpaint_result, send_to_likeness,
                     inpaint_reset_outputs, clear_inpaint_result) = build_inpaint_ui(gr, sys.modules[__name__])
                with gr.Tab("Settings", id="settings"):
                    with gr.Accordion("Network access", open=True):
                        gr.Markdown("Choose where Headliner can be opened. Changes take effect after restarting the server. "
                                    "A public link with no password can be used by anyone who has the link, including to change these settings.")
                        current_network = gr.Markdown(active_network_status())
                        gr.Button("Refresh current address").click(
                            active_network_status, outputs=current_network, queue=False, api_name=False)
                        network_config = read_settings()
                        network_mode = gr.Dropdown(
                            label="Access mode",
                            choices=[("This computer only", "local"), ("Local network (LAN)", "lan"),
                                     ("Temporary public link", "public")],
                            value=network_config["mode"],
                            elem_id="ggf-network-mode",
                        )
                        network_user = gr.Textbox(label="Username (optional; defaults to ggf if using a password)",
                                                  value=network_config["username"], max_lines=1)
                        network_password = gr.Textbox(label="Password (optional; blank means no login)",
                                                      type="password", value="", max_lines=1,
                                                      elem_id="ggf-network-password")
                        gr.Markdown("LAN access may prompt for a Windows Firewall rule. A temporary public link uses "
                                    "Gradio sharing. Any nonblank password you enter is accepted. Leaving it blank "
                                    "disables login after restart, even if a password was set before. "
                                    "Passwords that are set are stored locally as salted hashes.")
                        network_saved = gr.Markdown("")
                        gr.Button("Save network settings").click(
                            save_network_mode,
                            inputs=[network_mode, network_user, network_password],
                            outputs=[network_saved, network_password],
                            queue=False, api_name=False,
                        )
                    demo.load(active_network_status, outputs=current_network, queue=False, api_name=False)

            def open_inpaint(result):
                if result is None:
                    raise ValueError("Generate a likeness result first.")
                return result, gr.update(selected="inpaint"), *clear_inpaint_result()

            send_to_inpaint.click(open_inpaint, inputs=[likeness_result],
                                  outputs=[inpaint_input, tabs, *inpaint_reset_outputs], api_name=False)

            def select_likeness(result):
                if result is None:
                    raise ValueError("Generate an inpaint result first.")
                return gr.update(selected="likeness")

            def clear_likeness_input():
                return None

            def open_likeness(result):
                return result_as_body_input(result)

            send_to_likeness.click(
                select_likeness, inputs=[inpaint_result], outputs=[tabs],
                queue=False, api_name=False,
            ).success(
                clear_likeness_input, outputs=[body_input], queue=False, api_name=False,
            ).success(
                open_likeness, inputs=[inpaint_result], outputs=[body_input, body_mask],
                queue=False, api_name=False,
            )
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
    from core_runtime import available as fast_core_available, inpaint_available

    print(f"App-local Qwen core: {'available' if inpaint_available() else 'not installed'}")
    print(f"Fast likeness engine: {'available' if fast_core_available() else 'not installed'}")
    if model_profile() == "standard" and not inpaint_available():
        print("Standard-profile Inpaint: INCOMPLETE (run the installer or updater)")
        return 2
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


def save_network_mode(mode: str, username: str, password: str):
    import gradio as gr

    try:
        saved = save_settings(mode, username, password)
    except ValueError as error:
        return f"**Not saved:** {error}", gr.update()
    except OSError as error:
        return f"**Not saved:** Could not write network settings ({error}).", gr.update()
    label = {"local": "This computer only", "lan": "Local network", "public": "Temporary public link"}[saved["mode"]]
    access_note = (f"Login will use username **{saved['username']}** and the password you entered."
                   if saved["digest"] else
                   "**No login will be required. Anyone with the address can use the app and change Settings.**")
    return (f"Saved **{label}**. {access_note} Close this server and start Floyd Headliner again to apply it. "
            "The current address remains unchanged until restart."), ""


def active_network_status() -> str:
    label = {"local": "This computer only", "lan": "Local network", "public": "Temporary public link"}[ACTIVE_MODE]
    login = "Required" if ACTIVE_AUTH_ENABLED else "None"
    return f"**Running mode:** {label}  \n**Login:** {login}  \n**Current address:** {ACTIVE_URL or 'Starting...'}"


def main() -> int:
    global ACTIVE_MODE, ACTIVE_URL, ACTIVE_AUTH_ENABLED

    parser = argparse.ArgumentParser(description="Standalone Floyd Headliner")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--profile", choices=("standard", "low-vram", "low-vram-12gb"),
                        help="Start a specific installed mode without the console menu")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    try:
        settings = read_settings()
    except ValueError as error:
        print(f"Network settings error: {error}")
        print("The app was not exposed. Repair the local network_settings.json file before restarting.")
        return 1
    ACTIVE_MODE = settings["mode"]
    ACTIVE_AUTH_ENABLED = ACTIVE_MODE != "local" and bool(settings["digest"])
    if args.host not in ("127.0.0.1", "localhost"):
        print("--host cannot change network exposure. Choose LAN or public access in Settings instead.")
        return 1
    access = {"local": "127.0.0.1", "lan": "0.0.0.0", "public": "127.0.0.1"}[ACTIVE_MODE]

    if ACTIVE_MODE == "local":
        ports = (args.port,) if args.port is not None else range(7860, 7870)
        for port in ports:
            if existing_local_app(port):
                url = f"http://127.0.0.1:{port}"
                print(f"Floyd Headliner is already running at {url}")
                print("Close the running app's console before restarting to change model modes.")
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
        if args.port is not None and local_port_in_use(args.port):
            print(f"Port {args.port} is in use by another service. Start without --port to choose a free port.")
            return 1

    if not select_startup_profile(args.profile):
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    auth = None if not ACTIVE_AUTH_ENABLED else (
        lambda username, password: verify_login(username, password, settings)
    )
    _, local_url, public_url = demo.queue(default_concurrency_limit=1).launch(
        server_name=access,
        server_port=args.port,
        inbrowser=ACTIVE_MODE == "local" and not args.no_browser,
        share=ACTIVE_MODE == "public",
        auth=auth,
        show_error=True,
        css=APP_CSS,
        js=APP_JS,
        allowed_paths=[str(OUTPUT_DIR)],
        prevent_thread_lock=True,
    )
    ACTIVE_URL = public_url or local_url
    print(f"Floyd Headliner running in {ACTIVE_MODE} mode at {ACTIVE_URL}", flush=True)
    if ACTIVE_MODE == "public" and not public_url:
        demo.close()
        print("Public share link could not be created. The server was closed.")
        return 1
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
