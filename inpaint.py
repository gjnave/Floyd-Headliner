"""Local brush-guided editing using the installed Qwen model.

The canvas interaction is inspired by MagicQuill; no MagicQuill code or runtime
is bundled. Brush data stays separate from the original image. The generation
receives explicit visual guides, and only the selected pixels are composited.
"""
from __future__ import annotations

import base64
import gc
import io
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, PngImagePlugin


MAX_PIXELS = 16_777_216
MAX_STROKES = 2000
MAX_POINTS = 150_000
ASSET_DIR = Path(__file__).resolve().parent / "inpaint_assets"


def image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str) or not value.startswith("data:image/") or len(value) > 96_000_000:
        raise ValueError("Upload an image in the Inpaint canvas first.")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image is too large. Use an image up to 4096 × 4096 pixels.")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, IndexError, base64.binascii.Error) as exc:
        raise ValueError("The canvas image could not be read. Upload it again.") from exc


def _stroke(draw, points, width, fill):
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def prepare_edit(payload, grow=24, fill_closed=True):
    if not isinstance(payload, dict):
        raise ValueError("Upload an image and draw the area you want to edit.")
    original = decode_image(payload.get("image"))
    strokes = payload.get("strokes") or []
    if not isinstance(strokes, list) or len(strokes) > MAX_STROKES:
        raise ValueError("Too many brush strokes. Accept an edit or clear the drawing first.")
    masks = {key: Image.new("L", original.size) for key in ("add", "remove", "color")}
    colors = Image.new("RGBA", original.size)
    count = 0
    for stroke in strokes:
        tool = stroke.get("tool")
        if tool not in (*masks, "erase"):
            raise ValueError("Unknown canvas tool.")
        raw_points = stroke.get("points") or []
        count += len(raw_points)
        if count > MAX_POINTS:
            raise ValueError("Drawing is too complex. Accept an edit or simplify the drawing.")
        points = []
        for point in raw_points:
            if len(point) != 2 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in point):
                raise ValueError("Invalid brush coordinates.")
            points.append((round(max(0, min(1, point[0])) * (original.width - 1)),
                           round(max(0, min(1, point[1])) * (original.height - 1))))
        if not points:
            continue
        size = float(stroke.get("size", 20))
        if not math.isfinite(size):
            raise ValueError("Invalid brush size.")
        width = max(1, min(256, round(size)))
        if tool == "erase":
            for mask in masks.values():
                _stroke(ImageDraw.Draw(mask), points, width, 0)
            _stroke(ImageDraw.Draw(colors), points, width, (0, 0, 0, 0))
            continue
        _stroke(ImageDraw.Draw(masks[tool]), points, width, 255)
        # Last brush wins where different tools overlap.
        for other in masks:
            if other != tool:
                _stroke(ImageDraw.Draw(masks[other]), points, width, 0)
        if tool == "color":
            color = stroke.get("color", "#f5b942")
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("Invalid brush color.")
            _stroke(ImageDraw.Draw(colors), points, width, color)

    mask = ImageChops.lighter(ImageChops.lighter(masks["add"], masks["remove"]), masks["color"])
    if not mask.getbbox():
        raise ValueError("Draw with + Add, − Remove, or Color before generating.")
    if fill_closed:
        # Fill only closed loops made with + Add, never loops crossing unrelated tools.
        framed = ImageOps.expand(masks["add"], border=1, fill=0)
        exterior = framed.copy()
        ImageDraw.floodfill(exterior, (0, 0), 255)
        holes = ImageOps.invert(exterior).crop((1, 1, original.width + 1, original.height + 1))
        mask = ImageChops.lighter(mask, holes)
    radius = max(0, min(128, int(grow)))
    if radius:
        # MaxFilter is separable in Pillow; expands the edit region, not the guides.
        mask = mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    guide = original.copy()
    # Tool overlay colors belong to the browser canvas, not model conditioning.
    # A neutral guide is only needed when several tool types share a drawing.
    guide.paste((210, 210, 210), mask=masks["add"])
    guide.paste((55, 55, 55), mask=masks["remove"])
    guide.paste(colors.convert("RGB"), mask=masks["color"])
    left, top, right, bottom = mask.getbbox()
    padding = max(96, round(max(right - left, bottom - top) * .45))
    crop = (max(0, left - padding), max(0, top - padding),
            min(original.width, right + padding), min(original.height, bottom + padding))
    modes = [key for key, value in masks.items() if value.getbbox()]
    return original, guide, mask, crop, modes


def editing_prompt(instruction, modes, change_background=False):
    remove_only = modes == ["remove"]
    add_only = modes == ["add"]
    text = (
        "Edit <image1>, the original photograph. <image2> is an annotated editing guide, "
        "not a desired output. <image3> is the edit mask: white marks the editable area, "
        "black must stay unchanged. The neutral light/dark tool marks in <image2> show "
        "Add/Remove locations, not output colors. Only Color-tool strokes specify a desired "
        "output color. Keep exactly the same framing, camera, lighting and all unrelated "
        "content as <image1>. Return a finished natural image, without guide lines, masks, "
        "labels or annotations. "
    )
    if remove_only:
        text = (
            "Edit <image1>, the original photograph. <image2> is a black-and-white selection mask: "
            "white selects the object or detail to remove, black must stay unchanged. "
            "Delete the selected content and fill its former location naturally using the "
            "surrounding scene. Return the completed photograph with the same framing, camera "
            "and lighting, without a mask, outline, solid fill, or painted marks. "
        )
    elif add_only:
        text = (
            "Edit <image1>, the original photograph. <image2> is a black-and-white "
            "selection mask: the white shape shows where to ADD a realistic object and "
            "black must stay unchanged. The mask's white color is not an output color. "
            "Match the scene's camera, lighting and materials. Return a finished natural "
            "image without any mask or outline. "
        )
    if change_background:
        text += ("The user enabled background changes. You may edit the setting inside the "
                 "painted area as requested. Preserve people and foreground objects unless "
                 "the instruction explicitly changes them. ")
    else:
        text += ("Do not change the original background or setting: keep its scenery, "
                 "colors, lighting, and perspective. Edit only the selected subject or object. "
                 "If an object is removed, reconstruct only the background that was hidden "
                 "behind that object, matching the existing scene. Keep people present. ")
    if "add" in modes:
        text += ("The selected shape describes the placement of something to ADD or replace. "
                 "Turn that rough shape into a realistic object matching the scene; do not "
                 "copy the selection color or leave brush marks. ")
    if "remove" in modes:
        if not remove_only:
            text += "The dark neutral marks in <image2> mark content to REMOVE. "
        text += ("Remove that content. When removing clothing, replace it with natural "
                 "clothing suitable for the scene; do not erase the person or fill the area "
                 "with background. When removing an accessory, restore the underlying "
                 "appearance without adding another accessory. For other objects, reconstruct "
                 "the surface that was behind them. ")
    if "color" in modes:
        text += ("Other painted colors in <image2> are desired color guides. Apply those "
                 "colors naturally to the corresponding surfaces, retaining texture and shading. ")
    instruction = (instruction or "").strip()
    if instruction:
        text += "User's requested edit: " + instruction
    else:
        text += "Infer the intended edit from the sketch and the scene."
    if not change_background:
        text += (" Background changes are disabled. Do not replace or restyle the setting; "
                 "only restore matching background where a removed object previously covered it.")
    return text


def run_inpaint(runtime, payload, instruction, seed, randomize, steps, megapixels, grow,
                feather, fill_closed, keep_on_gpu, change_background=False, progress=None):
    original, guide, mask, crop, modes = prepare_edit(payload, grow, fill_closed)
    import torch

    width, height = runtime.calculate_working_size(original.crop(crop), megapixels)
    used_seed = random.randint(0, 2**63 - 1) if randomize else int(seed)
    if not 0 <= used_seed < 2**63:
        raise ValueError("Seed must be between 0 and 9223372036854775807.")
    started = time.perf_counter()
    inferred = ""
    if not (instruction or "").strip():
        if progress is not None:
            progress(.01, desc="Identifying the intended edit")
        inferred = suggest_prompt(runtime, payload, grow, fill_closed, change_background)
        instruction = inferred
    prompt = editing_prompt(instruction, modes, change_background)
    if progress is not None:
        progress(.02, desc="Loading local models")
    with runtime._PIPELINE_LOCK:
        pipe = runtime._load_pipeline()
        runtime.configure_speed_mode(pipe, keep_on_gpu, width, height)
        # The identity adapter must not affect general editing.
        pipe.set_adapters(["turbo"], adapter_weights=[1.0])
        if progress is not None:
            progress(.12, desc="Reading your image and drawing")

        def on_step(pipeline, index, timestep, values):
            if progress is not None:
                progress(.25 + .65 * (index + 1) / int(steps), desc=f"Inpainting {index + 1}/{int(steps)}")
            return values

        references = ([original.crop(crop), mask.crop(crop).convert("RGB")]
                      if modes in (["remove"], ["add"]) else
                      [original.crop(crop), guide.crop(crop), mask.crop(crop).convert("RGB")])
        generated = pipe(
            prompt=prompt,
            image=references,
            width=width, height=height, num_inference_steps=int(steps), true_cfg_scale=1.0,
            # Three 1 MP references can spill a 24 GB card into shared memory.
            # Guide resolution is independent of the generated patch dimensions.
            output_resolution=640,
            generator=torch.Generator(device="cuda").manual_seed(used_seed),
            callback_on_step_end=on_step,
        ).images[0]
        cache_hit = pipe.prompt_cache_hit
        # Return transient attention/decoder allocations before the next edit.
        gc.collect()
        torch.cuda.empty_cache()
    result = runtime.composite_selection(original, generated, mask, crop, feather)
    runtime.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = runtime.OUTPUT_DIR / f"floyd-inpaint-{datetime.now():%Y%m%d-%H%M%S-%f}-seed-{used_seed}.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Floyd Headliner", "Inpaint — Qwen Image 2.1 + Viggle Turbo; identity LoRA disabled")
    metadata.add_text("prompt", prompt)
    metadata.add_text("seed", str(used_seed))
    metadata.add_text("tools", ", ".join(modes))
    metadata.add_text("change_background", str(bool(change_background)).lower())
    result.save(filename, pnginfo=metadata)
    status = (f"Saved `{filename.name}`  \nSeed: `{used_seed}` | Time: `{time.perf_counter() - started:.1f} s` | "
              f"Working canvas: `{width} × {height}` | Tools: `{', '.join(modes)}` | Background edits: `{'allowed in painted area' if change_background else 'off'}` | Cache: `{'reused' if cache_hit else 'encoded'}`  \n"
              "Original dimensions retained. Pixels outside the expanded edit area are unchanged.")
    if progress is not None:
        progress(1, desc="Saved")
    if inferred:
        # Keep model-produced text out of Markdown/HTML interpretation.
        import html
        status += "  \nAutomatic edit instruction: " + html.escape(inferred)
    return (original, result), str(filename), status, str(used_seed), result


def suggest_prompt(runtime, payload, grow, fill_closed, change_background=False):
    """Reuse the installed vision-language encoder for a short Draw & Guess suggestion."""
    original, guide, mask, crop, modes = prepare_edit(payload, grow, fill_closed)
    import torch

    background_rule = ("The background may change inside the marked area."
                       if change_background else "Keep the original background unchanged.")
    message = (
        "These images show an original photo followed by the same photo with editing marks. "
        "Light neutral marks sketch an object to add; dark neutral marks show removal; "
        "other colored strokes specify a new color. Tool highlights are NOT colors "
        "requested for the finished object. Infer the real-world object suggested "
        "by the sketch's shape and location, such as eyewear over eyes or a hat over a head. "
        "Write ONE short image-edit instruction "
        "describing the intended change, using concrete object names. Do not describe the "
        "marks themselves. Do not include explanations. Active tools: " + ", ".join(modes) +
        ". " + background_rule
    )
    if modes == ["remove"]:
        message = (
            "Image 1 is the original photograph. Image 2 is a black-and-white mask with "
            "the selected area in white. "
            "Identify the real object underneath those marks in image 1. Write one precise "
            "image editing instruction beginning with 'Remove the' followed by the concrete "
            "object name. Describe what naturally replaces it. White is only a mask, not a "
            "desired output color. Do not say 'remove the white marks' or 'remove the selected region'. "
            "Do not add an object. For clothing edits, describe replacement clothing. "
            "For accessories, restore the underlying appearance. "
            "Return only the edit instruction. " + background_rule
        )
    elif modes == ["add"]:
        message = (
            "Image 1 is the original photograph. Image 2 is a black-and-white mask whose "
            "white shape shows where a new object should be added. Identify a plausible "
            "object from the shape and scene; white is not a requested output color. "
            "Return one short, concrete image editing instruction. " + background_rule
        )
    elif modes == ["color"]:
        message = (
            "Image 1 is the original photograph. Image 2 shows a painted color guide. "
            "Identify the surface under the painted area and the color chosen for it. "
            "Return one short instruction to recolor only that surface, preserving its "
            "texture and the rest of the scene. " + background_rule
        )
    images = [original.crop(crop),
              mask.crop(crop).convert("RGB") if modes in (["remove"], ["add"]) else guide.crop(crop)]
    for img in images:
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)
    with runtime._PIPELINE_LOCK:
        pipe = runtime._load_pipeline()
        device = pipe._execution_device
        if pipe.fast_resident:
            pipe.transformer.to("cpu")
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            pipe.text_encoder.to(device)
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": message}]}]
            text = pipe.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = pipe.processor(text=[text], images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = pipe.text_encoder.generate(**inputs, max_new_tokens=100, do_sample=False)
            suggestion = pipe.processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            if not suggestion:
                raise ValueError("Could not infer an edit. Describe what you want in the prompt box.")
            return suggestion
        finally:
            if pipe.fast_resident:
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()


def build_inpaint_ui(gr, runtime):
    with gr.Row(equal_height=True):
        canvas = gr.HTML(
            value=None, html_template=(ASSET_DIR / "canvas.html").read_text(encoding="utf-8"),
            css_template=(ASSET_DIR / "canvas.css").read_text(encoding="utf-8"),
            js_on_load=(ASSET_DIR / "canvas.js").read_text(encoding="utf-8"),
            elem_id="ggf-inpaint-canvas", scale=1, min_width=440,
        )
        comparison = gr.ImageSlider(type="pil", format="png", interactive=False,
                                    label="BEFORE / AFTER — original left, edit right",
                                    buttons=["fullscreen"], height=680, scale=1, min_width=440)
    change_background = gr.Checkbox(False, label="Change background in painted area",
                                    info="Off by default. Turn on to edit the setting, then paint the area to change.")
    gr.Markdown("Draw with **+ Add**, **− Remove**, or **Color**. A blank prompt identifies the intended edit "
                "and generates it in one click. Describe replacements, especially clothing, for more control. "
                "The original stays intact outside the painted edit area.", elem_classes="ggf-guide")
    payload = gr.JSON(value={}, visible=False)
    with gr.Row():
        keep_prompt = gr.Checkbox(False, label="Keep prompt", scale=1, min_width=125)
        instruction = gr.Textbox(label="What should change?", lines=2, scale=5,
                                 placeholder="Add a cowboy hat; remove the lamp; turn the coat blue…")
    guess = gr.Button("Guess from drawing", size="sm")
    gr.Markdown("Guessing is optional and may misread a sketch. You can edit the suggestion before generating.")
    preview_button = gr.Button("Preview edit area", size="sm")
    generate = gr.Button("Generate edit · Ctrl+Enter", variant="primary", elem_id="ggf-inpaint-button")
    with gr.Accordion("Inpaint settings", open=False):
        with gr.Row():
            grow = gr.Slider(0, 128, value=24, step=1, label="Room around strokes (image pixels)")
            feather = gr.Slider(0, 32, value=8, step=1, label="Blend edge softness")
        fill_closed = gr.Checkbox(True, label="Fill the inside of closed + Add outlines")
        gr.Markdown("Draw the outline of the whole object you want to add. Increase room around strokes if "
                    "the edit is clipped. For removal, paint over the whole object.")
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed")
            randomize = gr.Checkbox(True, label="Randomize seed")
            steps = gr.Slider(6, 24, value=6, step=1, label="Steps")
            small = runtime.model_profile() == "low-vram-12gb"
            megapixels = gr.Slider(.25 if small else .5, .5 if small else 1.5,
                                   value=.5 if small else 1.0, step=.05 if small else .1,
                                   label="Working megapixels (12 GB test)" if small else "Working megapixels")
        standard = runtime.model_profile() == "standard"
        keep_gpu = gr.Checkbox(standard, interactive=standard,
                              label="Keep model on GPU for faster repeat edits" if standard
                              else "Low-VRAM mode: automatic CPU offload")
    with gr.Accordion("Edit area preview", open=False) as preview_panel:
        gr.Markdown("Refresh this preview after changing your drawing or the Inpaint settings.")
        preview_image = gr.Image(label="Gold area can change; everything else stays original", interactive=False, type="pil", height=350)
    status = gr.Markdown(elem_classes="ggf-status")
    result_state = gr.State(None)
    with gr.Row():
        download = gr.DownloadButton("Download PNG", value=None, visible=False, size="sm")
        accept = gr.Button("Use result for the next edit", interactive=False)
    release = gr.Button("Release models / free GPU memory")

    def clear_prompt(keep):
        return gr.skip() if keep else ""

    canvas.change(fn=None, inputs=[keep_prompt, instruction], outputs=[instruction],
                  js="(keep, text) => keep ? text : ''",
                  show_progress="hidden", queue=False, api_name=False)

    capture_js = """(...args) => {
        const root = document.getElementById('ggf-inpaint-canvas');
        args[0] = root && root.getDrawing ? root.getDrawing() : {};
        return args;
    }"""

    def generate_edit(data, text, seed_value, random_seed, step_count, resolution, growth,
                      softness, fill, resident, change_scene, progress=gr.Progress()):
        pair, path, message, _used_seed, image = run_inpaint(
            runtime, data, text, seed_value, random_seed, step_count, resolution,
            growth, softness, fill, resident, change_scene, progress=progress,
        )
        return pair, gr.update(value=path, visible=True), message, image, gr.update(interactive=True)

    generate.click(generate_edit,
                   inputs=[payload, instruction, seed, randomize, steps, megapixels, grow, feather, fill_closed, keep_gpu, change_background],
                   outputs=[comparison, download, status, result_state, accept],
                   js=capture_js, concurrency_id="gpu", api_name="inpaint",
                   show_progress_on=[comparison])
    guess.click(lambda data, growth, fill, change_scene: suggest_prompt(runtime, data, growth, fill, change_scene),
                inputs=[payload, grow, fill_closed, change_background], outputs=[instruction], js=capture_js,
                concurrency_id="gpu", api_name="guess_inpaint_prompt")

    def preview_edit(data, growth, fill):
        original, _, mask, _, _ = prepare_edit(data, growth, fill)
        tinted = Image.blend(original, Image.new("RGB", original.size, "#f5b942"), .5)
        return Image.composite(tinted, original, mask), gr.update(open=True)

    preview_button.click(preview_edit, inputs=[payload, grow, fill_closed],
                         outputs=[preview_image, preview_panel], js=capture_js, api_name=False)

    def accept_result(result):
        if result is None:
            raise ValueError("Generate an edit first.")
        return {"load_image": image_data_url(result)}, gr.update(interactive=False)

    accept.click(accept_result, inputs=[result_state], outputs=[canvas, accept], api_name=False)
    canvas.upload(lambda keep: (None, gr.update(interactive=False), None,
                                gr.update(open=False), clear_prompt(keep)),
                  inputs=[keep_prompt],
                  outputs=[result_state, accept, preview_image, preview_panel, instruction],
                  show_progress="hidden", queue=False, api_name=False)
    release.click(runtime.release_models, outputs=[status], concurrency_id="gpu")
    gr.Markdown("Brush controls inspired by [MagicQuill](https://github.com/robbyant-research/MagicQuill). "
                "Powered here by your installed Qwen Image 2.1 model and Viggle Turbo. "
                "The identity LoRA is disabled for Inpaint.", elem_classes="ggf-guide")
    return canvas
