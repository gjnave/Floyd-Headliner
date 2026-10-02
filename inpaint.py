"""Prompt-based local image editing with the base Qwen Image 2.1 model."""
from __future__ import annotations

import gc
import random
import time
from datetime import datetime

from PIL import Image, ImageOps, PngImagePlugin


MAX_PIXELS = 16_777_216
MAX_SIDE = 4096


def prepare_image(image):
    if image is None:
        raise ValueError("Upload an input image before generating.")
    if not isinstance(image, Image.Image):
        raise ValueError("The input image could not be read. Upload it again.")
    if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_SIDE:
        raise ValueError("Image is too large. Use an image up to 4096 × 4096 pixels.")
    try:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("The input image could not be read. Upload it again.") from exc


def editing_prompt(instruction):
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("Type what you want to change before generating.")
    return (
        "Edit <image1> according to this instruction: " + instruction + ". "
        "Keep the original framing, people, objects, lighting, and background "
        "except where the instruction requires a change. Make the requested edit "
        "look natural and complete. Return a finished image."
    )


def run_inpaint(runtime, image, instruction, seed, randomize, steps, megapixels,
                keep_on_gpu, progress=None, turbo_preview=False):
    original = prepare_image(image)
    prompt = editing_prompt(instruction)
    step_count = int(steps)
    if not 1 <= step_count <= 60:
        raise ValueError("Steps must be between 1 and 60.")
    used_seed = random.randint(0, 2**63 - 1) if randomize else int(seed)
    if not 0 <= used_seed < 2**63:
        raise ValueError("Seed must be between 0 and 9223372036854775807.")

    if runtime.model_profile() == "standard":
        from core_runtime import run_inpaint as run_core_inpaint

        return run_core_inpaint(runtime, original, prompt, used_seed, step_count,
                                megapixels, keep_on_gpu, progress=progress,
                                turbo_preview=turbo_preview)

    import torch

    width, height = runtime.calculate_working_size(original, megapixels)
    started = time.perf_counter()
    if progress is not None:
        progress(.02, desc="Loading the base model")
    with runtime._PIPELINE_LOCK:
        # The optional likeness fast core keeps a separate large model set.
        # Release it before loading the standalone base model for Inpaint.
        runtime.release_core_models()
        pipe = runtime._load_pipeline()
        runtime.configure_speed_mode(pipe, keep_on_gpu, width, height)
        # The shared pipeline keeps adapters for Likeness Transfer. None are
        # active for image editing.
        pipe.disable_lora()
        if progress is not None:
            progress(.12, desc="Reading your image and instruction")

        def on_step(pipeline, index, timestep, values):
            if progress is not None:
                progress(.2 + .7 * (index + 1) / step_count,
                         desc=f"Editing {index + 1}/{step_count}")
            return values

        generated = pipe(
            prompt=prompt,
            image=[original],
            width=width, height=height,
            num_inference_steps=step_count, true_cfg_scale=1.0,
            output_resolution=640,
            generator=torch.Generator(device="cuda").manual_seed(used_seed),
            callback_on_step_end=on_step,
        ).images[0]
        cache_hit = pipe.prompt_cache_hit
        gc.collect()
        torch.cuda.empty_cache()

    result = generated if turbo_preview else generated.resize(original.size, Image.Resampling.LANCZOS)
    comparison_input = original.resize(result.size, Image.Resampling.LANCZOS) if turbo_preview else original
    runtime.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = runtime.OUTPUT_DIR / f"floyd-inpaint-{datetime.now():%Y%m%d-%H%M%S-%f}-seed-{used_seed}.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Floyd Headliner", "Base Qwen Image 2.1 edit; LoRAs disabled")
    metadata.add_text("prompt", prompt)
    metadata.add_text("seed", str(used_seed))
    result.save(filename, pnginfo=metadata)
    size_note = "Turbo preview saved at working size. " if turbo_preview else "Original dimensions retained. "
    status = (
        f"Saved {filename.name}  \n"
        f"Seed: {used_seed} | Time: {time.perf_counter() - started:.1f} s | "
        f"Working canvas: {width} × {height} | Base model | "
        f"Cache: {'reused' if cache_hit else 'encoded'}  \n"
        f"{size_note}The model may change details outside the requested object."
    )
    if progress is not None:
        progress(1, desc="Saved")
    return (comparison_input, result), str(filename), status, str(used_seed), result


def build_inpaint_ui(gr, runtime):
    gr.Markdown(
        "Upload an image and describe the change you want, such as “remove the hat.” "
        "The base Qwen Image 2.1 model edits the full image, so unrelated details may also change.",
        elem_classes="ggf-guide",
    )
    instruction = gr.Textbox(
        label="What should change?", lines=2,
        placeholder="For example: remove the hat and restore the hair underneath.",
    )
    with gr.Row(equal_height=True):
        source = gr.Image(type="pil", height=560, label="INPUT IMAGE",
                          scale=1, min_width=440, elem_id="ggf-inpaint-source")
        comparison = gr.ImageSlider(
            type="pil", format="png", interactive=False,
            label="OUTPUT — drag to compare with the input",
            buttons=["fullscreen"], height=560, scale=1, min_width=440,
        )
    turbo_generate = gr.Button(
        "Turbo preview · half-size · Ctrl+Enter", variant="secondary", elem_id="ggf-inpaint-turbo",
    )
    generate = gr.Button(
        "Generate edit (full size)", variant="primary", elem_id="ggf-inpaint-button",
    )
    with gr.Accordion("Edit settings", open=False):
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed")
            randomize = gr.Checkbox(True, label="Randomize seed")
            steps = gr.Slider(10, 60, value=40, step=1, label="Steps (base model)")
            small = runtime.model_profile() == "low-vram-12gb"
            megapixels = gr.Slider(
                .25 if small else .5, .5 if small else 1.5,
                value=.5 if small else 1.0, step=.05 if small else .1,
                label="Working megapixels (12 GB test)" if small else "Working megapixels",
            )
        standard = runtime.model_profile() == "standard"
        keep_gpu = gr.Checkbox(
            standard, interactive=standard,
            label="Keep model on GPU for faster repeat edits" if standard
            else "Low-VRAM mode: automatic CPU offload",
        )
    status = gr.Markdown(elem_classes="ggf-status")
    result_state = gr.State(None)
    with gr.Row():
        download = gr.DownloadButton("Download PNG", value=None, visible=False, size="sm")
        accept = gr.Button("Use result for the next edit", interactive=False)
        send_to_likeness = gr.Button("Send result to Likeness Transfer", interactive=False)
    release = gr.Button("Release models / free GPU memory")

    def generate_edit(input_image, text, seed_value, random_seed, step_count,
                      resolution, resident, progress=gr.Progress(), turbo=False):
        if turbo:
            resolution = runtime.turbo_megapixels(resolution)
        pair, path, message, _used_seed, image = run_inpaint(
            runtime, input_image, text, seed_value, random_seed,
            step_count, resolution, resident, progress=progress, turbo_preview=turbo,
        )
        return (pair, gr.update(value=path, visible=True), message, image,
                gr.update(interactive=True), gr.update(interactive=True))

    inputs = [source, instruction, seed, randomize, steps, megapixels, keep_gpu]
    outputs = [comparison, download, status, result_state, accept, send_to_likeness]
    generate.click(
        generate_edit,
        inputs=inputs, outputs=outputs,
        concurrency_id="gpu", api_name="inpaint", show_progress_on=[comparison],
    )
    def generate_turbo_edit(*args, progress=gr.Progress()):
        return generate_edit(*args, progress=progress, turbo=True)

    turbo_generate.click(
        generate_turbo_edit, inputs=inputs, outputs=outputs,
        concurrency_id="gpu", api_name="turbo_inpaint", show_progress_on=[comparison],
    )

    def clear_result():
        return (None, None, gr.update(value=None, visible=False),
                gr.update(interactive=False), gr.update(interactive=False), "")

    reset_outputs = [comparison, result_state, download, accept, send_to_likeness, status]

    def accept_result(result):
        if result is None:
            raise ValueError("Generate an edit first.")
        return result, *clear_result()

    accept.click(accept_result, inputs=[result_state], outputs=[source, *reset_outputs], api_name=False)
    source.upload(clear_result, outputs=reset_outputs,
                  show_progress="hidden", queue=False, api_name=False)
    source.clear(clear_result, outputs=reset_outputs,
                 show_progress="hidden", queue=False, api_name=False)
    release.click(runtime.release_models, outputs=[status], concurrency_id="gpu")
    gr.Markdown(
        "Uses the app-local Qwen core on the standard profile; low-VRAM profiles "
        "keep their existing offload path. Both LoRAs are disabled for this tab.",
        elem_classes="ggf-guide",
    )
    return source, result_state, send_to_likeness, reset_outputs, clear_result
