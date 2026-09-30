![Floyd Headliner — Likeness Transfer](assets/floydheadliner.png)

# Floyd Headliner — Likeness Transfer

## Get going fast

### [Quick Setup Helper → GetGoingFast.pro/tools/floydheadliner](https://getgoingfast.pro/tools/floydheadliner)

For walkthroughs and updates, visit
[The AI Hobby Guy on YouTube](https://youtube.com/@theaihobbyguy).

Floyd Headliner is a local Windows likeness-transfer app: provide a **body reference**
for pose, clothing, and scene, then a **head reference** for face, hair, and
likeness. It recreates the active image-edit path from
`Qwen-Image-2_1-BFS-Character-Swap-Image-Edit-I2I.json` as a direct Python app
.
It is standalone and does not require ComfyUI

The Qwen Image 2.1 base model and BFS/Viggle adapters come from their original
publishers; Get Going Fast does not claim ownership of those models.

## Install and run from source

For guided setup, use the [Quick Setup Helper](https://getgoingfast.pro/tools/floydheadliner)
at Get Going Fast. This public repository contains the app source; the
one-click setup package is a separate Get Going Fast member offering.

To set up this source checkout yourself on Windows, open **Command Prompt** in
the repository root and run:

```bat
git clone https://codeberg.org/Cognibuild/Floyd-Headliner
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --upgrade pip setuptools wheel
uv pip install --python .venv\Scripts\python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe app.py
```

![Floyd Headliner — Likeness Transfer](assets/fh2.png)


In the app, add the body/base image on the left and the head reference in the
middle. Optionally enter an **Extra prompt** such as “Remove the hat.” Click
**Swap Head**, or press **Ctrl+Enter**. Generated PNG files are saved in
`outputs/` in this source checkout.

## What is reproduced

- Qwen Image 2.1 direct Diffusers inference
- Body/base as `<image1>` and head reference as `<image2>`
- `bfs_head_v1.1_alternative_qwen_2.1.safetensors` at strength 1.0
- `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors` at strength 1.0
- Six denoising steps and CFG 1.0
- About a 1 MP ratio-preserving working canvas, rounded to multiples of 32
- 2x Lanczos output scaling
- Automatic GPU residency or CPU/GPU offload according to available VRAM

Diffusers uses Qwen Image 2.1's official FlowMatch Euler scheduler. It is the
standalone pipeline equivalent, not ComfyUI's `KSampler` implementation, so the
same seed is not expected to produce a pixel-identical image.

## Faster repeat transfers

The app caches the last prompt plus its body/head references and two deterministic
VAE reference encodings in system RAM. Both reference images, their ordering,
and the prompt are part of the cache key. Changing the seed or LoRA strengths
does not require re-encoding unchanged references. No denoising steps are skipped.

With at least 22 GiB of free VRAM at model load and a canvas up to about 1 MP,
the default fast mode retains the image transformer and VAE on the GPU. The text
encoder runs only when needed and returns to system RAM. Larger canvases or the
unchecked **Keep swap model on GPU** option use component CPU offload; a busy GPU
at startup selects slower sequential offload. On a mostly free 24 GB card, larger
512-pixel VAE tiles reduce encoding/decoding overhead. Tiling can slightly change
pixels versus the old 256-pixel tiles; steps, resolution and adapters are unchanged.

First generation includes model loading. New references or a changed prompt are
slower than repeated transfers using the same inputs. The result status reports total
time, memory mode and cache reuse. Fast mode holds substantial GPU memory between
transfers: click **Release models / free GPU memory** before running another GPU app.
This clears memory only, never downloaded models or saved images. The next transfer
reloads the models. Restart the app to load code updates; no new model download is
needed for this speed update.

Local RTX 4090 verification (six steps, 864 x 1248 working canvas, 1728 x 2496 PNG):
the previous build took 32.8 seconds initially and 23.3 seconds warm. The updated
build took 22.9 seconds for initial conditioning and 8.7-8.8 seconds for repeated
inputs. A changed head reference took 20.8 seconds. These measurements exclude
model loading, include saving the PNG, and are not a guarantee for every image.
At a fixed seed, cached and uncached runs of the updated build were pixel-identical.

![Floyd Headliner — Likeness Transfer](assets/fh3.png)

## Requirements

- Windows 10/11 64-bit
- Python 3.10 or 3.11
- NVIDIA GPU with a recent driver; tested target is RTX 4090 24 GB
- Roughly 60 GB free disk space during setup
- About 64 GB system RAM recommended for bf16 CPU offload

The source setup above uses a private virtual environment in `.venv/` and
keeps downloaded models in `models/`.

## Model sources

- [Qwen Image 2.1 base model](https://huggingface.co/Qwen/Qwen-Image-2.1) — all repository model files belong in `models/Qwen-Image-2.1/`.
- [BFS identity LoRA](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/bfs_head_v1.1_alternative_qwen_2.1.safetensors) — save as `models/loras/bfs_head_v1.1_alternative_qwen_2.1.safetensors`.
- [Viggle Turbo six-step LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors) — save as `models/loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors`.

`download_models.py` pins exact upstream revisions recorded on September 29, 2026.
The [Qwen Image 2.1 base model](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)
and [Viggle Turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/LICENSE)
use the Qwen Research License. Their materials are limited to non-commercial
research/evaluation unless the licensor grants a separate commercial license.
The license for Floyd Headliner's code does not change those upstream terms.

Floyd Headliner is named in honor of the legend Count Floyd

## License

Floyd Headliner's own app code and documentation are available under the
[GNU General Public License, version 3 or later](LICENSE); the complete GPLv3
text is in [COPYING](COPYING). Third-party models, adapters, and dependencies
remain under their respective licenses.
