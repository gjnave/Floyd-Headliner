"""Exact conditioning reuse and optional GPU residency for the pinned pipeline.

Only deterministic prompt/reference encodings are cached, never generated noise,
denoising steps, or transformer KV state. Both adapters remain independently
adjustable. CPU caches hold one prompt+image pair and at most two VAE encodings.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch
from diffusers import QwenImage21Pipeline


def image_key(image):
    return (image.mode, image.size, hashlib.sha256(image.tobytes()).digest())


def tensor_key(tensor):
    raw = tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return (tuple(tensor.shape), str(tensor.dtype), hashlib.sha256(raw).digest())


def copy_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, tuple):
        return tuple(copy_tensors(item, device) for item in value)
    return value


class FastQwenImage21Pipeline(QwenImage21Pipeline):
    # Inherit __init__ unchanged: Diffusers inspects its component signature.
    @property
    def fast_resident(self):
        return getattr(self, "_fast_resident", False)

    @property
    def _execution_device(self):
        if self.fast_resident:
            return torch.device("cuda:0")
        return super()._execution_device

    def enable_fast_residency(self):
        if self.fast_resident:
            return
        self.remove_all_hooks()
        self.to("cpu")
        self._fast_resident = True
        # Actual GPU transfers are lazy, in encode_prompt / prepare_latents.

    def disable_fast_residency(self):
        if self.fast_resident:
            self.to("cpu")
            self._fast_resident = False
            torch.cuda.empty_cache()

    def clear_conditioning_cache(self):
        self._prompt_cache = None
        self._vae_cache = OrderedDict()

    def maybe_free_model_hooks(self):
        if not self.fast_resident:
            return super().maybe_free_model_hooks()
        # Keep the transformer and VAE in VRAM until settings/input require
        # moving them, the user releases models, or the app exits.

    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None):
        device = device or self._execution_device
        key = (
            tuple([prompt] if isinstance(prompt, str) else (prompt or [])),
            tuple(image_key(item) for item in (image or [])),
            num_images_per_prompt,
        )
        cached = getattr(self, "_prompt_cache", None)
        if prompt_embeds is None and cached is not None and cached[0] == key:
            self.prompt_cache_hit = True
            return copy_tensors(cached[1], device)
        self.prompt_cache_hit = False
        if self.fast_resident:
            # The text encoder and transformer cannot coexist on a 24 GB card.
            self.transformer.to("cpu")
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            self.text_encoder.to(device)
        try:
            result = super().encode_prompt(
                prompt, image, device, num_images_per_prompt,
                prompt_embeds, prompt_embeds_mask, image_pad_mask,
            )
            if prompt_embeds is None:
                # Preserve all three values, including the image-token mask.
                self._prompt_cache = (key, copy_tensors(result, "cpu"))
            return result
        finally:
            if self.fast_resident:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

    def _encode_vae_image(self, image, generator):
        cache = getattr(self, "_vae_cache", None)
        if cache is None:
            self._vae_cache = cache = OrderedDict()
        key = tensor_key(image)
        if key in cache:
            cache.move_to_end(key)
            return copy_tensors(cache[key], image.device)
        if self.fast_resident:
            self.vae.to(image.device)
        # The pinned upstream implementation uses posterior argmax, not random
        # sampling. Reuse therefore does not consume or change the seed's RNG.
        result = super()._encode_vae_image(image, generator)
        cache[key] = copy_tensors(result, "cpu")
        while len(cache) > 2:
            cache.popitem(last=False)
        return result

    def prepare_latents(self, *args, **kwargs):
        result = super().prepare_latents(*args, **kwargs)
        if self.fast_resident:
            self.vae.to("cuda:0")
            self.transformer.to("cuda:0")
        return result
