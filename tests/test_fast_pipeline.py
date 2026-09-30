import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_pipeline import FastQwenImage21Pipeline, image_key, tensor_key
import app


class ConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        # No weights or CUDA are needed to test deterministic cache behavior.
        self.pipe = FastQwenImage21Pipeline.__new__(FastQwenImage21Pipeline)
        self.pipe.register_to_config()
        self.pipe._fast_resident = False
        self.body = Image.new("RGB", (32, 32), "red")
        self.head = Image.new("RGB", (32, 32), "blue")
        self.encoded = (torch.ones(1, 2, 3), None, torch.tensor([[True, False]]))

    def encode(self, prompt="head_swap", images=None, **kwargs):
        return self.pipe.encode_prompt(
            prompt, image=images or [self.body, self.head], device=torch.device("cpu"), **kwargs)

    def test_prompt_reuse_preserves_image_mask_and_copies_cache(self):
        with patch.object(QwenImage21Pipeline, "encode_prompt", return_value=self.encoded) as encode:
            first = self.encode()
            first[0].zero_()
            second = self.encode()
            self.assertEqual(encode.call_count, 1)
            self.assertTrue(self.pipe.prompt_cache_hit)
            self.assertTrue(torch.equal(second[0], torch.ones(1, 2, 3)))
            self.assertIsNone(second[1])
            self.assertTrue(torch.equal(second[2], torch.tensor([[True, False]])))
            second[2].zero_()
            self.assertTrue(self.encode()[2][0, 0])

    def test_either_image_prompt_or_order_invalidates_cache(self):
        with patch.object(QwenImage21Pipeline, "encode_prompt", return_value=self.encoded) as encode:
            self.encode()
            self.encode()
            self.body.putpixel((0, 0), (2, 3, 4))
            self.encode()
            self.head.putpixel((0, 0), (4, 3, 2))
            self.encode()
            self.encode(prompt="different prompt")
            self.encode(prompt="different prompt", images=[self.head, self.body])
            self.assertEqual(encode.call_count, 5)
            self.assertFalse(self.pipe.prompt_cache_hit)

    def test_supplied_embeddings_bypass_cache(self):
        with patch.object(QwenImage21Pipeline, "encode_prompt", return_value=self.encoded) as encode:
            self.encode()
            self.encode(prompt_embeds=torch.zeros(1, 2, 3), image_pad_mask=self.encoded[2])
            self.assertEqual(encode.call_count, 2)

    def test_vae_cache_is_deterministic_seed_independent_and_bounded(self):
        result = torch.ones(1, 4, 1, 2, 2)
        with patch.object(QwenImage21Pipeline, "_encode_vae_image", return_value=result) as encode:
            first = torch.zeros(1, 3, 1, 32, 32)
            second = first + 1
            third = first + 2
            self.pipe._encode_vae_image(first, torch.Generator().manual_seed(1))
            cached = self.pipe._encode_vae_image(first, torch.Generator().manual_seed(2))
            self.assertEqual(encode.call_count, 1)
            cached.zero_()
            self.assertTrue(self.pipe._encode_vae_image(first, None).all())
            self.pipe._encode_vae_image(second, None)
            self.pipe._encode_vae_image(third, None)
            self.assertEqual(len(self.pipe._vae_cache), 2)
            self.pipe._encode_vae_image(first, None)
            self.assertEqual(encode.call_count, 4)

    def test_clear_cache_forces_reencoding(self):
        with patch.object(QwenImage21Pipeline, "encode_prompt", return_value=self.encoded) as encode:
            self.encode()
            self.pipe.clear_conditioning_cache()
            self.encode()
            self.assertEqual(encode.call_count, 2)

    def test_keys_include_size_shape_dtype_and_pixels(self):
        self.assertNotEqual(image_key(self.body), image_key(self.head))
        self.assertNotEqual(tensor_key(torch.zeros(2, 2)), tensor_key(torch.zeros(1, 4)))
        self.assertNotEqual(tensor_key(torch.zeros(2, 2)), tensor_key(torch.zeros(2, 2).bfloat16()))

    def test_resident_mode_does_not_automatically_offload(self):
        with patch.object(QwenImage21Pipeline, "maybe_free_model_hooks") as offload:
            self.pipe._fast_resident = True
            self.pipe.maybe_free_model_hooks()
            offload.assert_not_called()
            self.pipe._fast_resident = False
            self.pipe.maybe_free_model_hooks()
            offload.assert_called_once()

    def test_large_canvas_or_disabled_checkbox_uses_offload(self):
        from unittest.mock import Mock
        for enabled, width, height in [(False, 1024, 1024), (True, 2048, 1024)]:
            pipe = Mock(fast_resident=True)
            with patch.object(app, "_RESIDENT_ALLOWED", True):
                app.configure_speed_mode(pipe, enabled, width, height)
            pipe.disable_fast_residency.assert_called_once()
            pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=0)


if __name__ == "__main__":
    unittest.main()
