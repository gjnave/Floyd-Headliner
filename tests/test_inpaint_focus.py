import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from inpaint import editing_prompt, prepare_image, run_inpaint


class InpaintFocusTests(unittest.TestCase):
    def test_requires_image_and_prompt_with_readable_errors(self):
        with self.assertRaisesRegex(ValueError, "Upload an input image"):
            prepare_image(None)
        with self.assertRaisesRegex(ValueError, "4096"):
            prepare_image(Image.new("RGB", (4097, 1)))
        with self.assertRaisesRegex(ValueError, "Type what you want"):
            editing_prompt("  ")

    def test_prompt_only_edit_uses_original_image_and_disables_loras(self):
        original = Image.new("RGB", (64, 96), "blue")

        class Pipe:
            fast_resident = False
            prompt_cache_hit = False
            disabled = False

            def disable_lora(self):
                self.disabled = True

            def set_adapters(self, *args, **kwargs):
                raise AssertionError("Inpaint must not activate a LoRA")

            def __call__(self, **kwargs):
                if not self.disabled:
                    raise AssertionError("LoRA was active during generation")
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (32, 48), "red")])

        pipe = Pipe()
        with patch.object(app, "model_profile", return_value="low-vram"), \
             patch.object(app, "release_core_models"), \
             patch.object(app, "_load_pipeline", return_value=pipe), patch("torch.Generator"), \
             patch.object(Image.Image, "save"), patch("torch.cuda.empty_cache"):
            result = run_inpaint(app, original, "remove the hat", 42, False, 40, .5, False)

        self.assertEqual(len(pipe.request["image"]), 1)
        self.assertEqual(pipe.request["image"][0].tobytes(), original.tobytes())
        self.assertIn("remove the hat", pipe.request["prompt"])
        self.assertNotIn("<image2>", pipe.request["prompt"])
        self.assertEqual(pipe.request["num_inference_steps"], 40)
        self.assertEqual(result[4].size, original.size)
        self.assertEqual(result[4].getpixel((32, 48)), (255, 0, 0))
        self.assertIn("outside the requested object", result[2])

    def test_inpaint_ui_has_two_images_with_generate_directly_below(self):
        config = app.build_ui().get_config_file()
        components = config["components"]
        source = next(component for component in components
                      if component["props"].get("label") == "INPUT IMAGE")
        comparison = next(component for component in components
                          if component["type"] == "imageslider" and
                          "OUTPUT" in (component["props"].get("label") or ""))
        generate = next(component for component in components
                        if component["props"].get("elem_id") == "ggf-inpaint-button")
        self.assertEqual(source["type"], "image")

        def find_direct_parent(target, node):
            if any(child["id"] == target for child in node.get("children", [])):
                return node
            for child in node.get("children", []):
                found = find_direct_parent(target, child)
                if found:
                    return found
            return None

        image_row = find_direct_parent(source["id"], config["layout"])
        self.assertEqual(image_row["id"],
                         find_direct_parent(comparison["id"], config["layout"])["id"])
        container = find_direct_parent(image_row["id"], config["layout"])
        child_ids = [child["id"] for child in container["children"]]
        self.assertEqual(child_ids[child_ids.index(image_row["id"]) + 1], generate["id"])

        inpaint_event = next(event for event in config["dependencies"]
                             if event.get("api_name") == "inpaint")
        self.assertEqual(inpaint_event["inputs"][0], source["id"])
        labels = {component["props"].get("label") for component in components}
        self.assertNotIn("Change background in painted area", labels)
        self.assertNotIn("Guess from drawing", labels)
        self.assertNotIn("Preview edit area", labels)
        self.assertFalse(any(component["props"].get("elem_id") == "ggf-inpaint-canvas"
                             for component in components))
        self.assertFalse(any(event.get("api_name") == "guess_inpaint_prompt"
                             for event in config["dependencies"]))
        steps = next(component for component in components
                     if component["props"].get("label") == "Steps (base model)")
        self.assertEqual(steps["props"]["value"], 40)

    def test_turbo_inpaint_halves_working_dimensions_and_saves_native_size(self):
        import inpaint

        ui = app.build_ui()
        turbo = next(event for event in ui.fns.values()
                     if event.name == "generate_turbo_edit")
        original = Image.new("RGB", (512, 512), "blue")
        preview = Image.new("RGB", (256, 256), "red")
        response = ((original.resize(preview.size), preview), "out.png", "done", "42", preview)
        with patch.object(inpaint, "run_inpaint", return_value=response) as run:
            values = turbo.fn(original, "remove the hat", 42, False, 40, 1.0, True)
        self.assertEqual(run.call_args.args[6], .25)
        self.assertTrue(run.call_args.kwargs["turbo_preview"])
        self.assertIs(values[3], preview)


if __name__ == "__main__":
    unittest.main()
