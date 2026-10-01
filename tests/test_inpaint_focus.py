import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from inpaint import editing_prompt, image_data_url, run_inpaint


class InpaintFocusTests(unittest.TestCase):
    def test_blank_remove_identifies_object_and_never_sends_pink_guide(self):
        payload = {"image": image_data_url(Image.new("RGB", (64, 64), "blue")),
                   "strokes": [{"tool":"remove", "size":10, "color":"#ff2d55",
                                "points":[[.4,.4],[.6,.6]]}]}
        class Pipe:
            fast_resident = False
            prompt_cache_hit = False
            def set_adapters(self, *args, **kwargs): pass
            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (64,64), "white")])
        pipe = Pipe()
        with patch.object(app, "_load_pipeline", return_value=pipe), patch("torch.Generator"), \
             patch.object(Image.Image, "save"), patch("torch.cuda.empty_cache"), \
             patch("inpaint.suggest_prompt", return_value="Remove the cup.") as guess:
            result = run_inpaint(app, payload, "", 42, False, 6, .5, 0, 0, True, False)
        guess.assert_called_once()
        self.assertIn("Remove the cup.", pipe.request["prompt"])
        self.assertEqual(len(pipe.request["image"]), 2)
        self.assertTrue(all(r == g == b for r,g,b in pipe.request["image"][1].getdata()))
        self.assertIn("Automatic Remove instruction", result[2])
    def test_remove_only_has_no_colored_guide_instruction(self):
        prompt = editing_prompt("Remove the cup", ["remove"])
        self.assertIn("black-and-white selection mask", prompt)
        self.assertNotIn("red/pink", prompt)
        self.assertNotIn("<image3>", prompt)
        self.assertIn("Remove the cup", prompt)
    def test_blank_remove_keeps_subject_and_replaces_clothing(self):
        prompt = editing_prompt("", ["remove"])
        self.assertIn("Keep people present", prompt)
        self.assertIn("replace it with natural clothing", prompt)
        self.assertIn("do not erase the person", prompt)

    def test_default_preserves_existing_background_but_fills_removed_object(self):
        prompt = editing_prompt("", ["remove"])
        self.assertIn("Do not change the original background or setting", prompt)
        self.assertIn("reconstruct only the background that was hidden", prompt)

    def test_background_changes_require_checkbox(self):
        default = editing_prompt("Make the sky sunset orange", ["color"])
        allowed = editing_prompt("Make the sky sunset orange", ["color"], True)
        self.assertIn("Do not change the original background", default)
        self.assertTrue(default.endswith("only restore matching background where a removed object previously covered it."))
        self.assertIn("enabled background changes", allowed)
        self.assertNotIn("Do not change the original background", allowed)

    def test_background_checkbox_is_off_and_connected_to_generate_and_guess(self):
        config = app.build_ui().get_config_file()
        control = next(component for component in config["components"]
                       if component["props"].get("label") == "Change background in painted area")
        self.assertEqual(control["type"], "checkbox")
        self.assertFalse(control["props"]["value"])
        self.assertFalse(any(component["props"].get("label") == "Focus"
                             for component in config["components"]))
        for name in ("inpaint", "guess_inpaint_prompt"):
            event = next(event for event in config["dependencies"] if event.get("api_name") == name)
            self.assertIn(control["id"], event["inputs"])


if __name__ == "__main__":
    unittest.main()
