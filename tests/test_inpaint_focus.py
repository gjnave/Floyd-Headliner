import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from inpaint import editing_prompt, image_data_url, prepare_edit, run_inpaint


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
        self.assertIn("Automatic edit instruction", result[2])
        self.assertNotEqual(result[4].getpixel((32, 32)), (0, 0, 255))

    def test_tool_overlay_colors_are_not_model_color_instructions(self):
        payload = {"image": image_data_url(Image.new("RGB", (80, 80), "blue")),
                   "strokes": [{"tool": "add", "size": 12, "color": "#00e676", "points": [[.25, .3]]},
                               {"tool": "remove", "size": 12, "color": "#ff2d55", "points": [[.75, .7]]}]}
        _, guide, _, _, modes = prepare_edit(payload, 0, False)
        self.assertEqual(modes, ["add", "remove"])
        for point in ((20, 24), (59, 55)):
            red, green, blue = guide.getpixel(point)
            self.assertEqual(red, green)
            self.assertEqual(green, blue)
        prompt = editing_prompt("Replace the selected object", modes)
        self.assertNotIn("red/pink", prompt)
        self.assertNotIn("bright green", prompt)

    def test_add_only_sends_monochrome_shape_not_green_guide(self):
        payload = {"image": image_data_url(Image.new("RGB", (64, 64), "blue")),
                   "strokes": [{"tool": "add", "size": 12, "color": "#00e676",
                                "points": [[.5, .5]]}]}
        class Pipe:
            fast_resident = False
            prompt_cache_hit = False
            def set_adapters(self, *args, **kwargs): pass
            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), "white")])
        pipe = Pipe()
        with patch.object(app, "_load_pipeline", return_value=pipe), patch("torch.Generator"), \
             patch.object(Image.Image, "save"), patch("torch.cuda.empty_cache"), \
             patch("inpaint.suggest_prompt", return_value="Add a small apple") as guess:
            result = run_inpaint(app, payload, "", 42, False, 6, .5, 0, 0, False, False)
        guess.assert_called_once()
        self.assertEqual(len(pipe.request["image"]), 2)
        self.assertIn("black-and-white selection mask", pipe.request["prompt"])
        self.assertIn("Add a small apple", pipe.request["prompt"])
        self.assertNotIn("green", pipe.request["prompt"])
        self.assertEqual(pipe.request["image"][1].getpixel((32, 32)), (255, 255, 255))
        self.assertIn("Automatic edit instruction", result[2])

    def test_color_tool_keeps_explicitly_chosen_output_color(self):
        payload = {"image": image_data_url(Image.new("RGB", (64, 64), "blue")),
                   "strokes": [{"tool": "color", "size": 12, "color": "#6633ff",
                                "points": [[.5, .5]]}]}
        _, guide, _, _, modes = prepare_edit(payload, 0, False)
        self.assertEqual(modes, ["color"])
        self.assertEqual(guide.getpixel((32, 32)), (102, 51, 255))
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

    def test_mask_change_clears_prompt_unless_kept_and_output_is_beside_canvas(self):
        ui = app.build_ui()
        config = ui.get_config_file()
        components = {component["id"]: component for component in config["components"]}
        canvas = next(component for component in config["components"]
                      if component["props"].get("elem_id") == "ggf-inpaint-canvas")
        comparison = next(component for component in config["components"]
                          if component["type"] == "imageslider" and "original left, edit right" in
                          (component["props"].get("label") or ""))
        keep = next(component for component in config["components"]
                    if component["props"].get("label") == "Keep prompt")
        prompt = next(component for component in config["components"]
                      if component["props"].get("label") == "What should change?")
        self.assertFalse(keep["props"]["value"])
        clear = next(event for event in config["dependencies"]
                     if event["targets"] == [(canvas["id"], "change")])
        self.assertEqual(clear["inputs"], [keep["id"], prompt["id"]])
        self.assertEqual(clear["outputs"], [prompt["id"]])
        self.assertEqual(clear["js"], "(keep, text) => keep ? text : ''")
        def parent_of(target, node):
            for child in node.get("children", []):
                if child["id"] == target:
                    return node["id"]
                found = parent_of(target, child)
                if found is not None:
                    return found
            return None
        self.assertEqual(parent_of(canvas["id"], config["layout"]),
                         parent_of(comparison["id"], config["layout"]))
        generate = next(event for event in ui.fns.values() if event.name == "generate_edit")
        self.assertEqual(components[generate.outputs[1]._id]["type"], "downloadbutton")
        generate_event = next(event for event in config["dependencies"]
                              if event.get("api_name") == "inpaint")
        self.assertEqual(generate_event["show_progress_on"], [comparison["id"]])
        self.assertFalse(any(components[output]["props"].get("label") == "Used seed"
                             for output in generate_event["outputs"]))


if __name__ == "__main__":
    unittest.main()
