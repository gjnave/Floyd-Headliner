import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from inpaint import editing_prompt


class InpaintFocusTests(unittest.TestCase):
    def test_blank_remove_on_person_keeps_subject_and_replaces_clothing(self):
        prompt = editing_prompt("", ["remove"], "Person")
        self.assertIn("Keep the person present", prompt)
        self.assertIn("replace it with natural clothing", prompt)
        self.assertIn("do not erase the person", prompt)

    def test_blank_remove_default_still_fills_surrounding_surface(self):
        prompt = editing_prompt("", ["remove"])
        self.assertIn("Remove those objects and reconstruct the natural background", prompt)
        self.assertNotIn("replace it with natural clothing", prompt)

    def test_focus_is_connected_to_generate_and_guess(self):
        config = app.build_ui().get_config_file()
        focus_id = next(component["id"] for component in config["components"]
                        if component["type"] == "radio" and component["props"].get("label") == "Focus")
        for name in ("inpaint", "guess_inpaint_prompt"):
            event = next(event for event in config["dependencies"] if event.get("api_name") == name)
            self.assertIn(focus_id, event["inputs"])


if __name__ == "__main__":
    unittest.main()
