import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from PIL import Image


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("bfs_swap_app", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APP)


class AppHelperTests(unittest.TestCase):
    def test_reuse_result_preserves_pixels_and_clears_selection(self):
        result = Image.new("RGB", (80, 100), "blue")
        editor, selected = APP.result_as_body_input(result)
        self.assertFalse(selected)
        self.assertEqual(editor["layers"], [])
        self.assertFalse(APP.has_painted_mask(editor))
        for key in ("background", "composite"):
            self.assertEqual(editor[key].size, result.size)
            self.assertEqual(editor[key].tobytes(), result.tobytes())
            self.assertIsNot(editor[key], result)
        editor["background"].putpixel((0, 0), (255, 0, 0))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))

    def test_reuse_requires_successful_result(self):
        with self.assertRaisesRegex(ValueError, "Generate a likeness result first"):
            APP.result_as_body_input(None)

    def test_reuse_button_receives_latest_generated_image_from_session_state(self):
        import gradio as gr
        with gr.Blocks() as ui:
            APP.build_likeness_ui(gr)
        generate = next(event for event in ui.fns.values() if event.name == "generate_transfer")
        reuse = next(event for event in ui.fns.values() if event.name == "result_as_body_input")
        clear = next(event for event in ui.fns.values()
                     if event.name == "<lambda>" and event.outputs == [reuse.outputs[0]])
        self.assertIsNone(clear.fn())
        self.assertIs(generate.outputs[4], reuse.inputs[0])
        self.assertIsInstance(reuse.inputs[0], gr.State)
        self.assertFalse(generate.outputs[5].interactive)
        self.assertFalse(generate.outputs[6].interactive)
        self.assertIs(reuse.outputs[0], generate.inputs[0])
        self.assertIs(reuse.outputs[1], generate.inputs[12])
        original = Image.new("RGB", (20, 20), "blue")
        for color in ("red", "green"):
            result = Image.new("RGB", (20, 20), color)
            with patch.object(APP, "run_swap_ui", return_value=((original, result), "out.png", "done", "42")):
                values = generate.fn()
            self.assertIs(values[4], result)
            self.assertTrue(values[5]["interactive"])
            self.assertTrue(values[6]["interactive"])
            editor, selected = reuse.fn(values[4])
            self.assertEqual(editor["background"].tobytes(), result.tobytes())
            self.assertFalse(selected)

    def test_head_upload_is_reasserted_and_result_can_return_from_inpaint(self):
        import sys

        with patch.dict(sys.modules, {APP.__name__: APP}):
            ui = APP.build_ui()
        upload = next(event for event in ui.fns.values()
                      if event.name == "retain_uploaded_head")
        head = upload.inputs[0]
        self.assertIs(head, upload.outputs[0])
        image = Image.new("RGB", (40, 30), "blue")
        retained = upload.fn(image)
        self.assertEqual(retained.tobytes(), image.tobytes())
        self.assertIsNot(retained, image)

        open_likeness = next(event for event in ui.fns.values()
                             if event.name == "open_likeness")
        select_likeness = next(event for event in ui.fns.values()
                               if event.name == "select_likeness")
        clear_likeness_input = next(event for event in ui.fns.values()
                                    if event.name == "clear_likeness_input")
        inpaint_generate = next(event for event in ui.fns.values()
                                if event.name == "generate_edit")
        self.assertIs(open_likeness.inputs[0], inpaint_generate.outputs[3])
        self.assertIs(select_likeness.inputs[0], inpaint_generate.outputs[3])
        self.assertIs(clear_likeness_input.outputs[0], open_likeness.outputs[0])
        dependencies = {event["id"]: event for event in ui.config["dependencies"]}
        self.assertEqual(dependencies[clear_likeness_input._id]["trigger_after"],
                         select_likeness._id)
        self.assertEqual(dependencies[open_likeness._id]["trigger_after"],
                         clear_likeness_input._id)
        self.assertTrue(dependencies[clear_likeness_input._id]["trigger_only_on_success"])
        self.assertTrue(dependencies[open_likeness._id]["trigger_only_on_success"])
        self.assertEqual(select_likeness.fn(image)["selected"], "likeness")
        self.assertIsNone(clear_likeness_input.fn())
        editor, mask_enabled = open_likeness.fn(image)
        self.assertEqual(editor["background"].tobytes(), image.tobytes())
        self.assertFalse(mask_enabled)
        with self.assertRaisesRegex(ValueError, "Generate an inpaint result first"):
            select_likeness.fn(None)

    def test_likeness_turbo_button_is_above_full_size_button(self):
        import gradio as gr

        with gr.Blocks() as ui:
            APP.build_likeness_ui(gr)
        buttons = [component for component in ui.blocks.values()
                   if isinstance(component, gr.Button)]
        turbo = next(index for index, button in enumerate(buttons)
                     if button.elem_id == "ggf-swap-turbo")
        full_size = next(index for index, button in enumerate(buttons)
                         if button.elem_id == "ggf-swap-button")
        self.assertLess(turbo, full_size)

    def test_all_profiles_show_turbo_buttons_and_running_mode(self):
        import sys

        labels = {
            "standard": "Standard · 24 GB",
            "low-vram": "Low VRAM · 16 GB target",
            "low-vram-12gb": "Experimental · 12 GB target",
        }
        with patch.dict(sys.modules, {APP.__name__: APP}):
            for profile, label in labels.items():
                with self.subTest(profile=profile), patch.dict(os.environ, {"FLOYD_HEADLINER_PROFILE": profile}):
                    ui = APP.build_ui()
                    components = {component["props"].get("elem_id"): component["props"]
                                  for component in ui.config["components"]}
                    self.assertTrue(components["ggf-swap-turbo"]["visible"])
                    self.assertTrue(components["ggf-inpaint-turbo"]["visible"])
                    self.assertIn(label, str(ui.config["components"]))

    def test_experimental_12gb_turbo_still_reduces_working_pixels(self):
        with patch.dict(os.environ, {"FLOYD_HEADLINER_PROFILE": "low-vram-12gb"}):
            image = Image.new("RGB", (1024, 1024))
            full = APP.calculate_working_size(image, .5)
            turbo = APP.calculate_working_size(image, APP.turbo_megapixels(.5))
        self.assertLess(turbo[0] * turbo[1], full[0] * full[1])

    def test_update_check_compares_release_markers_without_installing(self):
        self.assertEqual(APP.parse_release_version("2026.10.02.1"), (2026, 10, 2, 1))
        with self.assertRaises(ValueError):
            APP.parse_release_version("<script>bad</script>")
        with patch.object(APP, "VERSION_FILE", SimpleNamespace(read_text=lambda **_: "2026.10.02.1")):
            with patch.object(APP.urllib.request, "urlopen", return_value=io.BytesIO(b"2026.10.02.1\n")):
                message, available = APP.check_for_updates()
            self.assertFalse(available)
            self.assertIn("Up to date", message)
            with patch.object(APP.urllib.request, "urlopen", return_value=io.BytesIO(b"2026.10.03.1\n")):
                message, available = APP.check_for_updates()
            self.assertTrue(available)
            self.assertIn("3-UPDATE-Floyd-Headliner.bat", message)
            with patch.object(APP.urllib.request, "urlopen", side_effect=TimeoutError):
                message, available = APP.check_for_updates()
            self.assertFalse(available)
            self.assertIn("works offline", message)

    def test_update_controls_logo_and_output_targets_are_in_ui(self):
        import sys
        import gradio as gr

        with patch.dict(sys.modules, {APP.__name__: APP}):
            ui = APP.build_ui()
        components = {component["props"].get("elem_id"): component["props"]
                      for component in ui.config["components"]}
        self.assertFalse(components["ggf-update-button"]["interactive"])
        self.assertIn("ggf-check-updates", components)
        self.assertIn("ggf-likeness-output", components)
        self.assertIn("ggf-inpaint-output", components)
        self.assertIn("data:image/png;base64,", APP.hero_logo_html())
        self.assertIn("Your Time Is Limited.", str(ui.config["components"]))
        check = next(event for event in ui.fns.values() if event.name == "check_updates_ui")
        with patch.object(APP, "check_for_updates", return_value=("Update available", True)), \
                patch.object(gr, "Info") as info:
            status, update = check.fn()
        self.assertEqual(status, "Update available")
        self.assertTrue(update["interactive"])
        info.assert_called_once()

    def test_turbo_preview_quarters_working_pixel_budget_without_changing_steps(self):
        import gradio as gr
        with gr.Blocks() as ui:
            APP.build_likeness_ui(gr)
        turbo = next(event for event in ui.fns.values()
                     if event.name == "generate_turbo_transfer")
        inputs = [None] * len(turbo.inputs)
        inputs[7] = 6
        inputs[8] = 1.0
        inputs[9] = 2
        with patch.object(APP, "run_swap_ui", return_value=((None, None), "out.png", "done", "42")) as run:
            turbo.fn(*inputs)
        submitted = run.call_args.args
        self.assertEqual(submitted[7], 6)
        self.assertEqual(submitted[8], .25)
        self.assertEqual(submitted[9], 1)
        self.assertTrue(run.call_args.kwargs["turbo_preview"])

    def test_inpaint_button_receives_latest_likeness_result(self):
        import sys
        import gradio as gr
        with patch.dict(sys.modules, {APP.__name__: APP}):
            ui = APP.build_ui()
        generate = next(event for event in ui.fns.values() if event.name == "generate_transfer")
        open_inpaint = next(event for event in ui.fns.values() if event.name == "open_inpaint")
        self.assertIs(generate.outputs[4], open_inpaint.inputs[0])
        self.assertIsInstance(open_inpaint.inputs[0], gr.State)
        result = Image.new("RGB", (20, 20), "green")
        input_value, tab_update, *reset = open_inpaint.fn(result)
        self.assertIs(input_value, result)
        self.assertEqual(tab_update["selected"], "inpaint")
        self.assertIsNone(reset[0])
        self.assertIsNone(reset[1])

    def test_protect_hides_reference_and_restores_original_pixels(self):
        original = Image.new("RGB", (100, 80), "blue")
        painted = Image.new("L", original.size)
        painted.paste(255, (10, 10, 30, 30))
        reference, protected = APP.protected_reference(original, painted)
        self.assertEqual(reference.getpixel((20, 20)), (127, 127, 127))
        self.assertEqual(reference.getpixel((80, 60)), original.getpixel((80, 60)))
        result = APP.composite_selection(original, Image.new("RGB", original.size, "red"),
                                        APP.ImageOps.invert(protected), (0, 0, 100, 80), 8)
        self.assertEqual(result.getpixel((20, 20)), original.getpixel((20, 20)))
        self.assertEqual(result.getpixel((80, 60)), (255, 0, 0))
        delta = APP.ImageChops.difference(original, result)
        self.assertIsNone(APP.ImageChops.multiply(delta, protected.convert("RGB")).getbbox())

    def test_protect_focuses_opposite_side_for_two_people(self):
        original = Image.new("RGBA", (200, 100), "blue")
        class FakePipe:
            fast_resident = False
            prompt_cache_hit = False
            def enable_lora(self): pass
            def set_adapters(self, *args, **kwargs): pass
            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), "red")])
        for protected_box, untouched, changed in [
            ((15, 5, 65, 85), (35, 40), (165, 40)),
            ((135, 5, 185, 85), (165, 40), (35, 40)),
        ]:
            layer = Image.new("RGBA", original.size)
            layer.paste((245, 185, 66, 255), protected_box)
            pipe = FakePipe()
            with patch.object(APP, "_load_pipeline", return_value=pipe), \
                    patch("torch.Generator"), patch.object(Image.Image, "save"), \
                    patch("torch.cuda.empty_cache"):
                result, _, status, _ = APP.run_swap(
                    {"background": original, "layers": [layer]}, Image.new("RGB", (32, 32)),
                    APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 2, False,
                    selected_only=True, mask_mode="Protect painted area (experimental)", feather=0)
            self.assertLess(pipe.request["image"][0].width, original.width)
            self.assertEqual(result.size, original.size)
            self.assertEqual(result.getpixel(untouched), (0, 0, 255))
            self.assertEqual(result.getpixel(changed), (255, 0, 0))
            self.assertIn("unprotected side", status)

    def test_protect_mode_rejects_missing_mask(self):
        with self.assertRaisesRegex(ValueError, "Paint over the head to protect"):
            APP.run_swap(Image.new("RGB", (100, 80)), Image.new("RGB", (32, 32)),
                         APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 1,
                         mask_mode="Protect painted area (experimental)")
    def test_generate_editor_preprocessing_retries_incomplete_png(self):
        expected = {"background": Image.new("RGBA", (20, 20)), "layers": []}
        original = Mock(side_effect=[OSError("image file is truncated"),
                                    SyntaxError("broken PNG file"), expected])
        editor = SimpleNamespace(preprocess=original)
        APP.protect_editor_preprocessing(editor)
        payload = object()
        with patch.object(APP.time, "sleep") as sleep:
            self.assertIs(editor.preprocess(payload), expected)
        self.assertEqual(original.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_generate_editor_preprocessing_rejects_corrupt_image(self):
        import gradio as gr
        editor = SimpleNamespace(preprocess=Mock(side_effect=OSError("image file is truncated")))
        APP.protect_editor_preprocessing(editor)
        with patch.object(APP.time, "sleep"), self.assertRaises(gr.Error) as error:
            editor.preprocess(object())
        self.assertIn("re-upload", str(error.exception))
        self.assertEqual(editor.preprocess.__closure__[0].cell_contents.call_count, 5)

    def test_oversized_head_is_rejected_before_gradio_decodes_it(self):
        import gradio as gr
        head = SimpleNamespace(preprocess=Mock())
        APP.protect_head_preprocessing(head)
        with patch("upload_safety.validate_component_images", side_effect=ValueError("Image is too large")):
            with self.assertRaisesRegex(gr.Error, "Image is too large"):
                head.preprocess(object())
        head.preprocess.__closure__[0].cell_contents.assert_not_called()

    def test_oversized_body_is_rejected_before_gradio_decodes_it(self):
        import gradio as gr
        body = SimpleNamespace(preprocess=Mock())
        APP.protect_editor_preprocessing(body)
        with patch("upload_safety.validate_component_images", side_effect=ValueError("Image is too large")):
            with self.assertRaisesRegex(gr.Error, "Image is too large"):
                body.preprocess(object())
        body.preprocess.__closure__[0].cell_contents.assert_not_called()

    def check_startup(self, profiles, current, answer=None, requested=None):
        profile_file = Mock()
        with patch.object(APP, "available_startup_profiles", return_value=profiles), \
             patch.object(APP, "model_profile", return_value=current), \
             patch.object(APP, "PROFILE_FILE", profile_file), \
             patch.dict(APP.os.environ, {}, clear=True), \
             patch.object(APP.sys.stdin, "isatty", return_value=True), \
             patch("importlib.util.find_spec", return_value=object()), \
             patch("builtins.input", side_effect=answer) as prompt:
            result = APP.select_startup_profile(requested)
        return result, profile_file, prompt

    def test_startup_menu_switches_without_install(self):
        result, file, prompt = self.check_startup(
            ["standard", "low-vram", "low-vram-12gb"], "standard", ["invalid", "3"])
        self.assertTrue(result)
        self.assertEqual(prompt.call_count, 2)
        file.write_text.assert_called_once_with("low-vram-12gb\n", encoding="utf-8")

    def test_startup_enter_keeps_last_mode(self):
        result, file, _ = self.check_startup(
            ["standard", "low-vram", "low-vram-12gb"], "low-vram", [""])
        self.assertTrue(result)
        file.write_text.assert_called_once_with("low-vram\n", encoding="utf-8")

    def test_startup_single_variant_automatic(self):
        for profiles, current, expected in [(["standard"], "low-vram", "standard"),
                (["low-vram", "low-vram-12gb"], "low-vram-12gb", "low-vram-12gb"),
                (["low-vram", "low-vram-12gb"], "standard", "low-vram")]:
            result, file, prompt = self.check_startup(profiles, current)
            self.assertTrue(result)
            prompt.assert_not_called()
            file.write_text.assert_called_once_with(expected + "\n", encoding="utf-8")

    def test_startup_missing_or_unavailable_models_do_not_change_profile(self):
        for profiles, requested in [([], None), (["standard"], "low-vram")]:
            result, file, prompt = self.check_startup(profiles, "standard", requested=requested)
            self.assertFalse(result)
            file.write_text.assert_not_called()
            prompt.assert_not_called()

    def test_twelve_gb_profile_limits_canvas_and_selects_small_adapter(self):
        with patch.object(APP, "model_profile", return_value="low-vram-12gb"):
            self.assertEqual(APP.selected_turbo_lora(), APP.LOW_TURBO_LORA)
            self.assertEqual(APP.calculate_working_size(Image.new("RGB", (512, 512)), 2), (704, 704))

    def test_mask_checkbox_tracks_paint_and_erasure(self):
        background = Image.new("RGBA", (100, 80), "blue")
        empty = Image.new("RGBA", background.size)
        painted = empty.copy()
        painted.putpixel((12, 15), (245, 185, 66, 255))
        self.assertFalse(APP.has_painted_mask(None))
        self.assertFalse(APP.has_painted_mask({"background": background, "layers": []}))
        self.assertFalse(APP.has_painted_mask({"background": background, "layers": [empty]}))
        self.assertTrue(APP.has_painted_mask({"background": background, "layers": [painted]}))
        painted.putpixel((12, 15), (0, 0, 0, 0))
        self.assertFalse(APP.has_painted_mask({"background": background, "layers": [painted]}))

    def test_stray_paint_without_explicit_mask_uses_full_body(self):
        body = Image.new("RGBA", (200, 100), "blue")
        layer = Image.new("RGBA", body.size)
        layer.putpixel((4, 5), (245, 185, 66, 255))

        class FakePipe:
            prompt_cache_hit = False

            def enable_lora(self): pass
            def set_adapters(self, *args, **kwargs): pass
            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), "red")])

        pipe = FakePipe()
        with patch.object(APP, "_load_pipeline", return_value=pipe), \
                patch.object(APP, "configure_speed_mode"), patch("torch.Generator"), \
                patch.object(Image.Image, "save"), patch("torch.cuda.empty_cache"):
            result, _, status, _ = APP.run_swap(
                {"background": body, "layers": [layer]}, Image.new("RGB", (32, 32)),
                APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 1, False,
                selected_only=False,
            )
        self.assertEqual(pipe.request["image"][0].size, body.size)
        self.assertEqual(result.getpixel((50, 50)), (255, 0, 0))
        self.assertNotIn("Painted-area edit", status)

    def test_fast_core_ignores_unchecked_paint(self):
        import core_runtime

        body = Image.new("RGBA", (200, 100), "blue")
        layer = Image.new("RGBA", body.size)
        layer.putpixel((4, 5), (245, 185, 66, 255))
        submitted_sizes = []

        def cached_input(image, output_dir):
            submitted_sizes.append(image.size)
            return Path("unused-test-image.png")

        response = {
            "output": "unused-test-result.png", "working_size": (64, 64),
            "conditioning_cache_hit": False,
        }
        with patch.object(core_runtime, "available", return_value=True), \
                patch.object(core_runtime, "_cached_input", side_effect=cached_input), \
                patch.object(core_runtime.WORKER, "request", return_value=response), \
                patch.object(APP, "_PIPELINE", None), \
                patch.object(Image, "open", return_value=Image.new("RGB", (64, 64), "red")), \
                patch.object(Image.Image, "save"):
            result, _, status, _ = core_runtime.run_swap(
                APP, {"background": body, "layers": [layer]}, Image.new("RGB", (32, 32)),
                APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 1, True, "", False, 8,
                "Edit painted area",
            )
        self.assertEqual(submitted_sizes[0], body.size)
        self.assertEqual(result.size, (64, 64))
        self.assertNotIn("Painted-area result", status)

    def test_upload_and_edit_do_not_consume_generation_input(self):
        import gradio as gr
        from gradio.components.image_editor import EditorData, EditorDataBlobs

        with gr.Blocks() as ui:
            APP.build_likeness_ui(gr)
        generate = next(event for event in ui.fns.values() if event.name == "generate_transfer")
        editor = generate.inputs[0]
        for dependency in ui.config["dependencies"]:
            for target, event_name in dependency["targets"]:
                if target == editor._id:
                    self.assertNotIn(event_name, ("change", "input", "edit"))
                    self.assertEqual(dependency["inputs"], [])
        stream = io.BytesIO()
        Image.new("RGBA", (32, 24), "blue").save(stream, format="PNG")
        editor.blob_storage["mobile-upload"] = EditorDataBlobs(
            background=stream.getvalue(), layers=[], composite=stream.getvalue(),
        )
        value = editor.preprocess(EditorData(id="mobile-upload"))
        self.assertEqual(value["background"].size, (32, 24))
        self.assertEqual(value["background"].getpixel((0, 0))[:3], (0, 0, 255))
        # A consumed/expired upload must report a useful error, not an empty image.
        with self.assertRaisesRegex(gr.Error, "upload is not ready or has expired"):
            editor.preprocess(EditorData(id="mobile-upload"))

    def test_editor_ignores_paint_for_reference(self):
        body = Image.new("RGBA", (100, 80), "blue")
        original, mask, crop = APP.prepare_selection({"background": body, "composite": Image.new("RGB", body.size, "red")})
        self.assertEqual(original.getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(mask)
        self.assertIsNone(crop)

    def test_selection_requires_paint(self):
        with self.assertRaisesRegex(ValueError, "Paint over"):
            APP.prepare_selection({"background": Image.new("RGBA", (100, 80)), "layers": []}, True)

    def test_mask_crop_and_unpainted_pixels_preserved(self):
        body = Image.new("RGBA", (200, 100), "blue")
        layer = Image.new("RGBA", body.size)
        layer.paste((255, 190, 0, 255), (10, 10, 40, 50))
        original, mask, crop = APP.prepare_selection({"background": body, "layers": [layer]}, True)
        self.assertEqual(mask.getbbox(), (10, 10, 40, 50))
        self.assertEqual(crop, (0, 0, 72, 82))
        for softness in (0, 8, 32):
            result = APP.composite_selection(original, Image.new("RGB", (64, 64), "red"), mask, crop, softness)
            self.assertEqual(result.size, body.size)
            for y in range(body.height):
                for x in range(body.width):
                    if mask.getpixel((x, y)) == 0:
                        self.assertEqual(result.getpixel((x, y)), original.getpixel((x, y)))
            self.assertNotEqual(result.getpixel((25, 30)), original.getpixel((25, 30)))

    def test_selection_rejects_misaligned_layer(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            APP.prepare_selection({"background": Image.new("RGBA", (100, 80)), "layers": [Image.new("RGBA", (50, 40))]}, True)

    def test_comparison_uses_submitted_original_and_exact_seed(self):
        body = Image.new("RGBA", (100, 80), "blue")
        result = Image.new("RGB", (200, 160), "red")
        with patch.object(APP, "run_swap", return_value=(result, "result.png", "done", 7869839904662545972)):
            pair, path, status, seed = APP.run_swap_ui({"background": body})
        self.assertEqual(pair[0].size, pair[1].size)
        self.assertEqual(pair[0].getpixel((0, 0)), (0, 0, 255))
        self.assertIs(pair[1], result)
        self.assertEqual(seed, "7869839904662545972")

    def test_selected_generation_crops_and_saves_composite_not_crop(self):
        body = Image.new("RGBA", (300, 200), "blue")
        layer = Image.new("RGBA", body.size)
        layer.paste((255, 190, 0, 255), (30, 30, 70, 80))
        class FakePipe:
            fast_resident = False
            prompt_cache_hit = False
            def enable_lora(self):
                self.lora_enabled = True
            def set_adapters(self, *args, **kwargs): pass
            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (64, 64), "red")])
        pipe = FakePipe()
        saved = []
        def save(image, *args, **kwargs): saved.append(image.copy())
        with patch.object(APP, "_load_pipeline", return_value=pipe), \
                patch("torch.Generator"), patch.object(Image.Image, "save", save), \
                patch("torch.cuda.empty_cache"):
            result, _, status, _ = APP.run_swap(
                {"background": body, "layers": [layer]}, Image.new("RGB", (32, 32)),
                APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 2, False, "", True, 0)
        self.assertLess(pipe.request["image"][0].width, body.width)
        self.assertTrue(pipe.lora_enabled)
        self.assertEqual(result.size, body.size)
        self.assertEqual(result.getpixel((50, 50)), (255, 0, 0))
        self.assertEqual(result.getpixel((250, 50)), (0, 0, 255))
        self.assertEqual(saved[0].tobytes(), result.tobytes())
        self.assertIn("unpainted pixels unchanged", status)

    def test_landscape_size_preserves_ratio_and_rounds_to_32(self):
        with patch.dict(os.environ, {"FLOYD_HEADLINER_PROFILE": "standard"}):
            width, height = APP.calculate_working_size(Image.new("RGB", (1600, 900)), 1.0)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.08)

    def test_portrait_size_preserves_ratio_and_rounds_to_32(self):
        with patch.dict(os.environ, {"FLOYD_HEADLINER_PROFILE": "standard"}):
            width, height = APP.calculate_working_size(Image.new("RGB", (900, 1600)), 2.0)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 9 / 16, delta=0.04)

    def test_default_prompt_keeps_reference_order(self):
        self.assertIn("<image1>", APP.DEFAULT_PROMPT)
        self.assertIn("<image2>", APP.DEFAULT_PROMPT)
        self.assertLess(APP.DEFAULT_PROMPT.index("<image1>"), APP.DEFAULT_PROMPT.index("<image2>"))

    def test_model_profile_selects_matching_turbo_adapter(self):
        with patch.dict("os.environ", {"FLOYD_HEADLINER_PROFILE": "low-vram"}):
            self.assertEqual(APP.selected_turbo_lora(), APP.LOW_TURBO_LORA)
        with patch.dict("os.environ", {"FLOYD_HEADLINER_PROFILE": "standard"}):
            self.assertEqual(APP.selected_turbo_lora(), APP.TURBO_LORA)

    def test_extra_prompt_appends_without_changing_main_instruction(self):
        self.assertEqual(APP.compose_prompt(APP.DEFAULT_PROMPT, ""), APP.DEFAULT_PROMPT)
        self.assertEqual(
            APP.compose_prompt(APP.DEFAULT_PROMPT, "  Remove the hat.  "),
            APP.DEFAULT_PROMPT + "\nRemove the hat.",
        )
        self.assertEqual(APP.compose_prompt(APP.DEFAULT_PROMPT, None), APP.DEFAULT_PROMPT)

    def test_generation_passes_extra_prompt_after_main_prompt(self):
        class FakePipe:
            fast_resident = False
            prompt_cache_hit = False

            def enable_lora(self):
                pass

            def set_adapters(self, *args, **kwargs):
                pass

            def __call__(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(images=[Image.new("RGB", (32, 32))])

        pipe = FakePipe()
        with patch.object(APP, "_load_pipeline", return_value=pipe), \
                patch("torch.Generator"), patch.object(Image.Image, "save"), \
                patch("torch.cuda.empty_cache"):
            APP.run_swap(
                Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)),
                APP.DEFAULT_PROMPT, 42, False, 1, 1, 6, .5, 1,
                keep_on_gpu=False, extra_prompt="Remove the hat.",
            )
        self.assertEqual(pipe.request["prompt"], APP.DEFAULT_PROMPT + "\nRemove the hat.")
        self.assertEqual(len(pipe.request["image"]), 2)

    def test_generation_rejects_missing_body_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "body/base"):
            APP.run_swap(None, Image.new("RGB", (64, 64)), APP.DEFAULT_PROMPT, 1, False, 1, 1, 6, 1, 2)

    def test_generation_rejects_missing_head_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "head reference"):
            APP.run_swap(Image.new("RGB", (64, 64)), None, APP.DEFAULT_PROMPT, 1, False, 1, 1, 6, 1, 2)

    def test_fused_mlp_lora_is_split_into_gate_and_projection(self):
        import torch

        state = {
            "transformer.block.img_mlp.gate_up.lora_A.weight": torch.arange(6).reshape(2, 3),
            "transformer.block.img_mlp.gate_up.lora_B.weight": torch.arange(16).reshape(8, 2),
        }
        converted = APP.split_fused_mlp_lora(state)
        self.assertNotIn("transformer.block.img_mlp.gate_up.lora_A.weight", converted)
        self.assertEqual(converted["transformer.block.img_mlp.gate_layer.lora_B.weight"].shape, (4, 2))
        self.assertEqual(converted["transformer.block.img_mlp.proj.lora_B.weight"].shape, (4, 2))
        self.assertTrue(
            torch.equal(
                converted["transformer.block.img_mlp.gate_layer.lora_A.weight"],
                converted["transformer.block.img_mlp.proj.lora_A.weight"],
            )
        )

    def test_global_vram_fallback_returns_torch_value(self):
        from unittest.mock import patch

        class FakeCuda:
            @staticmethod
            def mem_get_info(index):
                self.assertEqual(index, 0)
                return 123456, 999999

        class FakeTorch:
            cuda = FakeCuda()

        with patch("subprocess.check_output", side_effect=OSError("missing")):
            self.assertEqual(APP.global_free_vram_bytes(FakeTorch), 123456)

    def test_optional_core_preserves_standalone_engine(self):
        source = APP_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("import comfy", source)
        self.assertIn("standalone (diffusers)", source)
        self.assertIn("from core_runtime import", source)

    def test_fast_core_defaults_to_app_local_files(self):
        import core_runtime

        self.assertEqual(core_runtime.CORE_ROOT, APP_PATH.parent / "vendor" / "comfy_core")
        self.assertEqual(core_runtime.MODEL_ROOT, APP_PATH.parent / "models" / "fast-core")
        self.assertTrue((core_runtime.CORE_ROOT / "nodes.py").is_file())


if __name__ == "__main__":
    unittest.main()
