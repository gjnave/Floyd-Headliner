import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class PortraitPreviewTests(unittest.TestCase):
    def test_full_original_and_mask_are_preserved(self):
        original = Image.new('RGB', (1848, 4000), 'blue')
        original.putpixel((700, 1700), (255, 0, 0))
        editor, selected, full = app.safe_body_input(original)
        self.assertFalse(selected)
        self.assertEqual(editor['background'].size, (473, 1024))
        self.assertEqual(full.tobytes(), original.tobytes())
        layer = Image.new('RGBA', editor['background'].size)
        ImageDraw.Draw(layer).rectangle((100, 200, 199, 299), fill=(255, 10, 20, 255))
        editor['layers'] = [layer]
        editor['original'] = full
        body, mask, crop = app.prepare_selection(editor, True)
        self.assertEqual(body.tobytes(), original.tobytes())
        self.assertEqual(mask.size, original.size)
        self.assertEqual(mask.getpixel((500, 900)), 255)
        self.assertEqual(mask.getpixel((40, 40)), 0)
        self.assertTrue(crop[0] <= 500 < crop[2])
        result = app.composite_selection(body, Image.new('RGB', (100, 100), 'green'), mask, crop, 0)
        self.assertEqual(result.getpixel((40, 40)), original.getpixel((40, 40)))

    def test_orientation_normalized_before_preview_and_small_images_not_upscaled(self):
        image = Image.new('RGB', (4000, 1848))
        exif = image.getexif()
        exif[274] = 6
        editor, _, full = app.safe_body_input(image)
        self.assertEqual(full.size, (1848, 4000))
        self.assertEqual(editor['background'].size, (473, 1024))
        small, _, _ = app.safe_body_input(Image.new('RGB', (80, 100)))
        self.assertEqual(small['background'].size, (80, 100))

    def test_session_original_reaches_standard_and_turbo_callbacks(self):
        import gradio as gr
        with gr.Blocks() as demo:
            controls = app.build_likeness_ui(gr)
        original = Image.new('RGB', (1848, 4000), 'blue')
        editor, _, full = app.safe_body_input(original)
        expected = ((original, original), 'example.png', 'done', '1')
        for name in ['generate_transfer', 'generate_turbo_transfer']:
            event = next(e for e in demo.fns.values() if e.name == name)
            args = [editor, Image.new('RGB', (10, 10)), 'prompt', 42, False, 1, 1, 6,
                    1.0, 2, True, '', False, 8, 'Edit painted area', app.FAST_CORE_LABEL, full]
            with patch.object(app, 'run_swap_ui', return_value=expected) as run:
                event.fn(*args)
                self.assertIs(run.call_args.args[0]['original'], full)
                self.assertEqual(len(run.call_args.args), 16)
            self.assertIs(event.inputs[-1], controls[-1])


if __name__ == '__main__':
    unittest.main()
