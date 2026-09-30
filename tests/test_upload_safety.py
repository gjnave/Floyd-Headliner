import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import gradio as gr
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upload_safety import install_upload_fix


class UploadCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_has_no_deferred_copy(self):
        import gradio.routes as routes
        original = AsyncMock(return_value=(["cache/background.png"], ["staged.png"], ["cache/background.png"]))
        with patch.object(routes, "upload_fn", original), patch("upload_safety.shutil.copyfile") as copy:
            # AsyncMock creates arbitrary attributes, so explicitly mark it unpatched.
            original._floyd_complete_uploads = False
            install_upload_fix()
            outputs, sources, destinations = await routes.upload_fn(None, "cache", 1000, None, force_move=False)
            copy.assert_called_once()
            self.assertEqual(copy.call_args.args[0], "staged.png")
            self.assertEqual(outputs, [copy.call_args.args[1]])
            self.assertNotEqual(outputs, ["cache/background.png"])
            self.assertEqual((sources, destinations), ([], []))

    async def test_copy_failure_does_not_report_success(self):
        import gradio.routes as routes
        original = AsyncMock(return_value=(["cache/background.png"], ["staged.png"], ["cache/background.png"]))
        original._floyd_complete_uploads = False
        with patch.object(routes, "upload_fn", original), \
             patch("upload_safety.shutil.copyfile", side_effect=OSError("disk full")):
            install_upload_fix()
            with self.assertRaisesRegex(OSError, "disk full"):
                await routes.upload_fn(None, "cache", 1000, None)


class UploadSafetyTests(unittest.TestCase):
    def test_repeated_concurrent_uploads_finish_before_response(self):
        import gradio.routes as routes
        install_upload_fix()
        first = routes.upload_fn
        install_upload_fix()
        self.assertIs(first, routes.upload_fn)
        with gr.Blocks() as demo:
            editor = gr.ImageEditor(type="pil")
        demo.max_file_size = None
        server = routes.App.create_app(demo)
        # Preserve test artifacts instead of deleting any files.
        cache = Path(__file__).resolve().parents[2] / "installer-cache" / ("upload-test-" + uuid4().hex)
        cache.mkdir(parents=True)
        server.uploaded_file_dir = str(cache)
        stream = io.BytesIO()
        Image.new("RGB", (256, 256), "gold").save(stream, format="PNG")
        content = stream.getvalue()

        with TestClient(server) as client:
            initial = client.post("/gradio_api/upload", files={"files": ("background.png", content, "image/png")})
            self.assertEqual(initial.status_code, 200, initial.text)
            initial_path = Path(initial.json()[0])

            def upload_and_decode(_):
                response = client.post("/gradio_api/upload", files={"files": ("background.png", content, "image/png")})
                self.assertEqual(response.status_code, 200, response.text)
                path = Path(response.json()[0])
                self.assertEqual(path.read_bytes(), content)
                payload = editor.data_model.model_validate({"background": {"path": str(path)}})
                decoded = editor.preprocess(payload)
                self.assertEqual(decoded["background"].getpixel((0, 0))[:3], (255, 215, 0))
                return path

            # Force the exact Windows destination-exists rename fallback.
            with patch("gradio.route_utils.os.rename", side_effect=FileExistsError("destination exists")):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    paths = list(pool.map(upload_and_decode, range(16)))
            self.assertEqual(len(set(paths)), 16)
            self.assertNotIn(initial_path, paths)
            self.assertEqual(initial_path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
