"""Windows-safe completion of Gradio's deferred upload publication.

Keep Gradio's multipart limits and path validation. On Windows, uploading the
same content/name again can make os.rename fail because the cache target exists.
Upstream returns that target before a background shutil.move overwrites it.
Instead publish a separate complete copy before returning the upload response.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
import shutil
from uuid import uuid4

import anyio
from fastapi import HTTPException
from PIL import Image


MAX_UPLOAD_PIXELS = 32_000_000
MAX_UPLOAD_SIDE = 8192
IMAGE_SIZE_MESSAGE = (
    "Image is too large. Resize it to at most 32 megapixels and "
    "8,192 pixels on either side, then upload it again."
)


def validate_image_path(path):
    """Read only the image header; never allocate its full pixel buffer."""
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Image.DecompressionBombError as error:
        raise ValueError(IMAGE_SIZE_MESSAGE) from error
    if width * height > MAX_UPLOAD_PIXELS or max(width, height) > MAX_UPLOAD_SIDE:
        raise ValueError(IMAGE_SIZE_MESSAGE)


def validate_component_images(payload):
    """Check cached Gradio files before a component decodes their pixels."""
    def field(value, name):
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    files = ([field(payload, "path")] if field(payload, "path") else
             [field(payload, "background"), *(field(payload, "layers") or []),
              field(payload, "composite")])
    for item in files:
        path = item if isinstance(item, str) else field(item, "path") if item is not None else None
        if path:
            validate_image_path(path)


def install_upload_fix():
    import gradio.routes as routes

    original = routes.upload_fn
    if getattr(original, "_floyd_complete_uploads", False):
        return

    @wraps(original)
    async def complete_upload(*args, **kwargs):
        # Defer only within the original helper, never past the HTTP response.
        args = list(args)
        if len(args) > 4:
            args[4] = False
            kwargs.pop("force_move", None)
        else:
            kwargs["force_move"] = False
        outputs, sources, destinations = await original(*args, **kwargs)
        pending = {str(Path(destination)): Path(source)
                   for source, destination in zip(sources, destinations, strict=True)}
        for output in outputs:
            candidate = pending.get(str(Path(output)), Path(output))
            try:
                await anyio.to_thread.run_sync(validate_image_path, candidate)
            except ValueError as error:
                raise HTTPException(status_code=413, detail=str(error)) from error
            except OSError as error:
                raise HTTPException(status_code=400, detail="The uploaded image could not be read. Upload a valid image.") from error
        replacements = {}
        for source, destination in zip(sources, destinations, strict=True):
            destination = Path(destination)
            # Never truncate a cache file that another event may be decoding.
            target = destination.with_name(
                f"{destination.stem}-upload-{uuid4().hex}{destination.suffix}"
            )
            await anyio.to_thread.run_sync(shutil.copyfile, source, str(target))
            replacements[str(destination)] = str(target)
        return [replacements.get(str(Path(path)), path) for path in outputs], [], []

    complete_upload._floyd_complete_uploads = True
    routes.upload_fn = complete_upload

