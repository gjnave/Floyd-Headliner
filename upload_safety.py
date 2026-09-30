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

