# Original work Copyright 2026 Valentyn Karachevtsev (SpriteLoom)
# Licensed under the Apache License, Version 2.0
# Modifications Copyright 2026 Alessio Russo (Pixagon)

"""OpenRouter as the picture maker: the same four tasks, done in the cloud.

Every task is one API call to an image-output model — the dedicated images
endpoint when the model requires it, with automatic fallback from
chat-completions.  No GPU, no model download, no torch import.  Prompts and
the sprite being edited leave this machine, which is why it's opt-in.

Implements pipeline.BasePipeline: load(), variant_seeds(), txt2img(),
edit_by_instruction(), inpaint(). Postprocessing in main.py is shared.
"""
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageChops, ImageFilter

from server.config import DEFAULT_OPENROUTER_MODEL
from server.instruct import (INPAINT_MARGIN, INPAINT_SUFFIX, STYLE_SUFFIX,
                             T2I_SUFFIX, flatten_to_key)
from server.pipeline import BasePipeline
from server.postprocess import remove_background

log = logging.getLogger("pixagon.openrouter")

# hosted models see more detail in a bigger picture; the pixel grid survives
# because the upscale is a whole-number nearest-neighbour one
MAX_SIDE = 1024
MAX_PARALLEL = 4      # variants requested at once
TIMEOUT = 180         # seconds per picture; image models can be slow
RETRY_WAITS = (2, 6)  # after a 429 / 5xx / network blip
# what Gemini's image_config takes; other models just ignore it
ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16",
                 "16:9", "21:9")
ONLY_IMAGE = " Reply with the image only."
INPAINT_MASK_NOTE = (" The second image is a mask: change only the area that"
                     " is white in it, and keep every other pixel exactly as"
                     " it is in the first image.")
APP_TITLE = "Pixagon"
APP_URL = "https://blkdmr.github.io/pixagon/"
IMAGE_MODEL_HELP = (
    "Choose a model with image output in the launcher's Backend step, e.g. "
    f"{DEFAULT_OPENROUTER_MODEL} (paid), then restart the server. "
    "Image input support alone is not enough.")


class ProviderError(RuntimeError):
    """A failure the user can act on, worded for the panel."""
    user_message = True  # main.py shows it as is, without the class name

    def __init__(self, msg, *, try_images=False):
        super().__init__(msg)
        self.try_images = try_images  # hint: retry on the images endpoint


def nearest_aspect(size) -> str:
    w, h = size
    ratio = w / h

    def distance(name):
        a, b = (int(x) for x in name.split(":"))
        return abs(a / b - ratio)
    return min(ASPECT_RATIOS, key=distance)


def wire_seed(seed) -> int:
    """Fold a 0..2^32-1 variant seed into 0..2^31-1. Google's API types
    seed as signed int32 and rejects anything larger with a 400."""
    return int(seed) % 2**31


def to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def from_url(url: str, fetch=None) -> Image.Image:
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
    else:  # some providers hand back a link instead of inline data
        raw = (fetch or _fetch)(url)
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGBA")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read()


def image_urls(response: dict) -> list[str]:
    """Picture URLs in a chat-completions reply, wherever the model put them."""
    if isinstance(response.get("error"), dict):
        raise ProviderError(response["error"].get("message") or "error")
    urls = []
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        parts = list(message.get("images") or [])
        content = message.get("content")
        if isinstance(content, list):
            parts += [p for p in content if isinstance(p, dict)
                      and p.get("type") == "image_url"]
        for part in parts:
            url = (part.get("image_url") or {}).get("url")
            if url:
                urls.append(url)
    return urls


def post_json(url: str, body: dict, api_key: str,
              timeout: float = TIMEOUT) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": APP_URL, "X-Title": APP_TITLE})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_message(err: urllib.error.HTTPError) -> str:
    """The error text from an OpenRouter reply. "Provider returned error" is
    only a wrapper; the upstream reason sits in metadata.raw, so add it."""
    try:
        data = json.loads(err.read().decode("utf-8", "replace"))
        error = data.get("error") or {}
        message = error.get("message") or str(err)
        raw = (error.get("metadata") or {}).get("raw")
        if raw:
            message += f" ({_raw_reason(raw)})"
        return message
    except (ValueError, AttributeError, OSError):
        return str(err)


def _raw_reason(raw) -> str:
    """The upstream provider's own message, pulled out of its raw JSON
    when it parses, else the raw text trimmed to a readable length."""
    try:
        inner = json.loads(raw) if isinstance(raw, str) else raw
        return str((inner.get("error") or {}).get("message") or raw)[:300]
    except (ValueError, AttributeError):
        return str(raw)[:300]


class OpenRouterPipeline(BasePipeline):
    def __init__(self, api_key: str | None, model: str, base_url: str,
                 post=None, sleep=time.sleep):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post = post or post_json
        self._sleep = sleep
        self._ready = False
        self._endpoint = None  # None = auto, "images" or "chat"

    # ---- the KleinPipeline interface

    def load(self):
        if not self.api_key:
            raise ProviderError(
                "no OpenRouter API key. Paste one in the launcher's Model "
                "step, or set OPENROUTER_API_KEY, then restart")
        if not self.model:
            raise ProviderError("no OpenRouter model chosen in the launcher's Backend step")
        self._ready = True
        log.info("using OpenRouter, model %s", self.model)

    def txt2img(self, prompt, target_size, variants=4, on_progress=None,
                seeds=None):
        self.load()
        extra = {"image_config": {"aspect_ratio": nearest_aspect(target_size)}}
        return self._many(prompt + T2I_SUFFIX + ONLY_IMAGE, [], variants,
                          seeds, on_progress, extra)

    def edit_by_instruction(self, instruction, image, variants=4,
                            on_progress=None, seeds=None):
        self.load()
        big, _ = prep_input(image)
        out = self._many(instruction + STYLE_SUFFIX + ONLY_IMAGE, [big],
                         variants, seeds, on_progress)
        return [match_size(img, big.size) for img in out]

    def animate_sequential(self, pipe_self, req, on_progress=None):
        """Sequential animation via cloud edits — same loop as local."""
        from server.animate import AnimationGenerator
        gen = AnimationGenerator()
        return gen.generate_sequential(self, req, on_progress)

    def inpaint(self, prompt, image, mask, variants=4, on_progress=None,
                seeds=None):
        """The model redraws the masked area; as with the local model, only
        that area is kept, everything else goes fully transparent."""
        self.load()
        big, k = prep_input(image)
        big_mask = prep_mask(mask, big.size, k)
        text = (prompt + STYLE_SUFFIX + INPAINT_SUFFIX + INPAINT_MASK_NOTE
                + ONLY_IMAGE)
        raw = self._many(text, [big, big_mask.convert("RGB")], variants,
                         seeds, on_progress)
        blank = Image.new("RGBA", big.size, (0, 0, 0, 0))
        out = []
        for img in raw:
            full = img.resize(big.size, Image.LANCZOS) \
                if img.size != big.size else img
            cleaned = remove_background(full, tolerance=16)
            keep = ImageChops.multiply(big_mask, cleaned.getchannel("A"))
            comp = blank.copy()
            comp.paste(cleaned, (0, 0), keep)
            out.append(comp)
        return out

    # ---- requests

    def _many(self, text, images, variants, seeds, on_progress, extra=None):
        seeds = seeds or self.variant_seeds(None, variants)
        results = [None] * variants
        done = 0
        pool = ThreadPoolExecutor(max_workers=max(1, min(variants,
                                                         MAX_PARALLEL)))
        try:
            futures = {pool.submit(self._one, text, images, seeds[i], extra): i
                       for i in range(variants)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if on_progress:
                    on_progress(done / variants)  # raises if the user cancels
        finally:
            # a cancel or failure must not wait for the remaining calls
            pool.shutdown(wait=False, cancel_futures=True)
        return results

    def _one(self, text, images, seed, extra=None) -> Image.Image:
        if self._endpoint == "images":
            return self._one_images(text, images, seed, extra)
        chat_err = None
        try:
            img = self._one_chat(text, images, seed, extra)
            self._endpoint = "chat"
            return img
        except ProviderError as e:
            if self._endpoint == "chat" or \
                    not self._needs_images_endpoint(e):
                raise
            log.info("%s needs the images endpoint, switching", self.model)
            chat_err = e
        try:
            img = self._one_images(text, images, seed, extra)
            self._endpoint = "images"
            return img
        except ProviderError:
            raise chat_err from None

    def _one_chat(self, text, images, seed, extra=None) -> Image.Image:
        """Request via chat/completions with image output."""
        content = [{"type": "text", "text": text}]
        content += [{"type": "image_url", "image_url": {"url": to_data_url(i)}}
                    for i in images]
        body = {"model": self.model,
                "messages": [{"role": "user", "content": content}],
                "modalities": ["image", "text"],
                "seed": wire_seed(seed)}
        body.update(extra or {})
        response = self._request(body)
        urls = image_urls(response)
        if not urls:
            raise ProviderError(
                f"{self.model} answered without a picture. {IMAGE_MODEL_HELP}")
        return from_url(urls[0])

    def _one_images(self, text, images, seed, extra=None) -> Image.Image:
        """Request via the dedicated /api/v1/images endpoint."""
        body = {"model": self.model, "prompt": text, "n": 1,
                "seed": wire_seed(seed)}
        if images:
            body["input_references"] = [
                {"type": "image_url", "image_url": {"url": to_data_url(i)}}
                for i in images]
        if extra:
            ar = (extra.get("image_config") or {}).get("aspect_ratio")
            if ar:
                body["aspect_ratio"] = ar
        response = self._request(body, "images")
        if isinstance(response.get("error"), dict):
            raise ProviderError(response["error"].get("message") or "error")
        data = response.get("data") or []
        if not data or not data[0].get("b64_json"):
            raise ProviderError(
                f"{self.model} answered without a picture. {IMAGE_MODEL_HELP}")
        raw = base64.b64decode(data[0]["b64_json"])
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGBA")

    @staticmethod
    def _needs_images_endpoint(error: ProviderError) -> bool:
        """True when the error says chat/completions can't produce images
        for this model — worth retrying on the dedicated images endpoint."""
        if getattr(error, "try_images", False):
            return True
        msg = str(error).lower()
        return ("image generation model" in msg
                or "/api/v1/images" in msg
                or "images endpoint" in msg)

    def _request(self, body: dict, path: str = "chat/completions") -> dict:
        url = self.base_url + "/" + path
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                return self._post(url, body, self.api_key)
            except urllib.error.HTTPError as e:
                message = _http_message(e)
                if e.code in (401, 403):
                    raise ProviderError(
                        f"OpenRouter refused the API key ({message})") from e
                if e.code == 402:
                    raise ProviderError(
                        f"OpenRouter: out of credits ({message})") from e
                if (e.code in (400, 404) and
                        "requested output modalities" in message.lower() and
                        "image" in message.lower()):
                    raise ProviderError(
                        f"{self.model} has no available OpenRouter endpoint "
                        f"for the requested image output. {IMAGE_MODEL_HELP}",
                        try_images=True) from e
                retry = e.code == 429 or e.code >= 500
                if not retry or attempt == len(RETRY_WAITS):
                    raise ProviderError(f"OpenRouter error {e.code}: "
                                        f"{message}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == len(RETRY_WAITS):
                    raise ProviderError(
                        f"could not reach OpenRouter: {e}") from e
            log.info("OpenRouter busy or unreachable, retrying")
            self._sleep(RETRY_WAITS[attempt])
        raise ProviderError("OpenRouter did not answer")  # not reached


def prep_input(image, max_side: int = MAX_SIDE):
    """Whole-number nearest-neighbour upscale on a flat key background, so
    pixel art reaches the model crisp; returns (image, factor)."""
    flat, _ = flatten_to_key(image)
    w, h = flat.size
    k = max(1, max_side // max(w, h))
    return flat.resize((w * k, h * k), Image.NEAREST), k


def prep_mask(mask, size, k: int = 1):
    """The selection at the same scale, grown a little past its edge. Grown
    at sprite size, then scaled: the same result as growing the big one by
    INPAINT_MARGIN * k, at a fraction of the cost."""
    small = mask.convert("L")
    if INPAINT_MARGIN > 0:
        small = small.filter(ImageFilter.MaxFilter(INPAINT_MARGIN * 2 + 1))
    return small.resize(size, Image.NEAREST)


def match_size(image, size, tolerance: float = 0.03):
    """Back to the input's size when the model only rescaled it; a reply in
    another shape is left alone for fit_into to handle."""
    if image.size == size:
        return image
    w, h = image.size
    if abs(w / h - size[0] / size[1]) <= tolerance * size[0] / size[1]:
        return image.resize(size, Image.LANCZOS)
    return image
