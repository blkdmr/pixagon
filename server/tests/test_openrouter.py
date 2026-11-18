"""OpenRouter backend, with the network replaced by a fake."""
import base64
import io
import json
import urllib.error

import pytest
from PIL import Image

from server import openrouter
from server.openrouter import OpenRouterPipeline, ProviderError


def png_url(color=(200, 40, 40, 255), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def reply(url=None, size=(64, 64)):
    """Chat-completions-style response with an image."""
    url = url or png_url(size=size)
    return {"choices": [{"message": {
        "role": "assistant", "content": "",
        "images": [{"type": "image_url", "image_url": {"url": url}}]}}]}


def images_reply(size=(64, 64), color=(200, 40, 40, 255)):
    """Dedicated /api/v1/images-style response with base64 image data."""
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"created": 1700000000, "data": [{"b64_json": b64}]}


class FakePost:
    """Records every call; answers from a list of replies or exceptions."""

    def __init__(self, *answers, size=(64, 64)):
        self.calls = []
        self.answers = list(answers)
        self.size = size

    def __call__(self, url, body, key):
        self.calls.append({"url": url, "body": body, "key": key})
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return reply(size=self.size)


def make(post, key="sk-or-test", model="google/gemini-2.5-flash-image"):
    return OpenRouterPipeline(key, model, "https://openrouter.ai/api/v1/",
                              post=post, sleep=lambda s: None)


def http_error(code, message="nope"):
    body = io.BytesIO(json.dumps({"error": {"message": message}}).encode())
    return urllib.error.HTTPError("u", code, message, {}, body)


def sprite(size=(16, 16)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(4, 12):
        for y in range(4, 12):
            img.putpixel((x, y), (30, 120, 220, 255))
    return img


def test_load_without_a_key_explains_where_to_put_it():
    with pytest.raises(ProviderError) as err:
        make(FakePost(), key=None).load()
    assert "OPENROUTER_API_KEY" in str(err.value)
    assert "Model step" in str(err.value)


def test_txt2img_asks_for_pictures_with_the_right_shape():
    post = FakePost()
    images = make(post).txt2img("gold coin", (32, 64), variants=3,
                                seeds=[1, 2, 3])
    assert len(images) == 3 and all(i.mode == "RGBA" for i in images)
    assert len(post.calls) == 3
    call = post.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["key"] == "sk-or-test"
    body = call["body"]
    assert body["model"] == "google/gemini-2.5-flash-image"
    assert body["modalities"] == ["image", "text"]
    assert body["image_config"] == {"aspect_ratio": "9:16"}  # nearest to 1:2
    text = body["messages"][0]["content"][0]["text"]
    assert text.startswith("gold coin") and "pixel art" in text
    assert sorted(c["body"]["seed"] for c in post.calls) == [1, 2, 3]


def test_progress_counts_finished_variants():
    seen = []
    make(FakePost()).txt2img("x", (32, 32), variants=4,
                             on_progress=seen.append)
    assert sorted(seen) == [0.25, 0.5, 0.75, 1.0]


def test_a_cancel_from_progress_stops_the_job():
    class Cancelled(Exception):
        pass

    def cancel(_):
        raise Cancelled()
    with pytest.raises(Cancelled):
        make(FakePost()).txt2img("x", (32, 32), variants=2,
                                 on_progress=cancel)


def test_edit_sends_the_sprite_crisp_and_returns_its_size():
    post = FakePost(size=(512, 512))  # the model rescaled the input
    out = make(post).edit_by_instruction("make it glow", sprite(),
                                         variants=1, seeds=[7])
    parts = post.calls[0]["body"]["messages"][0]["content"]
    assert parts[1]["type"] == "image_url"
    sent = openrouter.from_url(parts[1]["image_url"]["url"])
    assert sent.size == (1024, 1024)  # 16 px * 64, whole-number upscale
    assert out[0].size == (1024, 1024)  # back on the input's grid


def test_edit_leaves_a_reply_of_another_shape_alone():
    post = FakePost(size=(300, 600))
    out = make(post).edit_by_instruction("taller", sprite(), variants=1)
    assert out[0].size == (300, 600)  # fit_into in main.py handles it


def test_inpaint_sends_a_mask_and_keeps_only_the_masked_area():
    mask = Image.new("L", (16, 16), 0)
    for x in range(0, 6):
        for y in range(0, 6):
            mask.putpixel((x, y), 255)
    post = FakePost(reply(png_url((250, 250, 0, 255), (1024, 1024))))
    out = make(post).inpaint("add a hat", sprite(), mask, variants=1)
    parts = post.calls[0]["body"]["messages"][0]["content"]
    assert len(parts) == 3  # text, sprite, mask
    assert "mask" in parts[0]["text"]
    result = out[0]
    assert result.size == (1024, 1024)
    # far outside the (grown) selection: fully transparent
    assert result.getpixel((900, 900))[3] == 0


def test_prep_mask_grows_at_sprite_scale():
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((4, 4), 255)
    big = openrouter.prep_mask(mask, (64, 64), 8)
    # INPAINT_MARGIN sprite pixels each way, scaled by 8
    margin = openrouter.INPAINT_MARGIN
    assert big.getpixel(((4 - margin) * 8, 4 * 8)) == 255
    assert big.getpixel(((4 - margin - 1) * 8, 4 * 8)) == 0


def test_a_bad_key_is_reported_without_retrying():
    post = FakePost(http_error(401, "No auth credentials found"))
    with pytest.raises(ProviderError) as err:
        make(post).txt2img("x", (32, 32), variants=1)
    assert "API key" in str(err.value)
    assert "No auth credentials found" in str(err.value)
    assert len(post.calls) == 1


def test_no_credits_says_so():
    with pytest.raises(ProviderError) as err:
        make(FakePost(http_error(402))).txt2img("x", (32, 32), variants=1)
    assert "credits" in str(err.value)


def test_rate_limits_and_blips_are_retried():
    post = FakePost(http_error(429), urllib.error.URLError("reset"), reply())
    images = make(post).txt2img("x", (32, 32), variants=1)
    assert len(images) == 1 and len(post.calls) == 3


def test_gives_up_after_the_retries():
    post = FakePost(*[http_error(503)] * 5)
    with pytest.raises(ProviderError) as err:
        make(post).txt2img("x", (32, 32), variants=1)
    assert "503" in str(err.value)
    assert len(post.calls) == len(openrouter.RETRY_WAITS) + 1


def test_a_text_only_model_gets_a_helpful_error():
    post = FakePost({"choices": [{"message": {"content": "I can't draw"}}]})
    with pytest.raises(ProviderError) as err:
        make(post, model="some/text-model").txt2img("x", (32, 32), variants=1)
    assert "some/text-model" in str(err.value)
    assert "image output" in str(err.value)


@pytest.mark.parametrize("code", [400, 404])
def test_unsupported_image_output_explains_how_to_change_the_model(code):
    message = ("No endpoints found that support the requested output "
               "modalities: image, text")
    # a text-only model fails on both chat/completions and the images endpoint
    post = FakePost(http_error(code, message),
                    http_error(404, "Model not found"))
    model = "google/gemma-4-31b-it:free"
    with pytest.raises(ProviderError) as err:
        make(post, model=model).txt2img("x", (32, 32), variants=1)
    text = str(err.value)
    assert model in text
    assert "image output" in text
    assert "Backend step" in text
    assert openrouter.DEFAULT_OPENROUTER_MODEL in text
    assert "restart" in text
    assert "Image input support alone is not enough" in text
    assert len(post.calls) == 2  # tried both endpoints


def test_an_unrelated_404_keeps_the_provider_error():
    post = FakePost(http_error(404, "Model not found"))
    with pytest.raises(ProviderError, match="OpenRouter error 404: Model not found"):
        make(post).txt2img("x", (32, 32), variants=1)
    assert len(post.calls) == 1


def test_an_error_inside_a_200_reply_is_raised():
    post = FakePost({"error": {"message": "model not found"}})
    with pytest.raises(ProviderError) as err:
        make(post).txt2img("x", (32, 32), variants=1)
    assert "model not found" in str(err.value)


def test_image_urls_also_reads_pictures_inside_content():
    url = png_url()
    response = {"choices": [{"message": {"content": [
        {"type": "text", "text": "here"},
        {"type": "image_url", "image_url": {"url": url}}]}}]}
    assert openrouter.image_urls(response) == [url]


def test_from_url_fetches_a_link_reply():
    raw = base64.b64decode(png_url().split(",", 1)[1])
    image = openrouter.from_url("https://cdn.example/x.png",
                                fetch=lambda u: raw)
    assert image.size == (64, 64)


@pytest.mark.parametrize("size,ratio", [((32, 32), "1:1"), ((64, 32), "2:1"),
                                        ((48, 64), "3:4"), ((96, 54), "16:9")])
def test_nearest_aspect(size, ratio):
    got = openrouter.nearest_aspect(size)
    if ratio == "2:1":  # not offered; the closest is wider or narrower
        assert got in ("16:9", "21:9")
    else:
        assert got == ratio


# ---- images endpoint (image-generation-only models like FLUX 3) ----

def _flux_404():
    """The 404 OpenRouter returns when a chat-only endpoint is used for an
    image-generation-only model."""
    return http_error(404,
        "black-forest-labs/flux-3-image is an image generation model "
        "and cannot be used with the chat/completions endpoint. "
        "Use the /api/v1/images endpoint instead.")


def test_image_gen_model_falls_back_to_images_endpoint():
    post = FakePost(_flux_404(), images_reply())
    images = make(post, model="black-forest-labs/flux-3-image") \
        .txt2img("gold coin", (32, 32), variants=1, seeds=[42])
    assert len(images) == 1 and images[0].mode == "RGBA"
    assert len(post.calls) == 2
    # first call went to chat/completions, second to images
    assert post.calls[0]["url"].endswith("/chat/completions")
    assert post.calls[1]["url"].endswith("/images")


def test_images_endpoint_body_shape():
    post = FakePost(_flux_404(), images_reply())
    make(post, model="bf/flux").txt2img("sword", (32, 64), variants=1,
                                         seeds=[7])
    body = post.calls[1]["body"]
    assert body["model"] == "bf/flux"
    assert body["prompt"].startswith("sword") and "pixel art" in body["prompt"]
    assert body["n"] == 1
    assert body["seed"] == 7
    assert body["aspect_ratio"] == "9:16"
    assert "messages" not in body


def test_images_endpoint_is_cached_after_fallback():
    answers = [_flux_404(), images_reply(), images_reply(), images_reply()]
    post = FakePost(*answers)
    pipe = make(post, model="bf/flux")
    pipe.txt2img("a", (32, 32), variants=1, seeds=[1])
    # second request goes straight to images, no chat attempt
    pipe.txt2img("b", (32, 32), variants=1, seeds=[2])
    assert post.calls[2]["url"].endswith("/images")
    # only the first request tried chat/completions
    chat_calls = [c for c in post.calls if "chat" in c["url"]]
    assert len(chat_calls) == 1


def test_images_endpoint_edit_sends_input_references():
    post = FakePost(_flux_404(), images_reply(size=(512, 512)))
    out = make(post, model="bf/flux").edit_by_instruction(
        "make it glow", sprite(), variants=1, seeds=[7])
    body = post.calls[1]["body"]
    refs = body.get("input_references", [])
    assert len(refs) == 1
    assert refs[0]["type"] == "image_url"
    assert refs[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert out[0].mode == "RGBA"


def test_images_endpoint_inpaint_sends_two_references():
    mask = Image.new("L", (16, 16), 0)
    for x in range(0, 6):
        for y in range(0, 6):
            mask.putpixel((x, y), 255)
    post = FakePost(_flux_404(), images_reply(size=(1024, 1024)))
    make(post, model="bf/flux").inpaint("add hat", sprite(), mask,
                                         variants=1, seeds=[1])
    body = post.calls[1]["body"]
    refs = body.get("input_references", [])
    assert len(refs) == 2  # sprite + mask


def test_unsupported_modalities_falls_back_to_images_endpoint():
    """Models like FLUX.2 Klein return a 'requested output modalities' 404
    from chat/completions but work on the images endpoint."""
    err = http_error(404, "No endpoints found that support the requested "
                          "output modalities: image, text")
    post = FakePost(err, images_reply())
    images = make(post, model="black-forest-labs/flux.2-klein-4b") \
        .txt2img("coin", (32, 32), variants=1, seeds=[1])
    assert len(images) == 1 and images[0].mode == "RGBA"
    assert post.calls[0]["url"].endswith("/chat/completions")
    assert post.calls[1]["url"].endswith("/images")


def test_non_images_404_does_not_trigger_fallback():
    post = FakePost(http_error(404, "Model not found"))
    with pytest.raises(ProviderError, match="Model not found"):
        make(post).txt2img("x", (32, 32), variants=1)
    assert len(post.calls) == 1


def test_seed_above_int32_is_folded_for_the_wire():
    # variant seeds go up to 2^32-1; Google rejects seeds past 2^31-1
    post = FakePost()
    make(post).txt2img("coin", (32, 32), variants=2,
                       seeds=[2**32 - 1, 3948139334])
    sent = sorted(c["body"]["seed"] for c in post.calls)
    assert all(0 <= s < 2**31 for s in sent)
    assert sent == sorted([(2**32 - 1) % 2**31, 3948139334 % 2**31])


def test_provider_error_shows_upstream_reason():
    # OpenRouter wraps the real cause in metadata.raw; the panel must see it
    raw = json.dumps({"error": {"code": 400, "message":
                                "Invalid value at 'generation_config.seed'"}})
    body = io.BytesIO(json.dumps({"error": {
        "message": "Provider returned error", "code": 400,
        "metadata": {"raw": raw, "provider_name": "Google"}}}).encode())
    err = urllib.error.HTTPError("u", 400, "Bad Request", {}, body)
    with pytest.raises(ProviderError) as e:
        make(FakePost(err)).txt2img("coin", (32, 32), variants=1, seeds=[1])
    assert "generation_config.seed" in str(e.value)
