"""Tests for tools.video_generation_tool.

Covers:
- Provider availability gating (MuAPI primary, Fal fallback).
- MuAPI submit + poll lifecycle (mocked HTTP).
- Seedance / Kling / LTX payload shapes (incl. remove_watermark, images_list).
- Aspect ratio + duration normalization.
- Reference image cap (max 9 for Seedance 2.0, 1 for non-multi-reference models).
- video_url extraction across MuAPI + Fal response shapes.
- Registry registration.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools import video_generation_tool as vgt  # noqa: E402


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

class TestProviderAvailability:
    def test_check_returns_false_with_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            assert vgt.check_video_generation_requirements() is False

    def test_check_true_with_muapi_only(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "x"}, clear=True):
            assert vgt.check_video_generation_requirements() is True
            assert vgt.has_muapi() is True
            assert vgt.has_fal() is False

    def test_check_true_with_fal_only(self):
        with patch.dict(os.environ, {"FAL_KEY": "y"}, clear=True):
            assert vgt.check_video_generation_requirements() is True
            assert vgt.has_muapi() is False
            assert vgt.has_fal() is True


# ---------------------------------------------------------------------------
# Aspect / duration helpers
# ---------------------------------------------------------------------------

class TestNormalizers:
    def test_aspect_landscape(self):
        assert vgt._resolve_aspect("landscape") == "16:9"

    def test_aspect_portrait(self):
        assert vgt._resolve_aspect("portrait") == "9:16"

    def test_aspect_square(self):
        assert vgt._resolve_aspect("square") == "1:1"

    def test_aspect_native_passthrough(self):
        assert vgt._resolve_aspect("4:3") == "4:3"

    def test_aspect_invalid_defaults_landscape(self):
        assert vgt._resolve_aspect("widescreen_banana") == "16:9"

    def test_duration_snaps_to_allowed(self):
        # seedance-2.0 allows 5/10/15
        assert vgt._resolve_duration("seedance-2.0", 7) == 5
        assert vgt._resolve_duration("seedance-2.0", 12) == 10
        assert vgt._resolve_duration("seedance-2.0", 100) == 15

    def test_duration_kling26(self):
        assert vgt._resolve_duration("kling-2.6", 5) == 5
        assert vgt._resolve_duration("kling-2.6", 10) == 10
        assert vgt._resolve_duration("kling-2.6", 7) in (5, 10)


class TestReferenceImageNormalize:
    def test_caps_at_9_for_seedance2(self):
        urls = [f"http://x/{i}.png" for i in range(20)]
        out = vgt._normalize_reference_images(urls, "seedance-2.0")
        assert out is not None
        assert len(out) == 9

    def test_kling_2_6_caps_at_1(self):
        urls = [f"http://x/{i}.png" for i in range(5)]
        out = vgt._normalize_reference_images(urls, "kling-2.6")
        assert out == [urls[0]]

    def test_strips_blanks(self):
        out = vgt._normalize_reference_images(["", "  ", "http://a"], "seedance-2.0")
        assert out == ["http://a"]

    def test_none_passthrough(self):
        assert vgt._normalize_reference_images(None, "seedance-2.0") is None

    def test_string_coerces_to_list(self):
        out = vgt._normalize_reference_images("http://a", "seedance-2.0")
        assert out == ["http://a"]


# ---------------------------------------------------------------------------
# MuAPI payload shape
# ---------------------------------------------------------------------------

class TestSeedancePayloads:
    def test_seedance2_t2v_includes_remove_watermark(self):
        slug, payload = vgt._build_muapi_request(
            model_id="seedance-2.0", prompt="a sunset", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt=None, image_url=None, last_image_url=None,
            reference_images=None,
        )
        assert slug == "seedance-v2.0-t2v"
        assert payload["remove_watermark"] is True
        assert payload["prompt"] == "a sunset"
        assert payload["aspect_ratio"] == "16:9"
        assert payload["duration"] == 5
        assert payload["quality"] == "basic"

    def test_seedance2_i2v_with_images_list(self):
        urls = [f"http://x/{i}.png" for i in range(3)]
        slug, payload = vgt._build_muapi_request(
            model_id="seedance-2.0", prompt="animate", duration=5,
            aspect_ratio_value="16:9", quality="high", audio=True,
            negative_prompt=None, image_url=None, last_image_url=None,
            reference_images=urls,
        )
        assert slug == "seedance-v2.0-i2v"
        assert payload["images_list"] == urls
        assert payload["quality"] == "high"
        assert payload["remove_watermark"] is True

    def test_seedance2_i2v_caps_images_list_at_9(self):
        urls = [f"http://x/{i}.png" for i in range(20)]
        # _normalize is applied via the public surface, so cap at the helper level.
        cleaned = vgt._normalize_reference_images(urls, "seedance-2.0")
        slug, payload = vgt._build_muapi_request(
            model_id="seedance-2.0", prompt="multi-ref", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt=None, image_url=None, last_image_url=None,
            reference_images=cleaned,
        )
        assert len(payload["images_list"]) == 9

    def test_seedance_v15_t2v_audio_flag(self):
        slug, payload = vgt._build_muapi_request(
            model_id="seedance", prompt="ocean", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=False,
            negative_prompt=None, image_url=None, last_image_url=None,
            reference_images=None,
        )
        assert slug == "seedance-v1.5-pro-t2v"
        assert payload["generate_audio"] is False
        assert payload["remove_watermark"] is True
        assert payload["camera_fixed"] is False


class TestKlingPayloads:
    def test_kling3_t2v_uses_generate_audio_key(self):
        slug, payload = vgt._build_muapi_request(
            model_id="kling-3.0", prompt="a wave", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt="blur", image_url=None, last_image_url=None,
            reference_images=None,
        )
        assert slug == "kling-v3.0-pro-text-to-video"
        assert payload["generate_audio"] is True
        assert payload["negative_prompt"] == "blur"

    def test_kling26_uses_sound_key(self):
        slug, payload = vgt._build_muapi_request(
            model_id="kling-2.6", prompt="a wave", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt=None, image_url=None, last_image_url=None,
            reference_images=None,
        )
        assert slug == "kling-v2.6-pro-t2v"
        assert payload["sound"] is True
        assert "generate_audio" not in payload

    def test_kling3_i2v_includes_image_url_and_last_image(self):
        slug, payload = vgt._build_muapi_request(
            model_id="kling-3.0", prompt="animate", duration=10,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt=None, image_url="http://a.png",
            last_image_url="http://b.png", reference_images=None,
        )
        assert slug == "kling-v3.0-pro-image-to-video"
        assert payload["image_url"] == "http://a.png"
        assert payload["last_image"] == "http://b.png"

    def test_kling26_i2v_drops_last_image(self):
        slug, payload = vgt._build_muapi_request(
            model_id="kling-2.6", prompt="animate", duration=5,
            aspect_ratio_value="16:9", quality="basic", audio=True,
            negative_prompt=None, image_url="http://a.png",
            last_image_url="http://b.png", reference_images=None,
        )
        assert slug == "kling-v2.6-pro-i2v"
        assert "last_image" not in payload


# ---------------------------------------------------------------------------
# MuAPI HTTP plumbing (mocked urlopen)
# ---------------------------------------------------------------------------

def _mock_urlopen_factory(responses):
    """Return a urlopen-mock that yields responses in order."""
    iter_resp = iter(responses)

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return self._body

    def _fake_urlopen(req, timeout=60):
        body = next(iter_resp)
        return _FakeResp(json.dumps(body).encode("utf-8"))

    return _fake_urlopen


class TestMuapiHttp:
    def test_submit_returns_request_id(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory([{"request_id": "abc-123"}])):
                rid = vgt.submit_muapi_task("seedance-v2.0-t2v", {"prompt": "x"})
                assert rid == "abc-123"

    def test_submit_falls_back_to_id_field(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory([{"id": "x-9"}])):
                assert vgt.submit_muapi_task("foo", {}) == "x-9"

    def test_submit_raises_when_no_request_id(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory([{}])):
                with pytest.raises(vgt.MuapiError):
                    vgt.submit_muapi_task("foo", {})

    def test_submit_no_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(vgt.MuapiError):
                vgt.submit_muapi_task("foo", {})

    def test_poll_completes_after_processing(self):
        responses = [
            {"status": "pending"},
            {"status": "processing"},
            {"status": "completed",
             "video": {"url": "https://cdn.muapi.ai/v.mp4"}},
        ]
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory(responses)):
                with patch.object(vgt.time, "sleep", lambda s: None):
                    out = vgt.poll_muapi_until_complete("rid", poll_interval_s=0)
        assert vgt._extract_muapi_video_url(out) == "https://cdn.muapi.ai/v.mp4"

    def test_poll_failed_raises(self):
        responses = [{"status": "failed", "error": "out of credits"}]
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory(responses)):
                with pytest.raises(vgt.MuapiError) as exc_info:
                    vgt.poll_muapi_until_complete("rid", poll_interval_s=0)
        assert "out of credits" in str(exc_info.value)


class TestMuapiUrlExtract:
    def test_outputs_array_of_strings(self):
        data = {"outputs": ["https://cdn.muapi.ai/v.mp4"]}
        assert vgt._extract_muapi_video_url(data) == "https://cdn.muapi.ai/v.mp4"

    def test_outputs_array_of_objects(self):
        data = {"outputs": [{"url": "https://cdn.muapi.ai/v.mp4"}]}
        assert vgt._extract_muapi_video_url(data) == "https://cdn.muapi.ai/v.mp4"

    def test_video_object(self):
        data = {"video": {"url": "https://cdn.muapi.ai/v.mp4"}}
        assert vgt._extract_muapi_video_url(data) == "https://cdn.muapi.ai/v.mp4"

    def test_nested_data_video(self):
        data = {"data": {"video": {"url": "https://cdn.muapi.ai/v.mp4"}}}
        assert vgt._extract_muapi_video_url(data) == "https://cdn.muapi.ai/v.mp4"

    def test_no_url_returns_none(self):
        assert vgt._extract_muapi_video_url({"status": "completed"}) is None


class TestFalUrlExtract:
    def test_video_dict(self):
        assert vgt._extract_fal_video_url({"video": {"url": "https://x"}}) == "https://x"

    def test_videos_list(self):
        assert vgt._extract_fal_video_url({"videos": [{"url": "https://x"}]}) == "https://x"

    def test_unknown_returns_none(self):
        assert vgt._extract_fal_video_url({"status": "ok"}) is None


# ---------------------------------------------------------------------------
# End-to-end: video_generate_tool with mocked MuAPI + download
# ---------------------------------------------------------------------------

class TestVideoGenerateToolValidation:
    def test_empty_prompt_fails(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            result = json.loads(vgt.video_generate_tool(prompt=""))
        assert result["success"] is False

    def test_no_provider_keys_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            result = json.loads(vgt.video_generate_tool(prompt="x"))
        assert result["success"] is False
        assert "provider" in result["error"].lower()

    def test_unknown_model_fails(self):
        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k"}):
            result = json.loads(vgt.video_generate_tool(prompt="x", model="nope"))
        assert result["success"] is False
        assert "Unknown model" in result["error"]


class TestVideoGenerateToolMuapiSuccess:
    VIDEO_URL = "https://cdn.muapi.ai/generated/video.mp4"
    FAKE_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 11_000  # > 10 KB guard

    def _run(self, **kwargs):
        responses = [
            {"request_id": "rid-1"},                     # POST submit
            {"status": "completed",                        # GET poll
             "video": {"url": self.VIDEO_URL}},
        ]
        captured: dict = {}
        original_request = vgt._muapi_request

        def spy_request(method, path, body=None, timeout=60):
            captured.setdefault("calls", []).append((method, path, body))
            return original_request(method, path, body=body, timeout=timeout)

        def fake_download(url, path):
            with open(path, "wb") as fh:
                fh.write(self.FAKE_BYTES)

        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "test"}, clear=True):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory(responses)):
                with patch.object(vgt.urllib.request, "urlretrieve",
                                  side_effect=fake_download):
                    with patch.object(vgt, "_muapi_request", side_effect=spy_request):
                        with patch.object(vgt.time, "sleep", lambda s: None):
                            raw = vgt.video_generate_tool(prompt="a cat", **kwargs)
        return json.loads(raw), captured

    def test_default_model_is_seedance2(self):
        result, captured = self._run()
        assert result["success"] is True
        assert result["model"] == "seedance-2.0"
        assert result["provider"] == "muapi"
        assert result["media_tag"].startswith("MEDIA:")
        # First call hits the seedance T2V slug
        method, path, body = captured["calls"][0]
        assert method == "POST"
        assert path == "/seedance-v2.0-t2v"
        assert body["remove_watermark"] is True
        # Cleanup the temp file the download created
        if os.path.exists(result["video_path"]):
            os.unlink(result["video_path"])

    def test_passes_reference_images_to_seedance2_i2v(self):
        urls = ["http://a.png", "http://b.png", "http://c.png"]
        result, captured = self._run(reference_images=urls)
        assert result["success"] is True
        method, path, body = captured["calls"][0]
        assert path == "/seedance-v2.0-i2v"
        assert body["images_list"] == urls
        if os.path.exists(result["video_path"]):
            os.unlink(result["video_path"])

    def test_quality_high_passes_through(self):
        result, captured = self._run(quality="high")
        method, path, body = captured["calls"][0]
        assert body["quality"] == "high"
        if os.path.exists(result["video_path"]):
            os.unlink(result["video_path"])

    def test_audio_off_for_seedance2(self):
        # Seedance 2.0 doesn't expose generate_audio in the payload, but the
        # test verifies the call still succeeds when audio=False.
        result, _ = self._run(audio=False)
        assert result["success"] is True
        if os.path.exists(result["video_path"]):
            os.unlink(result["video_path"])


# ---------------------------------------------------------------------------
# Provider priority (MuAPI primary, Fal fallback)
# ---------------------------------------------------------------------------

class TestProviderPriority:
    def test_muapi_used_when_both_keys_present(self):
        responses = [
            {"request_id": "rid"},
            {"status": "completed", "video": {"url": "https://x.mp4"}},
        ]
        fake_bytes = b"\x00" * 12_000

        def fake_download(url, path):
            with open(path, "wb") as fh:
                fh.write(fake_bytes)

        with patch.dict(os.environ, {"MUAPIAPP_API_KEY": "k", "FAL_KEY": "f"}):
            with patch.object(vgt.urllib.request, "urlopen",
                              _mock_urlopen_factory(responses)):
                with patch.object(vgt.urllib.request, "urlretrieve",
                                  side_effect=fake_download):
                    with patch.object(vgt.time, "sleep", lambda s: None):
                        raw = vgt.video_generate_tool(
                            prompt="x", model="kling-3.0",
                        )
        result = json.loads(raw)
        assert result["provider"] == "muapi"
        if result["video_path"] and os.path.exists(result["video_path"]):
            os.unlink(result["video_path"])

    def test_fal_used_when_only_fal_key(self):
        # kling-3.0 lists fal as a fallback; with only FAL_KEY we skip muapi
        # and call _submit_via_fal directly.
        with patch.dict(os.environ, {"FAL_KEY": "f"}, clear=True):
            with patch.object(vgt, "_submit_via_fal",
                              return_value="https://cdn.fal.ai/x.mp4"):
                with patch.object(vgt, "_download_video",
                                  return_value="/tmp/fake.mp4"):
                    raw = vgt.video_generate_tool(prompt="x", model="kling-3.0")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["provider"] == "fal"

    def test_seedance_only_muapi_skipped_without_key(self):
        # seedance-2.0 has only "muapi" in provider_priority. Without the key
        # the tool must fail with "no provider" (not silently fall through).
        with patch.dict(os.environ, {"FAL_KEY": "f"}, clear=True):
            raw = vgt.video_generate_tool(prompt="x", model="seedance-2.0")
        result = json.loads(raw)
        assert result["success"] is False
        assert "provider" in result["error"].lower()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_video_generate_registered(self):
        from tools.registry import registry
        import tools.video_generation_tool  # noqa: F401
        entry = registry.get_entry("video_generate")
        assert entry is not None
        assert entry.toolset == "video_gen"

    def test_default_model_in_schema_enum(self):
        assert vgt.DEFAULT_MODEL_ID in vgt.VIDEO_GENERATE_SCHEMA["parameters"]["properties"]["model"]["enum"]

    def test_check_fn_returns_false_with_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            assert vgt.check_video_generation_requirements() is False
