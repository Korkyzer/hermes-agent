#!/usr/bin/env python3
"""
Video Generation Tools Module

Generates short MP4 videos from a text prompt or input images using a
two-tier provider stack:

  1. MuAPI  (https://api.muapi.ai/api/v1) -- primary, watermark-free
     Activated when MUAPIAPP_API_KEY is set.
     Submit -> poll /predictions/{request_id}/result every 3s, 5 min cap.

  2. Fal.ai -- fallback when MuAPI is not configured (or for Fal-only
     models like Kling Omni / Veo 3.1).
     Activated when FAL_KEY is set; reuses the same managed-gateway path
     as ``image_generation_tool.py`` so Nous Subscription users transit
     through the portal.

The model catalog mirrors the production registry in Korkyzer/hh-video-studio
(``src/lib/video-models.ts``) so a model id like ``seedance-2.0`` or
``kling-3.0`` behaves identically across both projects.

Output contract:
    JSON string with:
        {"success": bool,
         "video_path": "<local mp4 path>" | None,
         "video_url":  "<cdn url>"        | None,
         "media_tag":  "MEDIA:<path>"     | None,
         "provider":   "muapi" | "fal" | None,
         "model":      "<model id>",
         "duration_seconds": int}

The tool hides itself (``check_fn`` returns False) when neither
MUAPIAPP_API_KEY nor FAL_KEY is configured.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUAPI_BASE_URL = "https://api.muapi.ai/api/v1"
MUAPI_POLL_INTERVAL_S = 3
MUAPI_POLL_TIMEOUT_S = 5 * 60

VALID_ASPECT_RATIOS = ("landscape", "portrait", "square")
DEFAULT_ASPECT_RATIO = "landscape"

# Display alias -> provider-native aspect string.
_ASPECT_TO_RATIO = {
    "landscape": "16:9",
    "portrait":  "9:16",
    "square":    "1:1",
}

DEFAULT_DURATION = 5
DEFAULT_MODEL_ID = "seedance-2.0"
SEEDANCE_MAX_REFERENCE_IMAGES = 9


# ---------------------------------------------------------------------------
# Model catalog -- mirrors hh-video-studio video-models.ts
# ---------------------------------------------------------------------------
#
# Each entry declares routing data:
#   muapi.t2v / muapi.i2v -- MuAPI slug or None
#   fal.t2v   / fal.i2v   -- Fal endpoint or None
#   provider_priority     -- list of providers tried in order
#   features              -- audio, multi_ref, frames, negative_prompt
#   default_quality       -- Seedance 2.0 only ("basic" / "high")
#
# When ``provider_priority`` is e.g. ("muapi", "fal"), MuAPI is tried first
# and Fal is the silent fallback if MuAPI submission fails AND FAL_KEY is set.

VIDEO_MODELS: Dict[str, Dict[str, Any]] = {
    "kling-3.0": {
        "display": "Kling 3.0 Pro",
        "provider_priority": ("muapi", "fal"),
        "muapi": {
            "t2v": "kling-v3.0-pro-text-to-video",
            "i2v": "kling-v3.0-pro-image-to-video",
        },
        "fal": {
            "t2v": "fal-ai/kling-video/v3/pro/text-to-video",
            "i2v": "fal-ai/kling-video/v3/pro/image-to-video",
        },
        "allowed_durations": tuple(range(3, 16)),
        "supports_audio": True,
        "supports_negative_prompt": True,
        "supports_frames": True,
        "supports_multi_reference": False,
        "cost_per_sec": 0.336,  # audio on
    },
    "kling-2.6": {
        "display": "Kling 2.6 Pro",
        "provider_priority": ("muapi", "fal"),
        "muapi": {
            "t2v": "kling-v2.6-pro-t2v",
            "i2v": "kling-v2.6-pro-i2v",
        },
        "fal": {
            "t2v": "fal-ai/kling-video/v2.6/pro/text-to-video",
            "i2v": "fal-ai/kling-video/v2.6/pro/image-to-video",
        },
        "allowed_durations": (5, 10),
        "supports_audio": True,
        "supports_negative_prompt": True,
        "supports_frames": True,
        "supports_multi_reference": False,
        "cost_per_sec": 0.14,
    },
    "kling-omni": {
        "display": "Kling Omni",
        "provider_priority": ("fal",),  # Fal-only
        "muapi": None,
        "fal": {
            # Fal exposes Omni as an image-to-video / reference variant; we
            # use the standard image-to-video endpoint with reference images.
            "t2v": None,
            "i2v": "fal-ai/kling-video/o3/standard/image-to-video",
        },
        "allowed_durations": tuple(range(3, 16)),
        "supports_audio": True,
        "supports_negative_prompt": True,
        "supports_frames": True,
        "supports_multi_reference": True,
        "max_reference_images": 7,
        "cost_per_sec": 0.28,
    },
    "seedance": {
        "display": "Seedance v1.5 Pro",
        "provider_priority": ("muapi",),
        "muapi": {
            "t2v": "seedance-v1.5-pro-t2v",
            "i2v": "seedance-v1.5-pro-i2v",
        },
        "fal": None,
        "allowed_durations": (5, 8, 10),
        "supports_audio": True,
        "supports_frames": True,
        "supports_multi_reference": False,
        "cost_per_sec": 0.08,
    },
    "seedance-2.0": {
        "display": "Seedance 2.0",
        "provider_priority": ("muapi",),
        "muapi": {
            "t2v": "seedance-v2.0-t2v",
            "i2v": "seedance-v2.0-i2v",
        },
        "fal": None,
        "allowed_durations": (5, 10, 15),
        "supports_audio": True,
        "supports_frames": True,
        "supports_multi_reference": True,
        "max_reference_images": SEEDANCE_MAX_REFERENCE_IMAGES,
        "default_quality": "basic",       # basic = $0.12/s, high = $0.25/s
        "cost_per_sec": 0.12,
    },
    "ltx-video-2-fast": {
        "display": "LTX Video 2 Fast",
        "provider_priority": ("muapi", "fal"),
        "muapi": {
            "t2v": "ltx-video-2-fast-t2v",
            "i2v": "ltx-video-2-fast-i2v",
        },
        "fal": {
            "t2v": "fal-ai/ltx-video-v095/multiconditioning",
            "i2v": "fal-ai/ltx-video-v095/multiconditioning",
        },
        "allowed_durations": (6, 8, 10, 12, 14, 16, 18, 20),
        "supports_audio": False,
        "supports_frames": True,
        "supports_multi_reference": False,
        "cost_per_sec": 0.04,
    },
}

DEFAULT_MUAPI_QUALITY = "basic"
VALID_SEEDANCE_QUALITIES = ("basic", "high")


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

def _muapi_key() -> Optional[str]:
    key = os.getenv("MUAPIAPP_API_KEY", "").strip()
    return key or None


def _fal_key() -> Optional[str]:
    key = os.getenv("FAL_KEY", "").strip()
    return key or None


def has_muapi() -> bool:
    return _muapi_key() is not None


def has_fal() -> bool:
    return _fal_key() is not None


# ---------------------------------------------------------------------------
# MuAPI HTTP client (sync)
# ---------------------------------------------------------------------------

class MuapiError(RuntimeError):
    """Raised when a MuAPI submit / poll call fails."""


def _muapi_request(method: str, path: str, body: Optional[Dict[str, Any]] = None,
                   timeout: int = 60) -> Dict[str, Any]:
    key = _muapi_key()
    if key is None:
        raise MuapiError("MUAPIAPP_API_KEY is not set")

    url = f"{MUAPI_BASE_URL}{path}"
    data: Optional[bytes] = None
    headers = {"x-api-key": key, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", "replace")
        except Exception:
            err_body = ""
        raise MuapiError(f"MuAPI {method} {path} failed ({exc.code}): {err_body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise MuapiError(f"MuAPI {method} {path} network error: {exc.reason}") from exc

    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MuapiError(f"MuAPI returned non-JSON response: {raw[:200]!r}") from exc


def submit_muapi_task(slug: str, payload: Dict[str, Any]) -> str:
    """POST /{slug} -> request_id."""
    data = _muapi_request("POST", f"/{slug}", body=payload)
    request_id = data.get("request_id") or data.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise MuapiError(f"MuAPI submit returned no request_id: {data}")
    return request_id


def _normalize_muapi_status(data: Dict[str, Any]) -> str:
    raw = str(data.get("status") or "").lower()
    if raw == "completed":
        return "completed"
    if raw == "failed":
        return "failed"
    if raw in ("processing", "in_progress"):
        return "processing"
    return "pending"


def _extract_muapi_video_url(data: Dict[str, Any]) -> Optional[str]:
    """Pull the first http(s) video URL out of a MuAPI result envelope."""

    def _from_obj(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            url = obj.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
        return None

    def _from_array(items: Any) -> Optional[str]:
        if not isinstance(items, list):
            return None
        for item in items:
            if isinstance(item, str) and item.startswith("http"):
                return item
            url = _from_obj(item)
            if url:
                return url
        return None

    candidates: List[Optional[str]] = [
        _from_array(data.get("outputs")),
        _from_obj(data.get("video")),
        _from_obj(data.get("image")),
        _from_obj(data.get("output")),
        _from_obj(data.get("result")),
    ]

    nested = data.get("data") if isinstance(data.get("data"), dict) else None
    if nested:
        candidates.extend([
            _from_array(nested.get("outputs")),
            _from_obj(nested.get("video")),
            _from_obj(nested.get("image")),
            _from_obj(nested.get("output")),
            _from_obj(nested.get("result")),
        ])

    for url in candidates:
        if url:
            return url
    return None


def poll_muapi_until_complete(request_id: str,
                              poll_interval_s: int = MUAPI_POLL_INTERVAL_S,
                              timeout_s: int = MUAPI_POLL_TIMEOUT_S) -> Dict[str, Any]:
    """Poll /predictions/{id}/result until completed or timed out."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        data = _muapi_request("GET", f"/predictions/{request_id}/result")
        status = _normalize_muapi_status(data)
        if status == "completed":
            return data
        if status == "failed":
            err = data.get("error")
            raise MuapiError(str(err) if err else "MuAPI job reported failure")
        time.sleep(poll_interval_s)
    raise MuapiError(f"MuAPI job {request_id} timed out after {timeout_s}s")


# ---------------------------------------------------------------------------
# MuAPI payload builders
# ---------------------------------------------------------------------------

def _seedance_payload(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio_value: str,
    quality: str,
    audio: bool,
    image_url: Optional[str],
    last_image_url: Optional[str],
    reference_images: Optional[List[str]],
) -> Tuple[str, Dict[str, Any]]:
    meta = VIDEO_MODELS[model_id]
    is_v2 = model_id == "seedance-2.0"

    # Pick image-to-video when we have any image input.
    images = [u for u in (reference_images or []) if isinstance(u, str) and u.strip()]
    start_image = image_url or (images[0] if images else None)
    use_i2v = bool(start_image)

    slug_t2v = meta["muapi"]["t2v"]
    slug_i2v = meta["muapi"]["i2v"]

    if is_v2:
        images_list: List[str] = []
        if images:
            images_list = images[:SEEDANCE_MAX_REFERENCE_IMAGES]
        elif start_image:
            images_list = [start_image]

        if use_i2v:
            return slug_i2v, {
                "prompt": prompt,
                "images_list": images_list,
                "aspect_ratio": aspect_ratio_value,
                "duration": duration,
                "quality": quality,
                "remove_watermark": True,
            }
        return slug_t2v, {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio_value,
            "duration": duration,
            "quality": quality,
            "remove_watermark": True,
        }

    # Seedance v1.5
    base: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio_value,
        "resolution": "720p",
        "duration": duration,
        "generate_audio": audio,
        "camera_fixed": False,
        "remove_watermark": True,
    }
    if use_i2v:
        base["image_url"] = start_image
        if last_image_url:
            base["last_image"] = last_image_url
        return slug_i2v, base
    return slug_t2v, base


def _kling_muapi_payload(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio_value: str,
    audio: bool,
    image_url: Optional[str],
    last_image_url: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    meta = VIDEO_MODELS[model_id]
    is_kling_2_6 = model_id == "kling-2.6"
    audio_key = "sound" if is_kling_2_6 else "generate_audio"

    if image_url:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": duration,
            audio_key: audio,
        }
        # Kling 2.6 i2v doesn't accept last_image; 3.0 does.
        if last_image_url and not is_kling_2_6:
            payload["last_image"] = last_image_url
        return meta["muapi"]["i2v"], payload

    return meta["muapi"]["t2v"], {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio_value,
        "duration": duration,
        audio_key: audio,
    }


def _ltx_muapi_payload(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio_value: str,
    image_url: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    meta = VIDEO_MODELS[model_id]
    if image_url:
        return meta["muapi"]["i2v"], {
            "prompt": prompt,
            "image_url": image_url,
            "duration": duration,
            "aspect_ratio": aspect_ratio_value,
        }
    return meta["muapi"]["t2v"], {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio_value,
    }


def _build_muapi_request(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio_value: str,
    quality: str,
    audio: bool,
    negative_prompt: Optional[str],
    image_url: Optional[str],
    last_image_url: Optional[str],
    reference_images: Optional[List[str]],
) -> Tuple[str, Dict[str, Any]]:
    if model_id.startswith("seedance"):
        return _seedance_payload(
            model_id, prompt, duration, aspect_ratio_value, quality, audio,
            image_url, last_image_url, reference_images,
        )
    if model_id.startswith("kling-"):
        slug, payload = _kling_muapi_payload(
            model_id, prompt, duration, aspect_ratio_value, audio,
            image_url, last_image_url,
        )
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return slug, payload
    if model_id.startswith("ltx-"):
        return _ltx_muapi_payload(
            model_id, prompt, duration, aspect_ratio_value, image_url,
        )
    raise ValueError(f"No MuAPI payload builder for model '{model_id}'")


# ---------------------------------------------------------------------------
# Fal.ai fallback (optional)
# ---------------------------------------------------------------------------

def _build_fal_arguments(
    model_id: str,
    prompt: str,
    duration: int,
    aspect_ratio_value: str,
    audio: bool,
    negative_prompt: Optional[str],
    image_url: Optional[str],
    reference_images: Optional[List[str]],
) -> Tuple[str, Dict[str, Any]]:
    meta = VIDEO_MODELS[model_id]
    fal_cfg = meta.get("fal")
    if not fal_cfg:
        raise ValueError(f"Model '{model_id}' has no Fal endpoint")

    has_image = bool(image_url) or bool(reference_images)
    endpoint = fal_cfg.get("i2v") if has_image else fal_cfg.get("t2v")
    if not endpoint:
        # Fall back to the available endpoint shape.
        endpoint = fal_cfg.get("t2v") or fal_cfg.get("i2v")
    if not endpoint:
        raise ValueError(f"Model '{model_id}' has no usable Fal endpoint")

    args: Dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio_value,
        "duration": str(duration),
    }
    if meta.get("supports_negative_prompt") and negative_prompt:
        args["negative_prompt"] = negative_prompt
    if meta.get("supports_audio"):
        args["generate_audio"] = audio

    if has_image:
        first = image_url or (reference_images[0] if reference_images else None)
        if first:
            args["image_url"] = first
            args["start_image_url"] = first
        if meta.get("supports_multi_reference") and reference_images:
            cap = int(meta.get("max_reference_images", SEEDANCE_MAX_REFERENCE_IMAGES))
            args["reference_image_urls"] = reference_images[:cap]

    return endpoint, args


def _submit_via_fal(endpoint: str, arguments: Dict[str, Any]) -> str:
    """Submit a Fal job using the same managed/direct path as image_generation_tool.

    Returns the resulting CDN video URL.
    """
    try:
        from tools.image_generation_tool import _submit_fal_request  # type: ignore
    except Exception as exc:
        # Direct fal_client fallback when image_generation_tool isn't importable
        # in the test environment.
        try:
            import fal_client  # type: ignore
        except Exception as inner:
            raise RuntimeError(
                "Fal fallback unavailable: neither managed wrapper nor fal_client could be imported"
            ) from inner
        handler = fal_client.submit(endpoint, arguments=arguments)
        result = handler.get()
        return _extract_fal_video_url(result) or ""

    handler = _submit_fal_request(endpoint, arguments=arguments)
    result = handler.get()
    return _extract_fal_video_url(result) or ""


def _extract_fal_video_url(result: Any) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    video = result.get("video")
    if isinstance(video, dict) and isinstance(video.get("url"), str):
        return video["url"]
    videos = result.get("videos")
    if isinstance(videos, list) and videos and isinstance(videos[0], dict):
        url = videos[0].get("url")
        if isinstance(url, str):
            return url
    return None


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_video(url: str) -> str:
    """Download a remote video to a temp .mp4 path and return it.

    Mirrors the size-guard from PR #2984: if the body is < 10 KB we treat
    the response as a CDN placeholder and surface a descriptive error.
    """
    suffix = ".mp4"
    url_path = url.split("?")[0]
    tail = url_path.rsplit("/", 1)[-1]
    if "." in tail:
        ext = "." + tail.rsplit(".", 1)[-1].lower()
        if ext in (".mp4", ".mov", ".webm", ".m4v"):
            suffix = ext

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="hermes_video_")
    tmp_path = tmp.name
    tmp.close()

    logger.info("Downloading video -> %s", tmp_path)
    urllib.request.urlretrieve(url, tmp_path)
    size = os.path.getsize(tmp_path)
    logger.info("Downloaded %.1f KB", size / 1024)
    if size < 10_000:
        try:
            os.unlink(tmp_path)
        finally:
            pass
        raise ValueError(
            f"Downloaded video is suspiciously small ({size} bytes); "
            f"CDN placeholder or generation failed. URL: {url}"
        )
    return tmp_path


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _resolve_aspect(aspect_ratio: Optional[str]) -> str:
    key = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip().lower()
    if key in _ASPECT_TO_RATIO:
        return _ASPECT_TO_RATIO[key]
    if aspect_ratio in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"):
        return aspect_ratio  # accept native ratios verbatim
    return _ASPECT_TO_RATIO[DEFAULT_ASPECT_RATIO]


def _resolve_duration(model_id: str, duration: Optional[int]) -> int:
    meta = VIDEO_MODELS[model_id]
    allowed = meta["allowed_durations"]
    try:
        d = int(duration) if duration is not None else DEFAULT_DURATION
    except (TypeError, ValueError):
        d = DEFAULT_DURATION
    if d in allowed:
        return d
    # Snap to the nearest allowed duration.
    return min(allowed, key=lambda x: abs(x - d))


def _normalize_reference_images(
    reference_images: Any,
    model_id: str,
) -> Optional[List[str]]:
    if reference_images is None:
        return None
    if isinstance(reference_images, str):
        items = [reference_images]
    elif isinstance(reference_images, (list, tuple)):
        items = list(reference_images)
    else:
        return None

    cleaned: List[str] = []
    for it in items:
        if isinstance(it, str):
            v = it.strip()
            if v:
                cleaned.append(v)

    meta = VIDEO_MODELS.get(model_id, {})
    cap = int(meta.get("max_reference_images", SEEDANCE_MAX_REFERENCE_IMAGES))
    if not meta.get("supports_multi_reference"):
        cap = 1
    return cleaned[:cap] if cleaned else None


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def video_generate_tool(
    prompt: str,
    model: str = DEFAULT_MODEL_ID,
    duration: int = DEFAULT_DURATION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    quality: Optional[str] = None,
    audio: bool = True,
    negative_prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    last_image_url: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
) -> str:
    """Generate a short video. See module docstring for output contract."""
    started = datetime.datetime.now()
    model_id = (model or DEFAULT_MODEL_ID).strip().lower()

    def _fail(msg: str, provider: Optional[str] = None) -> str:
        logger.error("video_generate failed: %s", msg)
        return json.dumps({
            "success": False,
            "video_path": None,
            "video_url": None,
            "media_tag": None,
            "provider": provider,
            "model": model_id,
            "duration_seconds": int(duration) if isinstance(duration, int) else DEFAULT_DURATION,
            "error": msg,
        }, ensure_ascii=False)

    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return _fail("prompt is required and must be a non-empty string")

    if model_id not in VIDEO_MODELS:
        return _fail(
            f"Unknown model '{model}'. Valid: {sorted(VIDEO_MODELS.keys())}"
        )

    meta = VIDEO_MODELS[model_id]

    if not (has_muapi() or has_fal()):
        return _fail(
            "No video provider configured. Set MUAPIAPP_API_KEY (preferred) "
            "or FAL_KEY."
        )

    duration_resolved = _resolve_duration(model_id, duration)
    aspect_value = _resolve_aspect(aspect_ratio)

    quality_resolved = (quality or "").strip().lower()
    if quality_resolved not in VALID_SEEDANCE_QUALITIES:
        quality_resolved = meta.get("default_quality", DEFAULT_MUAPI_QUALITY)

    refs = _normalize_reference_images(reference_images, model_id)

    # Provider order: model preference, but skip providers without keys.
    provider_order: List[str] = []
    for provider in meta["provider_priority"]:
        if provider == "muapi" and has_muapi():
            provider_order.append("muapi")
        elif provider == "fal" and has_fal() and meta.get("fal"):
            provider_order.append("fal")

    if not provider_order:
        return _fail(
            f"No provider available for model '{model_id}' "
            f"(needs {', '.join(meta['provider_priority'])})"
        )

    last_error: Optional[str] = None
    for provider in provider_order:
        try:
            if provider == "muapi":
                slug, payload = _build_muapi_request(
                    model_id=model_id,
                    prompt=prompt.strip(),
                    duration=duration_resolved,
                    aspect_ratio_value=aspect_value,
                    quality=quality_resolved,
                    audio=bool(audio),
                    negative_prompt=negative_prompt,
                    image_url=image_url,
                    last_image_url=last_image_url,
                    reference_images=refs,
                )
                logger.info(
                    "MuAPI submit | model=%s slug=%s duration=%ss",
                    model_id, slug, duration_resolved,
                )
                request_id = submit_muapi_task(slug, payload)
                result = poll_muapi_until_complete(request_id)
                video_url = _extract_muapi_video_url(result)
                if not video_url:
                    raise MuapiError(
                        f"MuAPI completed but no video URL in response keys={list(result.keys())}"
                    )
            else:  # fal
                endpoint, args = _build_fal_arguments(
                    model_id=model_id,
                    prompt=prompt.strip(),
                    duration=duration_resolved,
                    aspect_ratio_value=aspect_value,
                    audio=bool(audio),
                    negative_prompt=negative_prompt,
                    image_url=image_url,
                    reference_images=refs,
                )
                logger.info(
                    "Fal submit | model=%s endpoint=%s duration=%ss",
                    model_id, endpoint, duration_resolved,
                )
                video_url = _submit_via_fal(endpoint, args)
                if not video_url:
                    raise RuntimeError("Fal returned no video URL")

            video_path = _download_video(video_url)
            elapsed = (datetime.datetime.now() - started).total_seconds()
            logger.info(
                "Video ready in %.1fs via %s | path=%s", elapsed, provider, video_path,
            )
            return json.dumps({
                "success": True,
                "video_path": video_path,
                "video_url": video_url,
                "media_tag": f"MEDIA:{video_path}",
                "provider": provider,
                "model": model_id,
                "duration_seconds": duration_resolved,
            }, ensure_ascii=False)

        except Exception as exc:  # try the next provider
            last_error = f"{provider} error: {exc}"
            logger.warning("Provider %s failed for %s: %s", provider, model_id, exc)
            continue

    return _fail(last_error or "All providers failed without an error", provider=None)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_video_generation_requirements() -> bool:
    """True when at least one provider key is configured."""
    return has_muapi() or has_fal()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": (
        "Generate a short MP4 video from a text prompt or input image(s). "
        "Provider stack: MuAPI (preferred, watermark-free) with Fal.ai as "
        "automatic fallback. When the user attached images on Discord/Slack "
        "(available as local file paths or URLs), pass them as "
        "reference_images for best identity preservation; Seedance 2.0 "
        "accepts up to 9. Returns a media_tag (MEDIA:<path>) -- include it "
        "in your reply to deliver the video as a native attachment. "
        "Generation takes 30-180 seconds; warn the user before starting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the video to generate.",
            },
            "model": {
                "type": "string",
                "enum": sorted(VIDEO_MODELS.keys()),
                "description": (
                    "Model id. Default 'seedance-2.0' (cheapest, supports "
                    "multi-image reference). Use 'kling-3.0' for premium "
                    "audio output. 'kling-omni' for Fal-only multi-reference."
                ),
                "default": DEFAULT_MODEL_ID,
            },
            "duration": {
                "type": "integer",
                "description": "Video length in seconds. Snapped to the model's allowed durations.",
                "default": DEFAULT_DURATION,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": "landscape (16:9), portrait (9:16), or square (1:1).",
                "default": DEFAULT_ASPECT_RATIO,
            },
            "quality": {
                "type": "string",
                "enum": list(VALID_SEEDANCE_QUALITIES),
                "description": (
                    "Seedance 2.0 quality tier: 'basic' ($0.12/s, default) "
                    "or 'high' ($0.25/s). Ignored for other models."
                ),
            },
            "audio": {
                "type": "boolean",
                "description": "Generate native audio (where supported). Default true.",
                "default": True,
            },
            "negative_prompt": {
                "type": "string",
                "description": "Things to avoid in the video (Kling models).",
            },
            "image_url": {
                "type": "string",
                "description": "Image URL or local path to animate (image-to-video).",
            },
            "last_image_url": {
                "type": "string",
                "description": "Optional end-frame image (Seedance v1.5, Kling 3.0).",
            },
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Up to 9 reference image URLs/paths. Used by Seedance 2.0 "
                    "(multi-reference) and Kling Omni. Pass Discord image "
                    "attachments here for identity-preserving video."
                ),
            },
        },
        "required": ["prompt"],
    },
}


def _handle_video_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required for video generation"})
    return video_generate_tool(
        prompt=prompt,
        model=args.get("model", DEFAULT_MODEL_ID),
        duration=int(args.get("duration", DEFAULT_DURATION) or DEFAULT_DURATION),
        aspect_ratio=args.get("aspect_ratio", DEFAULT_ASPECT_RATIO),
        quality=args.get("quality"),
        audio=bool(args.get("audio", True)),
        negative_prompt=args.get("negative_prompt"),
        image_url=args.get("image_url"),
        last_image_url=args.get("last_image_url"),
        reference_images=args.get("reference_images"),
    )


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=[],   # MUAPIAPP_API_KEY *or* FAL_KEY -- check_fn handles either
    is_async=False,
    emoji="🎬",
)
