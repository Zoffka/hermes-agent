"""Pollinations image generation backend.

Free, no-API-key image generation via pollinations.ai.
Supports FLUX and other diffusion models. Always available.

Selection precedence:

1. ``POLLINATIONS_IMAGE_MODEL`` env var
2. ``image_gen.pollinations.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our IDs)
4. :data:`DEFAULT_MODEL` — ``flux"
"""

from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "flux": {
        "display": "FLUX",
        "speed": "~5-15s",
        "strengths": "Best quality, strong prompt adherence",
    },
    "turbo": {
        "display": "Turbo",
        "speed": "~2-5s",
        "strengths": "Fastest, good for iteration",
    },
}

DEFAULT_MODEL = "flux"

_SIZES = {
    "landscape": (1024, 576),
    "square": (1024, 1024),
    "portrait": (576, 1024),
}


def _load_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> str:
    """Decide which model to use."""
    env_override = os.environ.get("POLLINATIONS_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override

    cfg = _load_config()
    poll_cfg = cfg.get("pollinations") if isinstance(cfg.get("pollinations"), dict) else {}
    candidate = None
    if isinstance(poll_cfg, dict):
        value = poll_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    return candidate if candidate is not None else DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class PollinationsImageGenProvider(ImageGenProvider):
    """Pollinations.ai free image generation backend."""

    @property
    def name(self) -> str:
        return "pollinations"

    @property
    def display_name(self) -> str:
        return "Pollinations"

    def is_available(self) -> bool:
        # Always available — no API key needed
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "free",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Pollinations",
            "badge": "free",
            "tag": "Free image generation via pollinations.ai — no API key needed",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id = _resolve_model()

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="pollinations",
                model=model_id,
                aspect_ratio=aspect,
            )

        width, height = _SIZES.get(aspect, _SIZES["landscape"])

        # Build URL
        encoded_prompt = urllib.parse.quote(prompt)
        seed = kwargs.get("seed", -1)
        if seed == -1:
            import random
            seed = random.randint(0, 999999)

        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width={width}&height={height}&seed={seed}&nologo=true"
        )

        # Optional: add model param if not flux
        if model_id != "flux":
            url += f"&model={model_id}"

        logger.info(
            "Pollinations generate: model=%s aspect=%s size=%dx%d seed=%s",
            model_id, aspect, width, height, seed,
        )

        try:
            # Download image
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Hermes-Agent/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                image_data = response.read()

            # Save to cache
            from agent.image_gen_provider import _images_cache_dir
            import datetime
            import uuid

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            short = uuid.uuid4().hex[:8]
            cache_dir = _images_cache_dir()
            path = cache_dir / f"pollinations_{ts}_{short}.png"
            path.write_bytes(image_data)

            logger.info("Pollinations image saved: %s (%d bytes)", path, len(image_data))

            return success_response(
                image=str(path),
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
                provider="pollinations",
                extra={"seed": seed, "width": width, "height": height},
            )

        except urllib.error.HTTPError as exc:
            logger.error("Pollinations HTTP %s: %s", exc.code, exc.reason)
            return error_response(
                error=f"Pollinations returned HTTP {exc.code}: {exc.reason}",
                error_type="http_error",
                provider="pollinations",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.error("Pollinations error: %s", exc, exc_info=True)
            return error_response(
                error=f"Failed to generate image: {exc}",
                error_type="provider_error",
                provider="pollinations",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(PollinationsImageGenProvider())
