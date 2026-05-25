#!/usr/bin/env python3
"""Tests for the Pollinations image generation plugin."""

from __future__ import annotations

from unittest.mock import MagicMock
import urllib.error
import urllib.parse
from email.message import Message


class TestPollinationsImageGenProviderSurface:
    def test_name_and_display_name(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        provider = PollinationsImageGenProvider()
        assert provider.name == "pollinations"
        assert provider.display_name == "Pollinations"

    def test_is_always_available(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        assert PollinationsImageGenProvider().is_available() is True

    def test_default_model_and_models(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        provider = PollinationsImageGenProvider()
        assert provider.default_model() == "flux"
        models = provider.list_models()
        ids = {m["id"] for m in models}
        assert ids == {"flux", "turbo"}
        for entry in models:
            assert entry["price"] == "free"
            for field in ("id", "display", "speed", "strengths", "price"):
                assert field in entry

    def test_setup_schema_has_no_required_env(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        schema = PollinationsImageGenProvider().get_setup_schema()
        assert schema["name"] == "Pollinations"
        assert schema["badge"] == "free"
        assert schema["env_vars"] == []


class TestPollinationsModelResolution:
    def test_env_model_wins(self, monkeypatch):
        from plugins.image_gen.pollinations import _resolve_model

        monkeypatch.setenv("POLLINATIONS_IMAGE_MODEL", "turbo")
        monkeypatch.setattr("plugins.image_gen.pollinations._load_config", lambda: {"pollinations": {"model": "flux"}})
        assert _resolve_model() == "turbo"

    def test_nested_config_model(self, monkeypatch):
        from plugins.image_gen.pollinations import _resolve_model

        monkeypatch.delenv("POLLINATIONS_IMAGE_MODEL", raising=False)
        monkeypatch.setattr("plugins.image_gen.pollinations._load_config", lambda: {"pollinations": {"model": "turbo"}})
        assert _resolve_model() == "turbo"

    def test_top_level_config_model_when_supported(self, monkeypatch):
        from plugins.image_gen.pollinations import _resolve_model

        monkeypatch.delenv("POLLINATIONS_IMAGE_MODEL", raising=False)
        monkeypatch.setattr("plugins.image_gen.pollinations._load_config", lambda: {"model": "turbo"})
        assert _resolve_model() == "turbo"

    def test_invalid_model_falls_back_to_flux(self, monkeypatch):
        from plugins.image_gen.pollinations import _resolve_model

        monkeypatch.setenv("POLLINATIONS_IMAGE_MODEL", "not-real")
        monkeypatch.setattr("plugins.image_gen.pollinations._load_config", lambda: {"pollinations": {"model": "also-bad"}})
        assert _resolve_model() == "flux"


class _FakeURLResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._data


class TestPollinationsGenerate:
    def test_empty_prompt_returns_invalid_argument(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        result = PollinationsImageGenProvider().generate("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert result["provider"] == "pollinations"

    def test_generate_downloads_and_caches_image(self, monkeypatch, tmp_path):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        monkeypatch.setattr("plugins.image_gen.pollinations._resolve_model", lambda: "flux")
        monkeypatch.setattr("agent.image_gen_provider._images_cache_dir", lambda: tmp_path)

        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["timeout"] = timeout
            return _FakeURLResponse(b"png-bytes")

        monkeypatch.setattr("plugins.image_gen.pollinations.urllib.request.urlopen", fake_urlopen)

        result = PollinationsImageGenProvider().generate(
            "a cyber cockpit",
            aspect_ratio="portrait",
            seed=777,
        )

        assert result["success"] is True
        assert result["provider"] == "pollinations"
        assert result["model"] == "flux"
        assert result["prompt"] == "a cyber cockpit"
        assert result["aspect_ratio"] == "portrait"
        assert result["seed"] == 777
        assert result["width"] == 576
        assert result["height"] == 1024
        path = tmp_path / result["image"].split("/")[-1]
        assert path.exists()
        assert path.read_bytes() == b"png-bytes"
        parsed = urllib.parse.urlparse(captured["url"])
        query = urllib.parse.parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert parsed.netloc == "image.pollinations.ai"
        assert "/prompt/a%20cyber%20cockpit" in captured["url"]
        assert query["width"] == ["576"]
        assert query["height"] == ["1024"]
        assert query["seed"] == ["777"]
        assert query["nologo"] == ["true"]
        assert "model" not in query
        assert captured["timeout"] == 120
        assert captured["headers"]["User-agent"] == "Hermes-Agent/1.0"

    def test_turbo_model_adds_model_query_param(self, monkeypatch, tmp_path):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        monkeypatch.setattr("plugins.image_gen.pollinations._resolve_model", lambda: "turbo")
        monkeypatch.setattr("agent.image_gen_provider._images_cache_dir", lambda: tmp_path)
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            return _FakeURLResponse(b"png")

        monkeypatch.setattr("plugins.image_gen.pollinations.urllib.request.urlopen", fake_urlopen)

        result = PollinationsImageGenProvider().generate("p", seed=1)

        assert result["success"] is True
        query = urllib.parse.parse_qs(urllib.parse.urlparse(captured["url"]).query)
        assert query["model"] == ["turbo"]

    def test_http_error_response(self, monkeypatch):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", Message(), None)

        monkeypatch.setattr("plugins.image_gen.pollinations._resolve_model", lambda: "flux")
        monkeypatch.setattr("plugins.image_gen.pollinations.urllib.request.urlopen", fake_urlopen)

        result = PollinationsImageGenProvider().generate("p")

        assert result["success"] is False
        assert result["error_type"] == "http_error"
        assert "HTTP 429" in result["error"]
        assert result["provider"] == "pollinations"

    def test_provider_exception_response(self, monkeypatch):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider

        monkeypatch.setattr("plugins.image_gen.pollinations._resolve_model", lambda: "flux")
        monkeypatch.setattr(
            "plugins.image_gen.pollinations.urllib.request.urlopen",
            lambda req, timeout: (_ for _ in ()).throw(RuntimeError("network down")),
        )

        result = PollinationsImageGenProvider().generate("p")

        assert result["success"] is False
        assert result["error_type"] == "provider_error"
        assert "network down" in result["error"]


class TestPollinationsRegistration:
    def test_register_wires_provider(self):
        from plugins.image_gen.pollinations import PollinationsImageGenProvider, register

        ctx = MagicMock()
        register(ctx)

        ctx.register_image_gen_provider.assert_called_once()
        (registered,), _ = ctx.register_image_gen_provider.call_args
        assert isinstance(registered, PollinationsImageGenProvider)
