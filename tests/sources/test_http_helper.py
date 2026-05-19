"""Tests for the shared ``borrowed_client`` httpx-lifecycle helper."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from crossmem.sources._http import borrowed_client


def test_borrowed_client_does_not_close_injected() -> None:
    """An externally supplied client must outlive the context."""
    injected = httpx.Client()
    try:
        with borrowed_client(injected, timeout=1.0) as client:
            assert client is injected
        assert injected.is_closed is False
    finally:
        injected.close()


def test_borrowed_client_closes_owned() -> None:
    """A helper-created client must be closed on exit."""
    captured: dict[str, httpx.Client] = {}
    with borrowed_client(None, timeout=1.0) as client:
        captured["client"] = client
        assert client.is_closed is False
    assert captured["client"].is_closed is True


def test_borrowed_client_passes_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kwargs given to the helper must be forwarded to ``httpx.Client``."""
    captured: dict[str, Any] = {}
    original_init = httpx.Client.__init__

    def spy_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", spy_init)

    kwargs = {
        "timeout": 7.5,
        "follow_redirects": False,
        "headers": {"User-Agent": "crossmem-test"},
    }
    with borrowed_client(None, **kwargs) as client:
        assert isinstance(client, httpx.Client)

    for key, value in kwargs.items():
        assert captured.get(key) == value


def test_borrowed_client_closes_owned_on_exception() -> None:
    """If the body raises, the helper-created client must still be closed."""
    captured: dict[str, httpx.Client] = {}
    with (
        pytest.raises(RuntimeError, match="boom"),
        borrowed_client(None, timeout=1.0) as client,
    ):
        captured["client"] = client
        raise RuntimeError("boom")
    assert captured["client"].is_closed is True


def test_borrowed_client_leaves_injected_open_on_exception() -> None:
    """If the body raises, an injected client must NOT be closed by the helper."""
    injected = httpx.Client()
    try:
        with (
            pytest.raises(RuntimeError, match="boom"),
            borrowed_client(injected, timeout=1.0),
        ):
            raise RuntimeError("boom")
        assert injected.is_closed is False
    finally:
        injected.close()
