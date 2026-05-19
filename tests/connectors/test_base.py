"""Tests for CLIConnector ABC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crossmem.connectors import base as base_module
from crossmem.connectors.base import CLIConnector

if TYPE_CHECKING:
    from pathlib import Path


def test_backup_config_not_exported_from_base():
    """The dead ``backup_config`` shim must not live on ``connectors.base``.

    The active implementation is in ``connectors.config_io``; the copy on
    ``connectors.base`` is unused and was removed in task 22.1.
    """
    assert not hasattr(base_module, "backup_config")


def test_cannot_instantiate():
    """CLIConnector cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CLIConnector()


def test_subclass_missing_method_cannot_instantiate(tmp_path):
    """A subclass that omits any abstract method cannot be instantiated."""

    class MissingName(CLIConnector):
        def detect(self) -> bool:
            return False

        def config_path(self) -> Path:
            return tmp_path / "cfg.json"

        def register(self, server_cmd: str) -> None:
            return None

        def unregister(self) -> None:
            return None

    with pytest.raises(TypeError):
        MissingName()

    class MissingDetect(CLIConnector):
        def name(self) -> str:
            return "x"

        def config_path(self) -> Path:
            return tmp_path / "cfg.json"

        def register(self, server_cmd: str) -> None:
            return None

        def unregister(self) -> None:
            return None

    with pytest.raises(TypeError):
        MissingDetect()

    class MissingConfigPath(CLIConnector):
        def name(self) -> str:
            return "x"

        def detect(self) -> bool:
            return False

        def register(self, server_cmd: str) -> None:
            return None

        def unregister(self) -> None:
            return None

    with pytest.raises(TypeError):
        MissingConfigPath()

    class MissingRegister(CLIConnector):
        def name(self) -> str:
            return "x"

        def detect(self) -> bool:
            return False

        def config_path(self) -> Path:
            return tmp_path / "cfg.json"

        def unregister(self) -> None:
            return None

    with pytest.raises(TypeError):
        MissingRegister()

    class MissingUnregister(CLIConnector):
        def name(self) -> str:
            return "x"

        def detect(self) -> bool:
            return False

        def config_path(self) -> Path:
            return tmp_path / "cfg.json"

        def register(self, server_cmd: str) -> None:
            return None

    with pytest.raises(TypeError):
        MissingUnregister()


def test_concrete_subclass(tmp_path):
    """A subclass implementing all abstract methods can be instantiated."""
    cfg = tmp_path / "cfg.json"

    class ConcreteConnector(CLIConnector):
        def __init__(self) -> None:
            self.registered: str | None = None
            self.unregistered = False

        def name(self) -> str:
            return "concrete"

        def detect(self) -> bool:
            return cfg.parent.exists()

        def config_path(self) -> Path:
            return cfg

        def register(self, server_cmd: str) -> None:
            self.registered = server_cmd

        def unregister(self) -> None:
            self.unregistered = True

    connector = ConcreteConnector()
    assert isinstance(connector, CLIConnector)
    assert connector.name() == "concrete"
    assert connector.detect() is True
    assert connector.config_path() == cfg
    connector.register("crossmem serve")
    assert connector.registered == "crossmem serve"
    connector.unregister()
    assert connector.unregistered is True
