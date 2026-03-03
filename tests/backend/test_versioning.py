from __future__ import annotations

import importlib.metadata

import pytest

from cellpose_kit.backend import versioning


def test_get_cellpose_version_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versioning.importlib.metadata, "version", lambda _: "3.1.0")
    assert versioning.get_cellpose_version() == "v3"


def test_get_cellpose_version_v4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versioning.importlib.metadata, "version", lambda _: "4.0.0")
    assert versioning.get_cellpose_version() == "v4"


def test_get_cellpose_version_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(versioning.importlib.metadata, "version", _raise)
    with pytest.raises(ImportError, match="Cellpose not found"):
        versioning.get_cellpose_version()


def test_get_cellpose_version_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versioning.importlib.metadata, "version", lambda _: "2.7.0")
    with pytest.raises(ImportError, match="not supported"):
        versioning.get_cellpose_version()


def test_get_cellpose_version_too_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versioning.importlib.metadata, "version", lambda _: "5.0.0")
    with pytest.raises(ImportError, match="Compatibility is not guaranteed"):
        versioning.get_cellpose_version()


def test_get_cellpose_version_unparseable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(versioning.importlib.metadata, "version", lambda _: "not_a_version")

    def _raise(_: str):
        raise ValueError("bad version")

    monkeypatch.setattr(versioning.version, "parse", _raise)
    with pytest.raises(ImportError, match="Could not determine"):
        versioning.get_cellpose_version()
