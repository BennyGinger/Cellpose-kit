from __future__ import annotations

from pathlib import Path

import pytest


def test_v3_configure_eval_params_sets_nuclear_channels(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    eval_params = backend.configure_eval_params({}, use_nuclear_channel=True, do_denoise=False)
    assert eval_params["channels"] == [1, 2]


def test_v3_configure_eval_params_3d_defaults(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    eval_params = backend.configure_eval_params({"do_3D": True}, use_nuclear_channel=False, do_denoise=False)
    assert eval_params["z_axis"] == 0
    assert eval_params["anisotropy"] == 2.0


def test_v3_stitch_threshold_disables_3d(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    eval_params = backend.configure_eval_params({"do_3D": True, "stitch_threshold": 0.5}, use_nuclear_channel=False, do_denoise=False)
    assert eval_params["do_3D"] is False


def test_v3_denoise_requires_channels_list(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    eval_params = backend.configure_eval_params({}, use_nuclear_channel=False, do_denoise=True)
    assert eval_params["channels"] == [0, 0]


def test_v3_configure_model_prefers_pretrained_file(backend_v3, tmp_path: Path) -> None:
    backend = backend_v3.BackendV3()
    model_file = tmp_path / "model.npz"
    model_file.write_text("dummy")
    mod_sets = backend.configure_model({"pretrained_model": str(model_file), "model_type": "cyto2"}, do_denoise=False)
    assert mod_sets["model_type"] is None
    assert mod_sets["pretrained_model"] == str(model_file)


def test_v3_invalid_model_type_falls_back_to_default(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    mod_sets = backend.configure_model({"model_type": "invalid_model"}, do_denoise=False)
    assert mod_sets["model_type"] == backend_v3.DEFAULT_MODEL


def test_v3_init_model_uses_default_on_invalid_model_type(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    model = backend.init_model({"model_type": "invalid_model"}, do_denoise=False)
    assert model.kwargs["model_type"] == backend_v3.DEFAULT_MODEL


def test_v3_init_model_denoise_sets_restore_type(backend_v3) -> None:
    backend = backend_v3.BackendV3()
    model = backend.init_model({}, do_denoise=True)
    assert model.kwargs["restore_type"] in {"denoise_cyto2", "denoise_cyto3"}
