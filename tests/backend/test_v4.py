from __future__ import annotations

from pathlib import Path


def test_v4_default_model_when_none_provided(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    mod_sets = backend.configure_model({}, do_denoise=False)
    assert mod_sets["pretrained_model"] == backend_v4.DEFAULT_MODEL


def test_v4_invalid_pretrained_model_uses_default(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    mod_sets = backend.configure_model({"pretrained_model": "missing_model"}, do_denoise=False)
    assert mod_sets["pretrained_model"] == backend_v4.DEFAULT_MODEL


def test_v4_accepts_pretrained_model_path(backend_v4, tmp_path: Path) -> None:
    backend = backend_v4.BackendV4()
    model_file = tmp_path / "model.pt"
    model_file.write_text("dummy")
    mod_sets = backend.configure_model({"pretrained_model": str(model_file)}, do_denoise=False)
    assert mod_sets["pretrained_model"] == str(model_file)


def test_v4_deprecated_model_type_converted(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    mod_sets = backend.configure_model({"model_type": "cpsam"}, do_denoise=False)
    assert mod_sets["pretrained_model"] == "cpsam"


def test_v4_configure_eval_params_3d_defaults(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    eval_params = backend.configure_eval_params({"do_3D": True}, use_nuclear_channel=False, do_denoise=False)
    assert eval_params["z_axis"] == 0
    assert eval_params["anisotropy"] == 2.0


def test_v4_stitch_threshold_disables_3d(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    eval_params = backend.configure_eval_params({"do_3D": True, "stitch_threshold": 0.25}, use_nuclear_channel=False, do_denoise=False)
    assert eval_params["do_3D"] is False


def test_v4_init_model_uses_default_on_invalid_pretrained_model(backend_v4) -> None:
    backend = backend_v4.BackendV4()
    model = backend.init_model({"pretrained_model": "missing_model"}, do_denoise=False)
    assert model.kwargs["pretrained_model"] == backend_v4.DEFAULT_MODEL
