from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any, TYPE_CHECKING

from cellpose_kit.backend.factory import load_backend
from cellpose_kit.workflow.runtime import ModelContext

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel


logger = logging.getLogger(__name__)

_warned_unknown_settings: set[tuple[str, tuple[str, ...]]] = set()
_warning_lock = Lock()


def _warn_unknown_settings(backend: Any,
                           backend_name: str,
                           user_settings: dict[str, Any],
                           do_denoise: bool,
                           ) -> None:
    supported_settings = getattr(backend, "supported_settings", None)
    if supported_settings is None:
        return

    unknown = tuple(sorted(set(user_settings) - supported_settings(do_denoise)))
    warning_key = (backend_name, unknown)
    if not unknown:
        return
    with _warning_lock:
        if warning_key in _warned_unknown_settings:
            return
        _warned_unknown_settings.add(warning_key)
    logger.warning("Ignoring unsupported Cellpose %s settings: %s",
                   backend_name,
                   ", ".join(unknown),)


def model_key(user_settings: dict[str, Any], do_denoise: bool) -> str:
    """
    Return the active backend's normalized model-construction settings.
    """
    backend, backend_name = load_backend()
    _warn_unknown_settings(backend, backend_name, user_settings, do_denoise)
    payload = {"backend": backend_name,
               "do_denoise": do_denoise,
               "model_settings": backend.configure_model(user_settings, do_denoise),}
    return json.dumps(payload, sort_keys=True, default=str)


def initialize_model(user_settings: dict[str, Any],
                     threading: bool = False,
                     do_denoise: bool = False,
                     model: CellposeModel | CellposeDenoiseModel | None = None,
                     ) -> ModelContext:
    """
    Initialize or attach a Cellpose model without configuring inference.
    """
    backend, backend_name = load_backend()
    if model is not None:
        model_instance = model
    else:
        model_instance = backend.init_model(user_settings, do_denoise)
    if model is not None:
        logger.info(f"Cellpose {backend_name} model reused.")
    else:
        logger.info(f"Cellpose {backend_name} model initialized.")

    model_context = ModelContext(model=model_instance,
                                 eval_params={},
                                 model_names=backend.model_names,
                                 backend_name=backend_name,)
    if threading:
        logger.info("Threading enabled: Adding lock for thread-safe model inference")
        model_context.lock = Lock()
    return model_context


def configure_inference(user_settings: dict[str, Any],
                        model_context: ModelContext,
                        use_nuclear_channel: bool = False,
                        do_denoise: bool = False,
                        ) -> ModelContext:
    """
    Configure evaluation parameters while reusing an initialized model.
    """
    backend, backend_name = load_backend()
    _warn_unknown_settings(backend, backend_name, user_settings, do_denoise)
    if backend_name != model_context.backend_name:
        raise RuntimeError(
            f"Cannot reuse a {model_context.backend_name!r} model with the {backend_name!r} backend.")

    eval_params = backend.configure_eval_params(user_settings,
                                                use_nuclear_channel,
                                                do_denoise,)
    do_3D = (bool(eval_params.get("do_3D", False))
             or float(eval_params.get("stitch_threshold", 0)) > 0)
    return ModelContext(model=model_context.model,
                        eval_params=eval_params,
                        do_3D=do_3D,
                        use_nuclear_channel=use_nuclear_channel,
                        model_names=model_context.model_names,
                        backend_name=backend_name,
                        lock=model_context.lock,)
