from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel


@dataclass
class ModelContext:
    """
    Class to hold the Cellpose model and its evaluation parameters, along with an optional lock for thread safety.
    
    Attributes:
        model (CellposeModel | CellposeDenoiseModel): The initialized Cellpose model instance.
        eval_params (dict): The evaluation parameters configured for the model.
        model_names (list[str] | None): List of available model names for validation and informational purposes.
        backend_name (str | None): Name of the backend being used (e.g., 'v3' or 'v4') for informational purposes.
        lock (Lock | None): Optional lock for thread-safe inference when threading is enabled.
    """
    
    model: CellposeModel | CellposeDenoiseModel
    eval_params: dict[str, Any]
    model_names: list[str] | None = None
    backend_name: str | None = None
    lock: Lock | None = None