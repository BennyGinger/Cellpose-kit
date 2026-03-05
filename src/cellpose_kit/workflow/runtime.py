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
        use_nuclear_channel (bool): Whether nuclear channel mode is enabled, for informational purposes.
        do_3D (bool): Whether the model is configured for 3D segmentation, for informational purposes.
        model_names (list[str] | None): List of available model names for validation and informational purposes.
        backend_name (str | None): Name of the backend being used (e.g., 'v3' or 'v4') for informational purposes.
        lock (Lock | None): Optional lock for thread-safe inference when threading is enabled.
    """
    
    model: CellposeModel | CellposeDenoiseModel
    eval_params: dict[str, Any]
    use_nuclear_channel: bool = False
    do_3D: bool = False
    model_names: list[str] | None = None
    backend_name: str | None = None
    lock: Lock | None = None
    
    def dump(self) -> dict[str, Any]:
        """
        Dump the model context information into a dictionary for logging or debugging purposes. This includes the model type, evaluation parameters, and configuration flags.
        """
        return {
            "model": self.model.__class__.__name__,
            "eval_params": self.eval_params,
            "use_nuclear_channel": self.use_nuclear_channel,
            "do_3D": self.do_3D,
            "model_names": self.model_names,
            "backend_name": self.backend_name,
            "is_lock": True if self.lock else False,
        }
    

def extract_model_name(model_context: ModelContext) -> str | None:
    """
    Extract the model name from the Cellpose model instance for informational purposes. It checks common attributes that may contain the model name, such as 'pretrained_model' or 'model_type'. If no valid model name is found, it returns None.
    """
    
    model = model_context.model
    for attr in ("pretrained_model", "model_type"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None