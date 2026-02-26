from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Self, TypeVar

from numpy.typing import NDArray
import numpy as np

from cellpose_kit.api import run_cellpose, setup_cellpose
from cellpose_kit.backend.runtime import ModelContext

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel


T = TypeVar('T', bound=np.generic)

@dataclass
class CellposeWrapper:
    """
    Wrapper class to hold cellpose model and evaluation parameters and execute inference with proper validation and optional threading support, regardless of the underlying Cellpose version (v3 or v4).
    
    Policy:
            - For nuclear channel handling:
                - v3: sets channels=[1,2] with 2 being the nuclear channel.
                - v4: Informational only (expects 3-channel input)
            - For denoising:
                - v3: If do_denoise=True, applies denoising settings.
                - v4: Denoising is not applicable and will be ignored if provided.
    
    Attributes:
        user_settings (dict): Dictionary containing the settings for Cellpose given by the user.
        threading (bool): If True, adds a lock for thread-safe inference.
        use_nuclear_channel (bool): If True, configures for nuclear channel usage.
        do_denoise (bool): If True, applies denoising to the input images.
        model (CellposeModel | CellposeDenoiseModel | None): Optional pre-initialized Cellpose model instance to use instead of creating a new one. Default is None.
    """
    user_settings: dict[str, Any]
    threading: bool = False
    use_nuclear_channel: bool = False
    do_denoise: bool = True
    model: CellposeModel | CellposeDenoiseModel | None = None
    
    _mod_ctx: ModelContext | None = None
    
    def setup(self) -> Self:
        """
        Setup Cellpose model and evaluation parameters once for reuse.
        """
        self._mod_ctx = setup_cellpose(self.user_settings, self.threading, self.use_nuclear_channel, self.do_denoise, self.model)
        
        return self
    
    def run(self, img: NDArray[T] | list[NDArray[T]], axis_order: str) -> tuple[NDArray[T] | list[NDArray[T]], list[NDArray[T] | list[NDArray[T]]], NDArray[T] | list[NDArray[T]]]:
        """
        Run Cellpose segmentation using pre-configured settings.
        
        Policy:
            - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
            - v4: Must have 3 channels

        Parameters:
            img: Input image(s) - NDArray or list of NDArrays

        Returns:
            tuple: (masks, flows, styles)
            - masks: NDArray (for single/batch) or list[NDArray] (for list input)
            - flows: list[NDArray] (for single/batch) or list[list[NDArray]] (for list input)
            - styles: NDArray (for single/batch) or list[NDArray] (for list input)
            
        Raises:
            RuntimeError: If the model context is not set up.
        """
        mod_ctx = self._mod_ctx
        if mod_ctx is None:
            raise RuntimeError("Model context is not set up. Please call setup() before running inference.")
        return run_cellpose(img, axis_order, mod_ctx)
    
    @property
    def version(self) -> str | None:
        """
        Get the Cellpose version being used based on the backend name in the model context.
        
        Returns:
            str | None: The Cellpose version (e.g., 'v3', 'v4') or None if not identifiable.
        """
        if self._mod_ctx and self._mod_ctx.backend_name:
            return self._mod_ctx.backend_name.lower()
        return None
    
    @property
    def model_names(self) -> list[str]:
        """
        Get the list of available model names from the model context.
        
        Returns:
            list[str]: List of available model names, if not set then returns an empty list.
        """
        if self._mod_ctx:
            names = self._mod_ctx.model_names
            return names if names is not None else []
        return []