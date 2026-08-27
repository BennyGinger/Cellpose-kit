from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, TYPE_CHECKING, TypeVar
import logging

from numpy.typing import NDArray
import numpy as np

from cellpose_kit.workflow.api import run_cellpose
from cellpose_kit.workflow.configuration import configure_inference, initialize_model, model_key
from cellpose_kit.workflow.runtime import ModelContext
from cellpose_kit.workflow.models import SegmentationResult

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel

T = TypeVar('T', bound=np.generic)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ModelEntry:
    model: Any
    lock: Lock | None
    model_names: list[str] | None
    backend_name: str | None


_models: dict[str, _ModelEntry] = {}
_models_lock = Lock()


def _get_model(user_settings: dict[str, Any], do_denoise: bool) -> _ModelEntry:
    key = model_key(user_settings, do_denoise)
    entry = _models.get(key)
    if entry is not None:
        logger.debug("Reusing initialized Cellpose model.")
        return entry

    with _models_lock:
        entry = _models.get(key)
        if entry is None:
            context = initialize_model(user_settings,
                                       threading=True,
                                       do_denoise=do_denoise,)
            entry = _ModelEntry(model=context.model,
                                lock=context.lock,
                                model_names=context.model_names,
                                backend_name=context.backend_name,)
            _models[key] = entry
    return entry

@dataclass
class CellposeWrapper:
    """
    Wrapper class to configure and run Cellpose across supported versions.
    
    Policy:
            - For model reuse:
                - Initialized models are automatically reused within the current Python process when their backend-specific model settings match. Each wrapper keeps independent evaluation settings and results, while wrappers using the same model share its inference lock. Separate processes have separate models.
            - For multiple channels:
                - in both v3 and v4, if the array provided has multiple channels, and nuclear mode is not enabled, the channels will be processed as independent inference batches. If nuclear mode is enabled, see below.
            - For nuclear channel handling:
                - v3: requires at least 2 channels, the first one being the 'cytoplasm' channel and the second one being the 'nuclear' channel. Additional channels will be ignored.
                - v4: requires exactly 3 channels, RGB-like. All 3 channels will be processed to produce only one mask output. If only 2 channels are provided with nuclear mode enabled, the input will be automatically padded to 3 channels by adding a blank channel. Any other number of channels will raise an error.
            - For array dimensionality:
                - Both v3 and v4 can handle 2D or 3D arrays. If 3D segmentation is enabled (do_3D=True or stitch_threshold >0), the input must have a Z-axis with at least 2 slices. If the dimensionality of the input array does not meet the requirements for the configuration, a ValueError will be raised with a clear message indicating the issue.
            - For denoising:
                - v3: If do_denoise=True, applies denoising settings.
                - v4: Denoising is not applicable and will be ignored if provided.
    
    Attributes:
        user_settings (dict): Dictionary containing the settings for Cellpose given by the user.
        use_nuclear_channel (bool): If True, configures for nuclear channel usage.
        do_denoise (bool): If True, applies denoising to the input images.
        model (CellposeModel | CellposeDenoiseModel | None): Optional pre-initialized Cellpose model instance to use instead of creating a new one. Default is None.
        threading (bool): If True, adds an inference lock for an explicitly supplied model.
    """
    user_settings: dict[str, Any]
    use_nuclear_channel: bool = False
    do_denoise: bool = True
    model: CellposeModel | CellposeDenoiseModel | None = None
    threading: bool = False
    
    _mod_ctx: ModelContext | None = None
    _segmentation_result: SegmentationResult | None = None
    
    
    @classmethod
    def from_dict(cls, settings_dict: dict[str, Any]) -> CellposeWrapper:
        """
        Create a CellposeWrapper instance from a dictionary of settings.
        
        Parameters:
            settings_dict (dict): A dictionary containing the settings for Cellpose. Expected keys include 'user_settings', 'threading', 'use_nuclear_channel', 'do_denoise', and 'model'.
        
        Returns:
            CellposeWrapper: An instance of CellposeWrapper initialized with the provided settings.
        """
        return cls(user_settings=settings_dict.get('user_settings', {}),
                   threading=settings_dict.get('threading', False),
                   use_nuclear_channel=settings_dict.get('use_nuclear_channel', False),
                   do_denoise=settings_dict.get('do_denoise', True),
                   model=settings_dict.get('model', None))
    
    def setup(self) -> None:
        """
        Attach a reusable model and configure this wrapper's evaluation settings.
        """
        if self.model is None:
            entry = _get_model(self.user_settings, self.do_denoise)
            model_context = ModelContext(model=entry.model,
                                         eval_params={},
                                         model_names=entry.model_names,
                                         backend_name=entry.backend_name,
                                         lock=entry.lock,)
        else:
            model_context = initialize_model(self.user_settings,
                                             self.threading,
                                             model=self.model,
                                             do_denoise=self.do_denoise,)
        self._mod_ctx = configure_inference(self.user_settings,
                                            model_context,
                                            self.use_nuclear_channel,
                                            self.do_denoise,)
        logger.debug(f"Cellpose setup completed with: {self._mod_ctx.dump()}")

    @classmethod
    def clear_models(cls) -> None:
        """
        Release references to all automatically reused models in this process.
        """
        with _models_lock:
            _models.clear()
    
    def run(self, img: NDArray[T], axis_order: str) -> NDArray[T]:
        """
        Run Cellpose segmentation using pre-configured settings.
        
        Policy:
            - v3: Flexible channel input, but must have >= 2 channels if nuclear mode enabled
            - v4: Must have 3 channels

        Parameters:
            img: Input image ndarray
            axis_order: String representing the axis order (e.g., "TCZYX", "YXC")

        Returns:
            A reconstructed mask array which should have the same shape as input array, except with the channel dimension, depending on the configuration. Multiple channels input will return multichannel mask output (same shape), execept if nuclear mode is enabled, then the output will be a single mask channel regardless of input channels. Otherwise, if mono-channel input is provided, the output will also be mono-channel, with the 'C' axis removed from the output axis order (if it was present in the input).
            
        Raises:
            RuntimeError: If the model context is not set up.
        """
        mod_ctx = self._mod_ctx
        if mod_ctx is None:
            raise RuntimeError("Model context is not set up. Please call setup() before running inference.")
        self._segmentation_result = run_cellpose(img, axis_order, mod_ctx)
        return self._segmentation_result.masks_array()
    
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
    
    @property
    def segmentation_result(self) -> SegmentationResult | None:
        """
        Return the latest raw segmentation result object, if available.
        """
        return self._segmentation_result
    
    @property
    def output_axis_order(self) -> str | None:
        """
        Get the reconstructed mask axis order.
        
        Returns:
            str | None: Output axes when segmentation has run, otherwise None.
        """
        if self._segmentation_result is None:
            return None
        return self._segmentation_result.output_axes
