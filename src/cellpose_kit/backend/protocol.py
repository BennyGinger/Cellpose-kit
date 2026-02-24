from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from abc import abstractmethod, ABC

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from cellpose.denoise import CellposeDenoiseModel


@dataclass
class Backend(ABC):
    
    model_names: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Add the model names to the backend instance. This is necessary for validating user input when specifying pretrained models.
        """
        try:
            from cellpose.models import MODEL_NAMES
            if MODEL_NAMES is not None:
                self.model_names = MODEL_NAMES
        except (ImportError, AttributeError):
            pass
    
    @abstractmethod
    def _configure_model(self, user_settings: dict[str, Any], do_denoise: bool) -> dict[str, Any]:
        """
        Configure the model settings based on user input. If missing or invalid, revert to defaults. It uses the default values from cellpose.
        
        Args:
            user_settings (dict): Dictionary containing the settings provided by the user.
            do_denoise (bool): If True, applies denoising settings.
        """
        ...
    
    @abstractmethod
    def init_model(self, user_settings: dict[str, Any], do_denoise: bool) -> CellposeModel | CellposeDenoiseModel:
        """
        Configure and initialize the Cellpose model with the given settings.
        
        Args:
            user_settings (dict): Dictionary containing the settings provided by the user.
            do_denoise (bool): If True, applies denoising settings.
        
        Returns:
            CellposeModel | CellposeDenoiseModel: The initialized Cellpose model.
        """
        ...
    
    @abstractmethod
    def configure_eval_params(self, user_settings: dict[str, Any], use_nuclear_channel: bool, do_denoise: bool) -> dict[str, Any]:
        """
        Configure the evaluation parameters based on user input. If missing or invalid, revert to defaults.
        
        Note for nuclear channel handling:
            - v3 uses 'channels=[1,2]' parameter to specify cytoplasm and nucleus channels
            - v4 expects 3-channel input where nuclear information is pre-incorporated
        
        Args:
            user_settings (dict): Dictionary containing the settings provided by the user.
            use_nuclear_channel (bool): If True, configures for nuclear channel usage.
            do_denoise (bool): If True, applies denoising settings. Only valid in v3, will be ignored in v4.
        
        Returns:
            Updated evaluation parameters as a dictionary.     
        """
        ...