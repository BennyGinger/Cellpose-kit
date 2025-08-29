# Check for cellpose import at package import
try:
	import cellpose  # noqa: F401
except ModuleNotFoundError:
	raise ImportError(
		"Cellpose is not installed. Please install cellpose>=3.0 to use cellpose_kit."
	)

from .api import setup_cellpose, run_cellpose, DEFAULT_SETTINGS, MOD_SETS, EVAL_SETS

__all__ = ["setup_cellpose", "run_cellpose", "DEFAULT_SETTINGS", "MOD_SETS", "EVAL_SETS"]
