import os, contextlib

# Check for cellpose import at package import
with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
	try:
		import cellpose  # noqa: F401
	except ModuleNotFoundError:
		raise ImportError(
			"Cellpose is not installed. Please install cellpose>=3.0 to use cellpose_kit."
		)

from .api import setup_cellpose, run_cellpose

__all__ = ["setup_cellpose", "run_cellpose"]
