import importlib.metadata
import logging
from typing import Optional

from packaging import version


logger = logging.getLogger(__name__)

# Supported major versions
SUPPORTED_VERSIONS = {3, 4}
MIN_VERSION = "3.0"

def _get_cellpose_version_info() -> tuple[Optional[str], Optional[int]]:
    """
    Get the installed Cellpose version information.
    Returns:
        tuple: (version_string, major_version) or (None, None) if not installed
    """
    try:
        # Try using importlib.metadata (Python 3.8+)
        version_string = importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        try:
            import cellpose
            version_string = getattr(cellpose, '__version__', None)
        except ImportError:
            logger.warning("Cellpose not found in current environment")
            return None, None
    if version_string is None:
        return None, None
    try:
        major_version = version.parse(version_string).major
        return version_string, major_version
    except Exception as e:
        logger.warning(f"Could not parse Cellpose version '{version_string}': {e}")
        return version_string, None

def _is_cellpose_version_supported(major_version: int) -> bool:
    """
    Check if a given major version is supported by this kit.
    
    Args:
        major_version: Major version number to check
        
    Returns:
        bool: True if supported, False otherwise
    """
    return major_version in SUPPORTED_VERSIONS

def get_cellpose_version() -> str:
    """
    Get the name of the Cellpose version. If the version is not found or not supported, raises an ImportError.

    Returns:
        str: Cellpose version (e.g., "v3", "v4")

    Raises:
        ImportError: If no compatible Cellpose version is found
    """
    cellpose_version, major_version = _get_cellpose_version_info()
    
    if cellpose_version is None:
        raise ImportError(f"Cellpose not found. Please install cellpose>={MIN_VERSION}")
    
    if major_version is None:
        raise ImportError(f"Could not determine Cellpose version from '{cellpose_version}'")
    
    if not _is_cellpose_version_supported(major_version):
        if major_version < min(SUPPORTED_VERSIONS):
            raise ImportError(
                f"Cellpose v{major_version}.x is not supported. "
                f"This kit requires Cellpose from v{min(SUPPORTED_VERSIONS)}.x to v{max(SUPPORTED_VERSIONS)}.x. "
                f"Supported versions: {sorted(SUPPORTED_VERSIONS)}. "
                f"Please upgrade: pip install cellpose>={MIN_VERSION}"
            )
        else:
            raise ImportError(
                f"Cellpose v{major_version}.x detected. This kit supports versions: {sorted(SUPPORTED_VERSIONS)}. "
                f"Compatibility is not guaranteed."
            )
    
    return f"v{major_version}"

cp_version = get_cellpose_version()
