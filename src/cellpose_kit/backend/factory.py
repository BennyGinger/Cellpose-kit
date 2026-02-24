from cellpose_kit.backend.protocol import Backend
from cellpose_kit.versioning import get_cellpose_version


def load_backend() -> tuple[Backend, str]:
    backend_name = get_cellpose_version()
    
    if backend_name == "v3":
        from cellpose_kit.backend.v3 import BackendV3
        backend = BackendV3()
        return backend, backend_name
    if backend_name == "v4":
        from cellpose_kit.backend.v4 import BackendV4
        backend = BackendV4()
        return backend, backend_name
    
    raise RuntimeError(f"Unsupported cellpose backend version: {backend_name}, (expected v3 or v4).")
