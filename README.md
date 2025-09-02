# Cellpose Kit

A clean, unified API wrapper for Cellpose v3 and v4 that provides version-transparent cell segmentation with consistent interfaces.

## Features

- **Version Agnostic**: Automatically detects and adapts to Cellpose v3 or v4
- **Clean API**: Simple setup-once, run-many pattern for efficient processing
- **Thread Safe**: Built-in threading support for concurrent inference
- **Flexible Input**: Handles single images, image lists, and batch processing
- **Backward Compatible**: Smooth migration path from Cellpose v3 to v4
- **Input Validation**: Automatic channel validation prevents runtime errors
- **Type Hints**: Full type annotations for better IDE support

## Installation

```bash
pip install cellpose-kit
```

**Prerequisites**: Cellpose v3.0+ must be installed
```bash
# For Cellpose v3
pip install cellpose>=3.0

# For Cellpose v4  
pip install cellpose>=4.0
```

## Quick Start

```python
import cellpose_kit
import numpy as np
from tifffile import imread

# Load your image
img = imread('your_image.tif')

# Setup Cellpose (once per session)
settings = cellpose_kit.setup_cellpose({
    'diameter': 30,  # v3: required, v4: deprecated (auto-detected)
    'flow_threshold': 0.4,
    'cellprob_threshold': 0.0
})

# Run segmentation (can be called multiple times)
masks, flows, styles = cellpose_kit.run_cellpose(img, settings)
```

## API Reference

### `setup_cellpose(cellpose_settings, threading=False, use_nuclear_channel=False, do_denoise=False)`

Configure Cellpose model and parameters for reuse.

**Parameters:**
- `cellpose_settings` (dict): Cellpose configuration parameters
- `threading` (bool): Enable thread-safe inference with locks
- `use_nuclear_channel` (bool): Configure nuclear channel usage
  - v3: Sets `channels=[1,2]` for cytoplasm + nucleus (requires >= 2 channels)
  - v4: Informational only (expects 3-channel input)
- `do_denoise` (bool): Apply denoising (v3 only, ignored in v4)

**Returns:**
- dict: Configured settings for `run_cellpose()`

### `run_cellpose(img, configured_settings)`

Execute Cellpose segmentation with pre-configured settings.

**Parameters:**
- `img`: Input image(s) - numpy array or list of arrays
- `configured_settings`: Settings from `setup_cellpose()`

**Returns:**
- tuple: `(masks, flows, styles)`

## Usage Examples

### Basic Segmentation

```python
import cellpose_kit

# Minimal setup with defaults
settings = cellpose_kit.setup_cellpose({})
masks, flows, styles = cellpose_kit.run_cellpose(image, settings)
```

### Custom Model Configuration

```python
# Using built-in models
settings = cellpose_kit.setup_cellpose({
    'model_type': 'cyto2',  # v3: cyto2, nuclei, etc.
    'pretrained_model': 'cpsam',  # v4: cpsam, etc.
    'diameter': 25,  # v3: required, v4: deprecated
    'flow_threshold': 0.5
})

# Using custom trained model
settings = cellpose_kit.setup_cellpose({
    'model_type': '/path/to/custom_model.pth',  # Works in both versions
    'diameter': None  # Auto-detect (v3 only, ignored in v4)
})
```

### Batch Processing

```python
# Process multiple images efficiently
image_list = [imread(f) for f in image_files]

settings = cellpose_kit.setup_cellpose({'diameter': 30})
masks, flows, styles = cellpose_kit.run_cellpose(image_list, settings)

# Results are lists matching input
for i, mask in enumerate(masks):
    print(f"Image {i}: found {mask.max()} cells")
```

### 3D Segmentation

```python
# 3D volume segmentation
settings = cellpose_kit.setup_cellpose({
    'do_3D': True,
    'anisotropy': 2.0,  # Z-resolution adjustment
    'diameter': 25 # v3: required, v4: deprecated
})

masks_3d, flows_3d, styles_3d = cellpose_kit.run_cellpose(volume, settings)
```

### 3D Stitching (Z-stack)

```python
# Process 2D slices then stitch in 3D
settings = cellpose_kit.setup_cellpose({
    'stitch_threshold': 0.75,
    'diameter': 30  # v3: required, v4: deprecated
})

# Input: (Z, Y, X) array
masks_stitched, flows, styles = cellpose_kit.run_cellpose(z_stack, settings)
```

### Thread-Safe Processing

```python
import concurrent.futures

# Setup with threading enabled
settings = cellpose_kit.setup_cellpose({}, threading=True)

def process_image(img):
    return cellpose_kit.run_cellpose(img, settings)

# Safe for concurrent use
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, image_list))
```

### Nuclear Channel Usage

```python
# For cytoplasm + nucleus segmentation
settings = cellpose_kit.setup_cellpose(
    {'diameter': 30},
    use_nuclear_channel=True
)

# v3: Automatically sets channels=[1,2] 
# v4: Expects 3-channel input (cyto, nucleus, background)
masks, flows, styles = cellpose_kit.run_cellpose(multi_channel_img, settings)
```

**Note**: When `use_nuclear_channel=True` in v3, your images must have at least 2 channels in the last dimension (e.g., shape `(H, W, 2)` or `(Z, H, W, 2)`). Single-channel images `(H, W)` will raise a validation error with guidance on how to fix the issue.

#### Channel Validation

Cellpose Kit automatically validates image formats to prevent runtime errors:

```python
# ❌ This will raise a clear ValidationError:
single_channel = np.random.randint(0, 255, (512, 512), dtype=np.uint8)  # Shape: (H, W)
settings = cellpose_kit.setup_cellpose({}, use_nuclear_channel=True)
# ValueError: Nuclear channel mode requires at least 2 channels...

# ✅ Correct approaches:
# Option 1: Use multi-channel image
multi_channel = np.random.randint(0, 255, (512, 512, 2), dtype=np.uint8)  # Shape: (H, W, C)
masks, flows, styles = cellpose_kit.run_cellpose(multi_channel, settings)

# Option 2: Disable nuclear channel mode for single-channel
settings_single = cellpose_kit.setup_cellpose({}, use_nuclear_channel=False)
masks, flows, styles = cellpose_kit.run_cellpose(single_channel, settings_single)
```

## Version Differences

Cellpose Kit automatically handles differences between versions:

| Feature | Cellpose v3 | Cellpose v4 |
|---------|-------------|-------------|
| **Model Parameter** | `model_type` | `pretrained_model` |
| **Available Models** | cyto, cyto2, nuclei | cpsam |
| **Nuclear Channels** | `channels=[1,2]` | 3-channel input expected |
| **Diameter** | ✅ Required parameter | ⚠️ Deprecated (auto-detected) |
| **Denoising** | ✅ Supported | ❌ Not implemented |
| **Model Files** | ✅ Custom models | ✅ Custom models |

### Migration from v3 to v4

Your existing code works automatically:

```python
# This works in both v3 and v4
settings = cellpose_kit.setup_cellpose({
    'model_type': 'cyto2',  # v3: uses cyto2, v4: falls back to cpsam with warning
    'diameter': 30  # v3: used for segmentation, v4: ignored (auto-detected)
})
```

**Note**: In v4, the `diameter` parameter is deprecated as Cellpose v4 automatically detects cell sizes. Your existing code will work but the diameter value will be ignored in v4.

## Configuration Options

### Common Parameters

```python
cellpose_settings = {
    # Model settings
    'model_type': 'cyto2',  # v3 model names
    'pretrained_model': 'cpsam',  # v4 model names or file paths
    'diameter': 30,  # Cell diameter in pixels (v3 only, deprecated in v4, None for auto)
    
    # Segmentation thresholds
    'flow_threshold': 0.4,  # Flow error threshold (0.0-1.0)
    'cellprob_threshold': 0.0,  # Cell probability threshold
    
    # 3D processing
    'do_3D': False,  # Enable 3D segmentation
    'anisotropy': 2.0,  # Z-resolution scaling factor
    'stitch_threshold': 0.0,  # 3D stitching threshold
    
    # Performance
    'batch_size': 8,  # GPU batch size
    'resample': True,  # High-quality boundaries
    
    # Size filtering
    'min_size': 15,  # Remove small objects
    'max_size_fraction': 0.4,  # Remove large objects
}
```

## Error Handling

```python
import cellpose_kit

try:
    settings = cellpose_kit.setup_cellpose({'model_type': 'invalid_model'})
except Exception as e:
    print(f"Setup failed: {e}")
    # Falls back to default model with warnings

try:
    masks, flows, styles = cellpose_kit.run_cellpose(img, invalid_settings)
except KeyError as e:
    print(f"Invalid settings: {e}")
    # Use setup_cellpose() to create valid settings

# Nuclear channel validation
try:
    single_channel_img = np.array(...)  # Shape: (512, 512)
    settings = cellpose_kit.setup_cellpose({}, use_nuclear_channel=True)
    masks, flows, styles = cellpose_kit.run_cellpose(single_channel_img, settings)
except ValueError as e:
    print(f"Channel validation error: {e}")
    # Either provide multi-channel image or set use_nuclear_channel=False
```

## Requirements

- Python ≥ 3.8
- Cellpose ≥ 3.0
- NumPy
- Packaging

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Citation

If you use Cellpose Kit in your research, please cite the original Cellpose papers:

```bibtex
@article{stringer2021cellpose,
  title={Cellpose: a generalist algorithm for cellular segmentation},
  author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
  journal={Nature methods},
  volume={18},
  number={1},
  pages={100--106},
  year={2021},
  publisher={Nature Publishing Group}
}
```
