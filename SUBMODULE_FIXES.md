## Overview

The project uses two Git submodules:
- `external/transfuser` - From https://github.com/autonomousvision/transfuser.git
- `external/Conv_Adapter` - From https://github.com/Hhhhhhao/Conv-Adapter.git

Several issues were encountered and fixed to make the submodules work properly with the project.

## Fixes Applied
### Deprecated np.string
**Issue**: In `external/transfuser/team_code_transfuser/data.py`, the code used `np.string_` which was removed in NumPy 2.0.

**Fix**: Replaced all instances of `np.string_` with `np.bytes_`:
```python
# Before
self.backbone = np.array(config.backbone).astype(np.string_)
self.images = np.array(self.images).astype(np.string_)
# ... and 6 other similar lines

# After
self.backbone = np.array(config.backbone).astype(np.bytes_)
self.images = np.array(self.images).astype(np.bytes_)
# ... and 6 other similar lines
```
### np.float -> float
**Issue**: used np.float
**Fix**: replace with float

### Conv-Adapter Import Fix
**Issue**: In `external/Conv_Adapter/models/backbones/efficientnet/efficientnet_blocks.py`, the import was absolute:
```python
from models.tuning_modules import PadPrompter, ConvAdapter
```

**Fix**: Changed to relative import:
**Fix**: Changed to relative import:
```python
from ...tuning_modules import PadPrompter, ConvAdapter
```

### Transfuser Import Fix
**Issue**: In `external/transfuser/team_code_transfuser/data.py`, the import was absolute:
```python
from utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform
```

**Fix**: Changed to relative import:
```python
from .utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform
```
