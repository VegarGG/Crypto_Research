# Adaptive Path Configuration

This document explains the adaptive path system implemented to fix hardcoded path issues in the ArcticDB configuration.

## Problem Solved

Previously, the codebase had hardcoded paths like `/Users/zway/Desktop/BTC_Project/DB` which only worked on specific systems. This has been replaced with an adaptive configuration system.

## New Configuration

### Automatic Path Detection
The system now automatically:
- Uses the project root directory as the base
- Creates `arctic_store/` directory for ArcticDB data
- Works on Windows, macOS, and Linux
- Creates directories if they don't exist

### Environment Variable Override
You can customize the ArcticDB path by setting an environment variable:

```bash
# Windows
set CRYPTO_RESEARCH_DB_PATH=D:\MyCustom\ArcticDB\Path

# Linux/macOS  
export CRYPTO_RESEARCH_DB_PATH=/home/user/custom/arcticdb/path
```

## Usage

### Python Files
For Python scripts, simply import the configuration:

```python
from config import config, ARCTIC_URI

# Use ARCTIC_URI for ArcticDB connections
from arcticdb import Arctic
arctic = Arctic(ARCTIC_URI)
```

### Pipeline Classes
Pipeline classes now automatically use adaptive paths:

```python
# Old way - hardcoded path
pipeline = TrendIndicatorPipeline(store_path="lmdb:///Users/zway/Desktop/BTC_Project/DB")

# New way - automatic adaptive path
pipeline = TrendIndicatorPipeline()  # Uses adaptive path automatically

# Or specify custom path
pipeline = TrendIndicatorPipeline(store_path="lmdb://my/custom/path")
```

### Jupyter Notebooks
For notebooks, you can either:

1. Use the new config system:
```python
from config import config, ARCTIC_URI
ac = Arctic(ARCTIC_URI)
```

2. Or maintain backward compatibility by updating the DB_PATH variable:
```python
import os
from pathlib import Path

# Adaptive path
project_root = Path.cwd()
DB_PATH = str(project_root / 'arctic_store')
```

## Files Updated

- `config.py` - New adaptive configuration system
- `features/trend_indicator_pipeline_pkg.py` - Updated all pipeline classes
- `features/trend_indicator_pipeline_math.py` - Updated all pipeline classes

## Backup Files Created

Backup copies were created before modifications:
- `features/trend_indicator_pipeline_pkg.py.backup`
- `features/trend_indicator_pipeline_math.py.backup`

## Testing

The solution has been tested and verified to work correctly:
- Configuration loads successfully
- ArcticDB directory is created automatically  
- Pipeline classes initialize with adaptive paths
- Cross-platform compatibility ensured