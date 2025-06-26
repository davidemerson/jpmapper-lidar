# JPMapper-LiDAR Tools

This directory contains utility tools and diagnostic scripts for JPMapper-LiDAR.

## Available Tools

### `diagnose_data_issues.py`
Comprehensive diagnostic tool for analyzing potential data quality issues in jpmapper-lidar rasterization and analysis.

**Usage:**
```bash
python tools/diagnose_data_issues.py
```

**Features:**
- Analyzes DSM GeoTIFF quality and statistics
- Validates CSV point coordinate ranges
- Examines analysis results for missing data issues
- Tests coordinate transformation accuracy
- Provides actionable recommendations

This tool helps identify and troubleshoot:
- Issues with rasterized GeoTIFF DSM
- Coordinate transformation problems  
- Missing data in LAS files vs GeoTIFF
- Analysis parameter issues
