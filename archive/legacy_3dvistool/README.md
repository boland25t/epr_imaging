# Point Cloud Pipeline

A Python utility for converting 3D coordinate data (UTM XYZ) into grid-based point clouds with scalar coloring, exportable as PLY files.

## Overview

This pipeline takes your undersea survey data and:

1. **Loads** coordinate and scalar data from CSV
2. **Creates** a 3D grid of evenly-spaced cell centroids
3. **Aggregates** scalar values from original points to grid cells
4. **Exports** the grid as CSV or PLY (Point Cloud format)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Edit `example_usage.py` with your CSV file path and column names:
   ```python
   INPUT_CSV = "your_data.csv"
   X_COLUMN = "easting"          # UTM Easting
   Y_COLUMN = "northing"         # UTM Northing
   Z_COLUMN = "depth"            # Depth/elevation
   SCALAR_COLUMN = "temperature" # Values for coloring
   CELL_SIZE = 10.0              # Grid cell size (same units as coordinates)
   ```

2. Run the script:
   ```bash
   python example_usage.py
   ```

3. Open the generated `.ply` file in a 3D viewer (CloudCompare, ParaView, Blender, etc.)

## Core Components

### `PointCloudPipeline` Class

**Initialization:**
```python
pipeline = PointCloudPipeline(cell_size=10.0, boundary_extension=50.0)
```
- `cell_size`: Edge length of cube cells (in your data units)
- `boundary_extension`: Extra distance beyond min/max coordinates (set to 0 for tight bounds)

**Methods:**

#### `load_csv(filepath, x_col, y_col, z_col, scalar_col)`
Loads coordinate and scalar data from CSV.

```python
pipeline.load_csv(
    "survey_data.csv",
    x_col="easting",
    y_col="northing", 
    z_col="depth",
    scalar_col="temperature"
)
```

#### `create_3d_grid()`
Generates a 3D grid with cell centroids as a multi-indexed Pandas DataFrame.

```python
grid_df = pipeline.create_3d_grid()
# Multi-index: (z, x, y)
# Each row is a cell centroid
```

The resulting DataFrame has:
- **Index**: (z, x, y) — depth level, then x-y position
- **Rows**: One per cell centroid
- **Structure**: 2D slices (x-y planes) indexed by depth (z)

#### `add_scalar_data(filepath, x_col, y_col, z_col, scalar_col, aggregation='mean')`
Assigns scalar values from original data points to nearest grid cells.

```python
pipeline.add_scalar_data(
    "survey_data.csv",
    x_col="easting",
    y_col="northing",
    z_col="depth",
    scalar_col="temperature",
    aggregation='mean'  # 'mean', 'min', 'max', or 'count'
)
```

**Aggregation options:**
- `mean`: Average value of points in cell
- `min`: Minimum value
- `max`: Maximum value
- `count`: Number of points in cell

#### `to_csv(filepath, include_index=True)`
Export grid centroids to CSV for inspection or further processing.

```python
pipeline.to_csv("grid_output.csv")
```

#### `to_ply(filepath, scalar_col=None, binary=True)`
Export grid as PLY point cloud format.

```python
pipeline.to_ply(
    "point_cloud.ply",
    scalar_col="scalar",  # Column to use for grayscale coloring
    binary=True           # Binary is faster & smaller; ASCII is human-readable
)
```

**Output format:**
- Vertices: X, Y, Z coordinates
- Colors: RGB (derived from scalar via grayscale mapping)
- File size: Binary PLY is ~50% smaller than ASCII for large grids

#### `summary()`
Print a human-readable summary of pipeline state.

```python
print(pipeline.summary())
```

## Data Format Requirements

### Input CSV

Must contain columns for:
- **X coordinate** (UTM Easting or similar)
- **Y coordinate** (UTM Northing or similar)
- **Z coordinate** (Depth, elevation, etc.)
- **Scalar values** (Temperature, salinity, density, etc.)

Example:
```
easting,northing,depth,temperature,salinity
500000,5000000,0,15.2,35.1
500010,5000010,5,14.8,35.2
```

### Output PLY

PLY (Polygon File Format) stores:
- **Vertices**: X, Y, Z coordinates (float32)
- **Colors**: RGB values (uint8 0-255)

Supports viewing in:
- **CloudCompare** (recommended for point clouds)
- **ParaView** (scientific visualization)
- **Blender** (3D modeling)
- **MeshLab** (mesh/point processing)
- Many other 3D software packages

**File sizes:**
- Binary: ~16 bytes/point + header
- ASCII: ~100 bytes/point + header

## Multi-Indexed DataFrame Structure

The grid DataFrame uses a MultiIndex for efficient spatial organization:

```
Level 0 (z/depth):     [0, 0, 0, 5, 5, 5, 10, 10, 10, ...]
Level 1 (x):           [100, 100, 100, 100, 100, 100, 100, 100, 100, ...]
Level 2 (y):           [200, 210, 220, 200, 210, 220, 200, 210, 220, ...]
                       
Column 'scalar':       [15.2, 14.8, 15.5, 14.9, 15.1, 15.3, ...]
```

Benefits:
- Group by depth level: `grid_df.loc[0]` gets all x-y points at z=0
- Slice specific regions: `grid_df.loc[(0, 100), :]` gets all points at z=0, x=100
- Natural representation of horizontal slices with depth indexing

## Example Workflow

```python
from point_cloud_pipeline import PointCloudPipeline

# Setup
pipeline = PointCloudPipeline(cell_size=5.0, boundary_extension=25.0)

# Load data
pipeline.load_csv(
    "survey.csv",
    x_col="utm_east", y_col="utm_north", z_col="depth", scalar_col="temp"
)

# Create grid (empty cells)
pipeline.create_3d_grid()

# Populate with scalar values
pipeline.add_scalar_data(
    "survey.csv",
    x_col="utm_east", y_col="utm_north", z_col="depth", scalar_col="temp",
    aggregation='mean'
)

# Export
pipeline.to_csv("grid.csv")
pipeline.to_ply("cloud.ply", scalar_col="temp", binary=True)
```

## Troubleshooting

**Large PLY files?**
- Increase `cell_size` to reduce grid resolution
- Use `binary=True` in `to_ply()`
- Reduce `boundary_extension`

**Missing scalar values in PLY?**
- Ensure `add_scalar_data()` was called before `to_ply()`
- Pass `scalar_col` parameter to `to_ply()`

**Points not mapping to cells?**
- Cell size may be too large; try a smaller value
- Check that original data is within expected bounds

## Performance Notes

- Grid creation scales with cell count: $\text{cells} = \frac{(\text{x\_range} \times \text{y\_range} \times \text{z\_range})}{\text{cell\_size}^3}$
- Scalar aggregation uses nearest-neighbor assignment (fast, O(n))
- PLY export is I/O limited for large grids (1M+ points)

For datasets > 10M original points, consider:
1. Increasing cell size
2. Filtering data by region first
3. Using binary PLY format

## License

MIT (or your preferred license)
