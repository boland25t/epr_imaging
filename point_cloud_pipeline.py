"""
Point Cloud Pipeline: Convert 3D coordinates to grid-based PLY file

This module handles:
1. Loading coordinate and scalar data from CSV
2. Creating a 3D grid with user-defined cell size
3. Generating cell centroids in a multi-indexed Pandas DataFrame
4. Exporting to CSV and PLY formats
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Callable
import csv
from pathlib import Path
from PIL import Image
from plyfile import PlyElement, PlyData
import matplotlib.cm as cm


def idw_fill(power: float = 2, max_distance: Optional[float] = None) -> Callable:
    """
    Returns an IDW fill function compatible with apply_cell_function.

    Fills NaN (empty) cells by inverse-distance weighting from populated cells.
    Uses scipy cKDTree for fast neighbour lookup; processes empty cells in
    batches to keep memory usage bounded on large grids.

    Args:
        power: Distance weighting exponent. Higher values give more weight to
               nearby points (default 2).
        max_distance: If set, empty cells farther than this distance from all
                      known points remain NaN.

    Returns:
        A function (df: DataFrame) -> Series for use with apply_cell_function.
        The DataFrame passed in has columns: x, y, z, ix, iy, iz, scalar.
    """
    def _fill(df: pd.DataFrame) -> pd.Series:
        from scipy.spatial import cKDTree

        vals = df['scalar'].values.copy().astype(float)
        known_mask = ~np.isnan(vals)
        empty_mask = np.isnan(vals)

        if known_mask.sum() == 0 or empty_mask.sum() == 0:
            return pd.Series(vals, index=df.index)

        known_xyz  = df[['x', 'y', 'z']].values[known_mask]
        empty_xyz  = df[['x', 'y', 'z']].values[empty_mask]
        known_vals = vals[known_mask]

        tree       = cKDTree(known_xyz)
        k          = min(len(known_xyz), 64)
        dist_bound = max_distance if max_distance is not None else np.inf
        idw_vals   = np.full(len(empty_xyz), np.nan)

        # Batch over empty cells to keep peak memory bounded (~100 MB/batch)
        batch_size = 100_000
        for start in range(0, len(empty_xyz), batch_size):
            end        = min(start + batch_size, len(empty_xyz))
            dists, idx = tree.query(empty_xyz[start:end], k=k,
                                    distance_upper_bound=dist_bound, workers=-1)

            if k == 1:
                dists = dists[:, np.newaxis]
                idx   = idx[:, np.newaxis]

            in_range = dists < np.inf
            with np.errstate(divide='ignore'):
                w = np.where(dists == 0, np.inf,
                             np.where(in_range, dists ** (-power), 0.0))

            w_sum    = w.sum(axis=1)
            safe_idx = np.clip(idx, 0, len(known_vals) - 1)
            nbr_vals = np.where(in_range, known_vals[safe_idx], 0.0)

            with np.errstate(invalid='ignore'):
                idw_vals[start:end] = np.where(
                    w_sum > 0,
                    (w * nbr_vals).sum(axis=1) / w_sum,
                    np.nan
                )

        vals[empty_mask] = idw_vals
        return pd.Series(vals, index=df.index)

    return _fill


def ordinary_kriging_fill(variogram_model: str = "spherical", max_known: int = 2000) -> Callable:
    """
    Returns an Ordinary Kriging fill function compatible with apply_cell_function.

    Performs 2D Ordinary Kriging independently per Z (depth) layer.  Fitting
    the variogram per slice keeps memory and runtime tractable for trackline
    data, which is essentially a 1-D curve in 3-D space.

    Args:
        variogram_model: Variogram model type passed to pykrige OrdinaryKriging.
                         Options: 'spherical' (default), 'gaussian', 'exponential',
                         'linear', 'power'.
        max_known: Maximum number of known points used for variogram fitting per
                   Z slice.  If a slice has more, a random subsample is used.
                   Kriging system solve is O(n³), so keep this ≤ 3000.

    Returns:
        A function (df: DataFrame) -> Series for use with apply_cell_function.

    Raises:
        ImportError: If pykrige is not installed.
    """
    def _fill(df: pd.DataFrame) -> pd.Series:
        try:
            from pykrige.ok import OrdinaryKriging
        except ImportError:
            raise ImportError(
                "PyKrige is required for Ordinary Kriging. "
                "Install it with: pip install pykrige"
            )

        vals = df['scalar'].values.copy().astype(float)
        known_mask = ~np.isnan(vals)
        empty_mask = np.isnan(vals)

        if known_mask.sum() == 0 or empty_mask.sum() == 0:
            return pd.Series(vals, index=df.index)

        # Process each Z slice independently (2-D kriging per depth layer).
        # This keeps the covariance matrix tractable and respects the layered
        # structure of underwater sensor data.
        rng = np.random.default_rng(seed=0)  # deterministic subsampling
        z_levels = np.unique(df['z'].values)
        for z_val in z_levels:
            z_mask  = df['z'].values == z_val
            z_known = z_mask & known_mask
            z_empty = z_mask & empty_mask

            if z_known.sum() < 3 or z_empty.sum() == 0:
                continue

            x_k = df['x'].values[z_known]
            y_k = df['y'].values[z_known]
            v_k = vals[z_known]
            x_q = df['x'].values[z_empty]
            y_q = df['y'].values[z_empty]

            # Subsample for variogram fitting when there are many known points.
            n = len(x_k)
            if n > max_known:
                idx = rng.choice(n, max_known, replace=False)
                x_fit, y_fit, v_fit = x_k[idx], y_k[idx], v_k[idx]
            else:
                x_fit, y_fit, v_fit = x_k, y_k, v_k

            try:
                ok = OrdinaryKriging(
                    x_fit, y_fit, v_fit,
                    variogram_model=variogram_model,
                    verbose=False,
                    enable_plotting=False,
                )
                predicted, _ = ok.execute('points', x_q, y_q)
                vals[z_empty] = np.asarray(predicted)
            except Exception:
                # Leave this slice's empty cells as NaN rather than crashing.
                pass

        return pd.Series(vals, index=df.index)

    return _fill


def rbf_fill(kernel: str = "thin_plate_spline", neighbors: int = 50) -> Callable:
    """
    Returns an RBF fill function compatible with apply_cell_function.

    Uses scipy RBFInterpolator to fill NaN cells.  Suitable for smooth
    interpolation; slower than IDW on large grids.

    Args:
        kernel: RBF kernel type passed to RBFInterpolator (default 'thin_plate_spline').
        neighbors: Number of nearest neighbours used per query point.
    """
    def _fill(df: pd.DataFrame) -> pd.Series:
        from scipy.interpolate import RBFInterpolator

        vals = df['scalar'].values.copy().astype(float)
        known_mask = ~np.isnan(vals)
        empty_mask = np.isnan(vals)

        if known_mask.sum() == 0 or empty_mask.sum() == 0:
            return pd.Series(vals, index=df.index)

        known_xyz  = df[['x', 'y', 'z']].values[known_mask]
        empty_xyz  = df[['x', 'y', 'z']].values[empty_mask]
        known_vals = vals[known_mask]

        try:
            rbf = RBFInterpolator(
                known_xyz, known_vals,
                kernel=kernel,
                neighbors=min(neighbors, len(known_xyz)),
            )
            vals[empty_mask] = rbf(empty_xyz)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise ValueError(
                f"RBF interpolation failed: {exc}\n\n"
                "Vehicle tracklines are 1-D curves in 3-D space — the polynomial "
                "basis of RBFInterpolator (thin_plate_spline, default degree=1) "
                "becomes rank-deficient when depth is nearly constant.\n\n"
                "Fix: use IDW fill instead (it has no rank requirement)."
            ) from exc
        return pd.Series(vals, index=df.index)

    return _fill


class PointCloudPipeline:
    """Pipeline for creating point cloud PLY files from 3D coordinate data."""
    
    def __init__(self, cell_size: float, boundary_extension: float = 0.0,
                 log_fn: Optional[Callable[[str], None]] = None):
        """
        Initialize the pipeline.

        Args:
            cell_size: Size of cube cells in data units
            boundary_extension: Additional distance beyond min/max coordinates (in units)
            log_fn: Optional callable(str) for progress messages.  Defaults to
                    self._logln(); pass the app's log callback to route output to the GUI.
        """
        self.cell_size = cell_size
        self.boundary_extension = boundary_extension
        self.df_grid = None
        self.bounds = None
        self.scalar_name = 'scalar'
        self._log = log_fn or print

    def _logln(self, msg: str = "") -> None:
        """Log one line, stripping any leading newline used for spacing in prints."""
        self._log(msg.lstrip("\n"))
        
    def load_csv(self, filepath: str, x_col: str, y_col: str, z_col: str, 
                 scalar_col: str) -> pd.DataFrame:
        """
        Load coordinate and scalar data from CSV.
        
        Args:
            filepath: Path to CSV file
            x_col: Column name for X coordinate (or Easting for UTM)
            y_col: Column name for Y coordinate (or Northing for UTM)
            z_col: Column name for Z coordinate (or depth)
            scalar_col: Column name for scalar values to color by
            
        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(filepath)
        self.data = df[[x_col, y_col, z_col, scalar_col]].copy()
        self.data.columns = ['x', 'y', 'z', self.scalar_name]
        
        self._logln(f"Loaded {len(self.data)} points from {filepath}")
        self._logln(f"  X range: {self.data['x'].min():.2f} to {self.data['x'].max():.2f}")
        self._logln(f"  Y range: {self.data['y'].min():.2f} to {self.data['y'].max():.2f}")
        self._logln(f"  Z range: {self.data['z'].min():.2f} to {self.data['z'].max():.2f}")
        self._logln(f"  Scalar range: {self.data[self.scalar_name].min():.2f} to {self.data[self.scalar_name].max():.2f}")
        
        return self.data
    
    def _calculate_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate grid bounds based on data and boundary extension.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples
        """
        bounds = {
            'x': (self.data['x'].min() - self.boundary_extension,
                  self.data['x'].max() + self.boundary_extension),
            'y': (self.data['y'].min() - self.boundary_extension,
                  self.data['y'].max() + self.boundary_extension),
            'z': (self.data['z'].min() - self.boundary_extension,
                  self.data['z'].max() + self.boundary_extension),
        }
        self.bounds = bounds
        return bounds
    
    def create_3d_grid(self) -> pd.DataFrame:
        """
        Create a 3D grid of cell centroids with multi-indexed structure.
        
        The DataFrame has a MultiIndex on z (depth) and regular index for (x, y) pairs.
        
        Returns:
            Multi-indexed DataFrame with centroids
        """
        if self.data is None:
            raise ValueError("Load data first with load_csv()")
        
        bounds = self._calculate_bounds()

        # Compute grid dimensions from integer cell counts
        nx = int(np.ceil((bounds['x'][1] - bounds['x'][0]) / self.cell_size))
        ny = int(np.ceil((bounds['y'][1] - bounds['y'][0]) / self.cell_size))
        nz = int(np.ceil((bounds['z'][1] - bounds['z'][0]) / self.cell_size))

        total = nx * ny * nz
        self._logln(f"\nGrid dimensions:")
        self._logln(f"  X cells: {nx}")
        self._logln(f"  Y cells: {ny}")
        self._logln(f"  Z cells: {nz}")
        self._logln(f"  Total cells: {total:,}")
        _CELL_HARD_LIMIT = 100_000_000
        if total > _CELL_HARD_LIMIT:
            raise ValueError(
                f"Grid would have {total:,} cells — exceeds the {_CELL_HARD_LIMIT:,}-cell "
                f"safety limit.  Increase cell_size (currently {self.cell_size}) or reduce "
                f"the data extent."
            )
        if total > 1_000_000:
            self._logln(f"  Warning: large grid ({total:,} cells) — consider a larger cell_size")

        # Build centroids via meshgrid — vectorized, avoids Python loop overhead
        # and floating-point drift from np.arange accumulation
        ix_arr = np.arange(nx, dtype=np.int32)
        iy_arr = np.arange(ny, dtype=np.int32)
        iz_arr = np.arange(nz, dtype=np.int32)
        iz_grid, ix_grid, iy_grid = np.meshgrid(iz_arr, ix_arr, iy_arr, indexing='ij')

        df = pd.DataFrame({
            'iz': iz_grid.ravel(),
            'ix': ix_grid.ravel(),
            'iy': iy_grid.ravel(),
            'x': bounds['x'][0] + (ix_grid.ravel() + 0.5) * self.cell_size,
            'y': bounds['y'][0] + (iy_grid.ravel() + 0.5) * self.cell_size,
            'z': bounds['z'][0] + (iz_grid.ravel() + 0.5) * self.cell_size,
        })
        df_indexed = df.set_index(['z', 'x', 'y'])

        self.df_grid = df_indexed
        self._logln(f"\nCreated multi-indexed grid with {len(df_indexed)} centroids")

        return df_indexed
    
    def to_csv(self, filepath: str, include_index: bool = True) -> None:
        """
        Export grid centroids to CSV.
        
        Args:
            filepath: Output CSV file path
            include_index: Whether to include multi-index columns
        """
        if self.df_grid is None:
            raise ValueError("Create grid first with create_3d_grid()")
        
        # Reset index to make coordinates regular columns
        df_export = self.df_grid.reset_index()
        df_export.to_csv(filepath, index=False)
        
        self._logln(f"\nExported grid to {filepath}")
    
    def add_scalar_data(self, filepath: str, x_col: str, y_col: str, z_col: str,
                        scalar_col: str, aggregation: str = 'mean') -> None:
        """
        Add scalar values from data points to nearest grid cells.
        
        Args:
            filepath: Path to source data CSV
            x_col: Column name for X coordinate
            y_col: Column name for Y coordinate
            z_col: Column name for Z coordinate
            scalar_col: Column name for scalar values
            aggregation: 'mean', 'min', 'max', or 'count' for aggregating multiple values per cell
        """
        if self.df_grid is None:
            raise ValueError("Create grid first with create_3d_grid()")
        
        # Load data
        data = pd.read_csv(filepath)
        data = data[[x_col, y_col, z_col, scalar_col]].copy()
        data.columns = ['x', 'y', 'z', 'scalar_value']

        # Bin each data point to a cell using floor division on integer indices —
        # avoids float-matching errors from independent np.arange accumulation
        data['ix'] = np.floor((data['x'] - self.bounds['x'][0]) / self.cell_size).astype(int)
        data['iy'] = np.floor((data['y'] - self.bounds['y'][0]) / self.cell_size).astype(int)
        data['iz'] = np.floor((data['z'] - self.bounds['z'][0]) / self.cell_size).astype(int)

        # Aggregate scalar values by integer cell index
        aggregated = data.groupby(['iz', 'ix', 'iy'])['scalar_value'].agg(aggregation).reset_index()
        aggregated.columns = ['iz', 'ix', 'iy', self.scalar_name]

        # Merge on integer indices — exact, no float drift
        self.df_grid = self.df_grid.reset_index()
        self.df_grid = self.df_grid.merge(aggregated, on=['ix', 'iy', 'iz'], how='left')

        # Leave empty cells as NaN so to_ply can color them distinctly
        self.df_grid = self.df_grid.set_index(['z', 'x', 'y'])

        filled = self.df_grid[self.scalar_name].notna().sum()
        self._logln(f"Added scalar data: {aggregation} of {filled} cells with values")
    
    def apply_cell_function(self, func: Callable) -> None:
        """
        Apply a user-defined function to modify cell scalar values.

        The function receives the full grid DataFrame with columns:
            x, y, z        — cell centroid coordinates
            ix, iy, iz     — integer cell indices
            scalar         — current scalar value (NaN for empty cells)

        It must return a Series of updated scalar values with the same index.

        Args:
            func: Callable with signature (df: DataFrame) -> Series.
                  Use the built-in idw_fill() or supply your own.

        Example:
            from point_cloud_pipeline import idw_fill
            pipeline.apply_cell_function(idw_fill(power=2, max_distance=10))

            pipeline.apply_cell_function(lambda df: df['scalar'] * 1.1)
        """
        if self.df_grid is None:
            raise ValueError("Create grid first with create_3d_grid()")

        df = self.df_grid.reset_index()
        df[self.scalar_name] = func(df)
        self.df_grid = df.set_index(['z', 'x', 'y'])

        filled = self.df_grid[self.scalar_name].notna().sum()
        self._logln(f"Cell function applied: {filled} cells with values")

    def to_ply(self, filepath: str, scalar_col: Optional[str] = None,
               binary: bool = True, color_mode: str = 'rgb',
               show_empty: bool = True, log_scale: bool = False,
               percentile_cap: float = 100.0,
               zero_percentile: float = 0.0) -> None:
        """
        Export grid centroids to PLY format.

        Args:
            filepath: Output PLY file path
            scalar_col: Column name for coloring. If None, uses constant color.
            binary: If True, write binary PLY; if False, write ASCII
            color_mode: 'rgb' uses viridis colormap; 'grayscale' maps to gray ramp
            show_empty: If True, empty cells are white; if False, omitted
            log_scale: If True, apply log scaling before color mapping — spreads
                       low-end variation that linear scale would compress
            percentile_cap: Clip values above this percentile to that value before
                            color mapping (e.g. 99.0 caps the top 1% so extreme
                            hotspots don't dominate the color range)
        """
        if self.df_grid is None:
            raise ValueError("Create grid first with create_3d_grid()")

        df_export = self.df_grid.reset_index()

        # Determine which cells have data
        if scalar_col and scalar_col in df_export.columns:
            has_data = df_export[scalar_col].notna().values
        else:
            has_data = np.zeros(len(df_export), dtype=bool)

        # Optionally hide near-zero values (treat them as empty)
        if zero_percentile > 0.0 and has_data.any():
            vals = df_export[scalar_col].values.astype(float)
            threshold = np.percentile(vals[has_data], zero_percentile)
            has_data = has_data & (vals >= threshold)

        # Optionally drop empty cells
        if not show_empty:
            df_export = df_export[has_data].reset_index(drop=True)
            has_data = np.ones(len(df_export), dtype=bool)

        n = len(df_export)

        # Build RGB array — white for empty cells
        rgb = np.full((n, 3), 255, dtype=np.uint8)

        if scalar_col and scalar_col in df_export.columns:
            scalar_vals = df_export.loc[has_data, scalar_col].values.astype(float)

            if len(scalar_vals) > 0:
                # Ceiling: clip values above the chosen percentile
                if percentile_cap < 100.0:
                    cap_val = np.percentile(scalar_vals, percentile_cap)
                    clipped = np.clip(scalar_vals, None, cap_val)
                    self._logln(f"  Percentile cap ({percentile_cap}%): {cap_val:.2f}  "
                          f"(clipped {(scalar_vals > cap_val).sum()} cells)")
                else:
                    clipped = scalar_vals

                # Log scaling
                if log_scale:
                    floor = np.maximum(clipped.min(), 1e-9)
                    clipped = np.log(np.maximum(clipped, floor))

                lo, hi = clipped.min(), clipped.max()
                normalized = (
                    (clipped - lo) / (hi - lo)
                    if hi > lo
                    else np.full(len(clipped), 0.5)
                )

                if color_mode == 'grayscale':
                    gray   = (normalized * 255).astype(np.uint8)
                    colors = np.stack([gray, gray, gray], axis=1)
                else:
                    colors = (cm.viridis(normalized)[:, :3] * 255).astype(np.uint8)

                rgb[has_data] = colors

        vertex_data = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex_data['x'] = df_export['x'].values.astype(np.float32)
        vertex_data['y'] = df_export['y'].values.astype(np.float32)
        vertex_data['z'] = df_export['z'].values.astype(np.float32)
        vertex_data['red']   = rgb[:, 0]
        vertex_data['green'] = rgb[:, 1]
        vertex_data['blue']  = rgb[:, 2]

        vertex_el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_el], text=not binary).write(filepath)

        self._logln(f"\nExported to PLY: {filepath}")
        self._logln(f"  Format: {'binary' if binary else 'ASCII'}")
        self._logln(f"  Color mode: {color_mode}, log_scale={log_scale}")
        self._logln(f"  Empty cells: {'hidden' if not show_empty else 'white'}")
        self._logln(f"  Points: {len(vertex_data)}")
    
    def to_raster_slices(self, output_dir: str, altitude_step: float,
                         scalar_col: Optional[str] = None,
                         color_mode: str = 'rgb', log_scale: bool = False,
                         percentile_cap: float = 100.0,
                         pixels_per_cell: int = 1,
                         local_norm: bool = False) -> None:
        """
        Export 2D raster PNG slices at regular altitude intervals.

        Each PNG covers the full x-y extent of the grid. Cells with data are
        rendered as solid colour squares; empty cells are white.

        Args:
            output_dir: Directory to write PNG files into (created if needed)
            altitude_step: Vertical spacing between slice centres (same units
                           as z). Each slice aggregates all cells within
                           ±altitude_step/2 of the centre.
            scalar_col: Column to colour by (defaults to pipeline scalar name)
            color_mode: 'rgb' (viridis) or 'grayscale'
            log_scale: Apply log scaling before colour mapping
            percentile_cap: Clip values above this percentile before mapping
            pixels_per_cell: Output pixels per grid cell (default 1; use larger
                             values for bigger images with filled-in squares)
            local_norm: Normalize each slice to its own min/max instead of the
                        global range. Maximises contrast within each slice but
                        makes slices incomparable to each other.
        """
        if self.df_grid is None:
            raise ValueError("Create grid first with create_3d_grid()")

        scalar_col = scalar_col or self.scalar_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        df = self.df_grid.reset_index()

        # --- Early exit if grid has no data ---
        all_vals = df[scalar_col].dropna().values.astype(float)
        if len(all_vals) == 0:
            self._logln("No data to render.")
            return
        if df.empty or df['ix'].isna().all():
            self._logln("No grid cells to render.")
            return

        nx = int(df['ix'].max()) + 1
        ny = int(df['iy'].max()) + 1

        # --- Global colour normalization ---

        cap_val   = None
        floor_val = None

        if percentile_cap < 100.0:
            cap_val  = np.percentile(all_vals, percentile_cap)
            all_vals = np.clip(all_vals, None, cap_val)
            self._logln(f"  Percentile cap ({percentile_cap}%): {cap_val:.2f}")

        if log_scale:
            floor_val = max(all_vals.min(), 1e-9)
            all_vals  = np.log(np.maximum(all_vals, floor_val))

        global_min = all_vals.min()
        global_max = all_vals.max()

        # --- Slice centres ---
        z_min = df['z'].min()
        z_max = df['z'].max()
        if altitude_step <= 0:
            raise ValueError(
                f"altitude_step must be positive, got {altitude_step}."
            )

        half  = altitude_step / 2.0

        if altitude_step < self.cell_size:
            self._logln(f"  Warning: altitude_step ({altitude_step}) < cell_size "
                  f"({self.cell_size}); some slices may be empty")

        slice_centers = np.arange(z_min + half, z_max + half + 1e-9, altitude_step)
        self._logln(f"\nRendering {len(slice_centers)} slices into {output_dir}/")

        for center in slice_centers:
            band = df[(df['z'] >= center - half) & (df['z'] < center + half)]
            band = band.dropna(subset=[scalar_col])

            # Aggregate multiple z cells within the band onto the x-y plane
            agg = band.groupby(['ix', 'iy'])[scalar_col].mean().reset_index()

            # White background — shape (ny, nx, 3)
            img = np.full((ny, nx, 3), 255, dtype=np.uint8)

            if len(agg) > 0:
                vals = agg[scalar_col].values.astype(float)

                if cap_val is not None:
                    vals = np.clip(vals, None, cap_val)
                if log_scale:
                    vals = np.log(np.maximum(vals, floor_val))

                if local_norm:
                    s_min, s_max = vals.min(), vals.max()
                    normalized = (
                        np.clip((vals - s_min) / (s_max - s_min), 0, 1)
                        if s_max > s_min
                        else np.full(len(vals), 0.5)
                    )
                else:
                    normalized = (
                        np.clip((vals - global_min) / (global_max - global_min), 0, 1)
                        if global_max > global_min
                        else np.full(len(vals), 0.5)
                    )

                if color_mode == 'grayscale':
                    gray   = (normalized * 255).astype(np.uint8)
                    colors = np.stack([gray, gray, gray], axis=1)
                else:
                    colors = (cm.viridis(normalized)[:, :3] * 255).astype(np.uint8)

                xi = agg['ix'].values
                yi = ny - 1 - agg['iy'].values  # flip so north is up

                img[yi, xi] = colors

            if pixels_per_cell > 1:
                img = np.repeat(np.repeat(img, pixels_per_cell, axis=0),
                                pixels_per_cell, axis=1)

            fname = Path(output_dir) / f"slice_z{center:+09.3f}m.png"
            Image.fromarray(img, "RGB").save(str(fname))
            self._logln(f"  {fname.name}  ({len(agg)} cells with data)")

        self._logln(f"Done.")

    def summary(self) -> str:
        """Return a summary of the current pipeline state."""
        summary = []
        summary.append("=== Point Cloud Pipeline Summary ===")
        summary.append(f"Cell size: {self.cell_size}")
        summary.append(f"Boundary extension: {self.boundary_extension}")
        
        if self.data is not None:
            summary.append(f"Data points loaded: {len(self.data)}")
        
        if self.bounds:
            summary.append(f"Grid bounds:")
            for dim, (mn, mx) in self.bounds.items():
                summary.append(f"  {dim}: {mn:.2f} to {mx:.2f}")
        
        if self.df_grid is not None:
            summary.append(f"Grid centroids: {len(self.df_grid)}")
        
        return "\n".join(summary)
