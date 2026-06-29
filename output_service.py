"""
output_service.py — Dataset-level output generation (nav + sensor GeoTIFF and PLY).

Operates on interp_full.csv (the workspace-level full-resolution interpolation produced
by the "Build interp_full.csv" button on the Inputs tab).  Not tied to any specific job
or video interval.

Expected interp_full.csv columns (required for UTM-based outputs):
    unix_time, lat, lon, easting, northing, depth, water_depth, utm_zone, [sensor channels…]

Output directory structure written inside the workspace:

    outputs/
      nav_depth/<run_NNN>/nav_depth.tif
      nav_trackline/<run_NNN>/
          nav_trackline.ply
          grid.csv.gz          ← voxel grid cache for this run
          run.meta.json
          slices/<run_NNN>/    ← child slice sets
              *.png
              run.meta.json
      sensor_2d/<ch>/<run_NNN>/<ch>_2d.tif
      sensor_3d/<ch>/<run_NNN>/
          <ch>_3d.ply
          grid.csv.gz
          run.meta.json
          slices/<run_NNN>/
              *.png
              run.meta.json
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# Columns that are navigation/coordinate metadata, not sensor data values.
_NAV_META_COLS = frozenset({
    "unix_time", "lat", "lon", "easting", "northing", "depth",
    "water_depth", "alt", "heading", "pitch", "roll",
    "utm_zone", "frame_filename",
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sensor_channels_from_csv(path: str) -> list[str]:
    """Return numeric sensor-value columns from interp_full.csv.

    Reads only the header row for speed.  Excludes navigation/coordinate
    columns and non-numeric columns.  Returns an empty list if the file
    cannot be read or has no sensor columns.
    """
    try:
        df = pd.read_csv(path, nrows=2)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in _NAV_META_COLS]
    except Exception:
        return []


def _utm_zone_to_epsg(zone_str: str) -> int:
    """Convert a UTM zone string such as '33T' or '33N' to an EPSG integer code."""
    zone_num    = int("".join(c for c in zone_str if c.isdigit()))
    zone_letter = "".join(c for c in zone_str if c.isalpha()).upper()
    is_north    = zone_letter >= "N"   # D–M south, N–X north
    return (32600 if is_north else 32700) + zone_num


def _check_utm_cols(df: pd.DataFrame) -> None:
    missing = [c for c in ("easting", "northing", "depth") if c not in df.columns]
    if missing:
        raise ValueError(
            f"interp_full.csv is missing UTM columns: {missing}. "
            "Run 'Build interp_full.csv' again to regenerate with UTM coordinates."
        )


@contextmanager
def _neg_depth_csv(interp_path: str):
    """Context manager that yields a temp CSV path where Z = -depth.

    The 'depth' column in interp_full.csv is a positive distance-from-surface
    value.  Negating it places the vehicle track at negative Z so that, in
    standard 3-D viewers (CloudCompare, etc.), deeper points appear lower —
    matching the physical convention used in the 3dvistool.
    """
    df = pd.read_csv(interp_path)
    df["_z"] = -df["depth"]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="epr_3d_"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    try:
        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OutputService
# ---------------------------------------------------------------------------

class OutputService:
    """Generates dataset-level outputs (GeoTIFF, PLY) from interp_full.csv."""

    def __init__(self, log_fn=None):
        self._log = log_fn or print

    # ------------------------------------------------------------------
    # Run-directory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_run_dir(parent: Path) -> Path:
        """Create and return the next numbered run_NNN subdirectory under parent.

        Each call to a generate method creates a new versioned directory so
        previous outputs are never overwritten.  Run numbers start at 001 and
        increment atomically by scanning existing siblings.
        """
        parent.mkdir(parents=True, exist_ok=True)
        n = 1
        for d in parent.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    n = max(n, int(d.name[4:]) + 1)
                except ValueError:
                    pass
        run_dir = parent / f"run_{n:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _save_grid_csv(pipeline, path: str) -> None:
        df = pipeline.df_grid.reset_index()
        df.dropna(subset=[pipeline.scalar_name]).to_csv(
            path, index=False, compression="gzip"
        )

    @staticmethod
    def _pipeline_from_grid_csv(path: str, log_fn=None):
        """Reconstruct a PointCloudPipeline from a cached grid CSV.

        Returns None if the file is empty (from a pre-guard stale save),
        so callers fall back to a full rebuild.
        """
        from point_cloud_pipeline import PointCloudPipeline
        df = pd.read_csv(path, compression="gzip")
        if df.empty:
            return None
        z_unique = np.array(sorted(df["z"].unique()))
        # Use the minimum step between consecutive Z levels to guard against
        # floating-point edge cases at grid boundaries.
        cell_size = float(np.diff(z_unique).min()) if len(z_unique) > 1 else 1.0
        pipeline = PointCloudPipeline(cell_size=cell_size, log_fn=log_fn)
        pipeline.df_grid = df.set_index(["z", "x", "y"])
        pipeline.scalar_name = "scalar"
        return pipeline

    @staticmethod
    def _save_meta(path: str, meta: dict) -> None:
        """Write product metadata alongside an output file or into an output directory.

        For file products (PLY, GeoTIFF) the metadata is written next to the file
        as ``<file>.meta.json``.  For directory products (raster slices) the
        metadata is written inside the directory as ``run.meta.json``.
        """
        meta["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p = Path(path)
        meta_path = (p / "run.meta.json") if p.is_dir() else Path(str(path) + ".meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Shared grid builders (used by both PLY and raster-slice methods)
    # ------------------------------------------------------------------

    def _build_nav_grid(self, interp_path: str, run_dir: Path, cell_size: float):
        """Return a PointCloudPipeline with the nav water-depth grid.

        Stores grid.csv.gz inside run_dir so subsequent slice calls can reload
        it without re-building from interp_full.csv.
        """
        from point_cloud_pipeline import PointCloudPipeline

        cache_path = run_dir / "grid.csv.gz"
        if cache_path.exists():
            pipeline = self._pipeline_from_grid_csv(str(cache_path), log_fn=self._log)
            if pipeline is not None:
                self._log(f"Nav grid ({cell_size} m): loaded from run cache.")
                return pipeline
            self._log(f"Nav grid ({cell_size} m): cache empty — rebuilding…")
        else:
            self._log(f"Nav grid ({cell_size} m): building from interp_full.csv…")

        df_check = pd.read_csv(interp_path, nrows=1)
        _check_utm_cols(df_check)
        if "water_depth" not in df_check.columns:
            raise ValueError(
                "interp_full.csv is missing 'water_depth' column. "
                "Regenerate interp_full.csv."
            )

        with _neg_depth_csv(interp_path) as tmp_csv:
            pipeline = PointCloudPipeline(cell_size=cell_size, log_fn=self._log)
            pipeline.load_csv(tmp_csv, "easting", "northing", "_z", "water_depth")
            pipeline.create_3d_grid()
            pipeline.add_scalar_data(tmp_csv, "easting", "northing", "_z", "water_depth")
            n_filled = int(pipeline.df_grid[pipeline.scalar_name].notna().sum())
            self._log(f"  Nav grid: {n_filled} cells with data")
            if n_filled == 0:
                raise ValueError(
                    "No water_depth data snapped to the 3D grid. "
                    "Check that interp_full.csv has valid easting/northing/depth columns."
                )
            self._save_grid_csv(pipeline, str(cache_path))

        return pipeline

    def _build_sensor_grid(
        self,
        interp_path: str,
        run_dir: Path,
        channel: str,
        cell_size: float,
        aggregation: str,
        fill_method: str,
    ):
        """Return a PointCloudPipeline with a sensor channel grid.

        Stores grid.csv.gz inside run_dir so subsequent slice calls can reload
        it without re-building from interp_full.csv.
        """
        from point_cloud_pipeline import PointCloudPipeline, idw_fill, ordinary_kriging_fill, rbf_fill

        cache_path = run_dir / "grid.csv.gz"
        if cache_path.exists():
            pipeline = self._pipeline_from_grid_csv(str(cache_path), log_fn=self._log)
            if pipeline is not None:
                self._log(f"Sensor grid ({channel}, {cell_size} m, {fill_method}): loaded from run cache.")
                return pipeline
            self._log(f"Sensor grid ({channel}): cache empty — rebuilding…")
        else:
            self._log(f"Sensor grid ({channel}, {cell_size} m, {fill_method}): building…")

        df_head = pd.read_csv(interp_path, nrows=1)
        if channel not in df_head.columns:
            raise ValueError(
                f"Column '{channel}' not found in interp_full.csv. "
                "Regenerate interp_full.csv if you recently added this sensor channel."
            )
        _check_utm_cols(df_head)

        with _neg_depth_csv(interp_path) as tmp_csv:
            pipeline = PointCloudPipeline(cell_size=cell_size, log_fn=self._log)
            pipeline.load_csv(tmp_csv, "easting", "northing", "_z", channel)
            pipeline.create_3d_grid()
            pipeline.add_scalar_data(
                tmp_csv, "easting", "northing", "_z", channel,
                aggregation=aggregation,
            )
            n_raw = int(pipeline.df_grid[pipeline.scalar_name].notna().sum())
            self._log(f"  {channel}: {n_raw} cells with raw data")
            if n_raw == 0:
                raise ValueError(
                    f"No '{channel}' data snapped to the 3D grid. "
                    "Check that interp_full.csv has valid sensor values for this channel."
                )
            if fill_method == "idw":
                pipeline.apply_cell_function(idw_fill())
            elif fill_method == "rbf":
                pipeline.apply_cell_function(rbf_fill())
            elif fill_method == "kriging":
                pipeline.apply_cell_function(ordinary_kriging_fill())
            n_filled = int(pipeline.df_grid[pipeline.scalar_name].notna().sum())
            self._log(f"  {channel}: {n_filled} cells after fill")
            self._save_grid_csv(pipeline, str(cache_path))

        return pipeline

    # ------------------------------------------------------------------
    # 3D PLY outputs — always in UTM (metric; physically meaningful for 3D)
    # ------------------------------------------------------------------

    def generate_nav_3d_ply(
        self,
        interp_path: str,
        output_dir: str,
        cell_size: float = 1.0,
    ) -> str:
        """Write a nav trackline PLY coloured by water depth.

        Each call creates a new versioned run directory containing:
          nav_trackline.ply, grid.csv.gz, run.meta.json

        Returns the run directory path so it can be used as the target for
        generate_nav_slices_from_run.
        """
        run_dir = self._next_run_dir(Path(output_dir) / "nav_trackline")
        out_path = str(run_dir / "nav_trackline.ply")

        pipeline = self._build_nav_grid(interp_path, run_dir, cell_size)
        pipeline.to_ply(
            out_path,
            scalar_col=pipeline.scalar_name,
            binary=True,
            color_mode="rgb",
            show_empty=False,
        )
        interp_stat = Path(interp_path).stat()
        self._save_meta(str(run_dir), {
            "product": "nav_3d_ply",
            "settings": {"cell_size_m": cell_size},
            "interp_source": {
                "mtime": interp_stat.st_mtime,
                "size":  interp_stat.st_size,
            },
        })
        self._log(f"Nav 3D trackline PLY: {out_path}")
        return str(run_dir)

    def generate_sensor_3d_ply(
        self,
        interp_path: str,
        output_dir: str,
        channel: str,
        cell_size: float = 1.0,
        aggregation: str = "mean",
        fill_method: str = "idw",
        zero_mask_pct: float = 0.0,
    ) -> str:
        """Write volumetric PLY(s) for one sensor channel.

        Always writes {channel}_3d.ply (all non-empty cells).
        When zero_mask_pct > 0, also writes {channel}_3d_signal.ply with
        the bottom N % of values removed, making near-zero background invisible.

        Each call creates a new versioned run directory containing:
          {channel}_3d.ply, [{channel}_3d_signal.ply], grid.csv.gz, run.meta.json

        Returns the run directory path.
        """
        run_dir  = self._next_run_dir(Path(output_dir) / "sensor_3d" / channel)
        out_path = str(run_dir / f"{channel}_3d.ply")

        pipeline = self._build_sensor_grid(
            interp_path, run_dir, channel, cell_size, aggregation, fill_method
        )
        pipeline.to_ply(
            out_path,
            scalar_col=pipeline.scalar_name,
            binary=True,
            color_mode="rgb",
            show_empty=False,
        )
        self._log(f"Sensor 3D PLY ({channel}): {out_path}")

        if zero_mask_pct > 0.0:
            signal_path = str(run_dir / f"{channel}_3d_signal.ply")
            pipeline.to_ply(
                signal_path,
                scalar_col=pipeline.scalar_name,
                binary=True,
                color_mode="rgb",
                show_empty=False,
                zero_percentile=zero_mask_pct,
            )
            self._log(f"Signal PLY ({channel}, bottom {zero_mask_pct:.0f}% hidden): {signal_path}")

        interp_stat = Path(interp_path).stat()
        self._save_meta(str(run_dir), {
            "product": "sensor_3d_ply",
            "channel": channel,
            "settings": {
                "cell_size_m":   cell_size,
                "aggregation":   aggregation,
                "fill_method":   fill_method,
                "zero_mask_pct": zero_mask_pct,
            },
            "interp_source": {
                "mtime": interp_stat.st_mtime,
                "size":  interp_stat.st_size,
            },
        })
        return str(run_dir)

    # ------------------------------------------------------------------
    # CF-compliant gridded NetCDF output
    # ------------------------------------------------------------------

    def generate_sensor_netcdf(
        self,
        interp_path: str,
        output_dir: str,
        channel: str,
        cell_size: float = 1.0,
        aggregation: str = "mean",
        fill_method: str = "idw",
        units: str = "",
    ) -> str:
        """Write a CF-1.8 gridded NetCDF (.nc) for one sensor channel.

        Builds the same regular 3-D grid used for the sensor 3D PLY, then
        writes it as a dense depth × northing × easting array with CF metadata
        and a UTM grid_mapping so it loads in xarray / QGIS / Panoply.  Reuses
        the run's cached grid.csv.gz when present.

        Each call creates sensor_netcdf/{channel}/run_NNN/{channel}.nc.

        Returns the path to the written .nc file.
        """
        from netcdf_export import write_gridded_netcdf, grid_to_arrays

        df_head = pd.read_csv(interp_path, nrows=1)
        _check_utm_cols(df_head)
        epsg = None
        try:
            cols = [c for c in ("utm_zone", "lat", "lon") if c in df_head.columns]
            epsg = self._utm_epsg_from_df(pd.read_csv(interp_path, usecols=cols) if cols
                                          else df_head)
        except Exception:
            pass

        run_dir = self._next_run_dir(Path(output_dir) / "sensor_netcdf" / channel)
        pipeline = self._build_sensor_grid(
            interp_path, run_dir, channel, cell_size, aggregation, fill_method
        )

        easting, northing, depth, data = grid_to_arrays(
            pipeline.df_grid.reset_index(), scalar_col=pipeline.scalar_name
        )
        out_path = str(run_dir / f"{channel}.nc")
        write_gridded_netcdf(
            out_path,
            easting=easting, northing=northing, depth=depth, data=data,
            channel=channel, units=units, epsg=epsg,
            title=f"{channel} gridded sensor field",
            log_fn=self._log,
        )

        self._save_meta(str(run_dir / "run.meta.json"), {
            "product": "sensor_netcdf",
            "channel": channel,
            "settings": {
                "cell_size_m": cell_size,
                "aggregation": aggregation,
                "fill_method": fill_method,
                "units": units,
                "epsg": epsg,
            },
        })
        self._log(f"Sensor NetCDF ({channel}): {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Raster slice outputs (PNG per depth band) — target-based
    #
    # Slices are generated FROM an existing 3D run directory (which already
    # contains grid.csv.gz).  This means cell_size, aggregation, and fill
    # are locked to the parent 3D run; only display parameters vary per
    # slice set.  Slice sets live under {3d_run_dir}/slices/run_NNN/.
    # ------------------------------------------------------------------

    def generate_nav_slices_from_run(
        self,
        run_dir: str,
        altitude_step: float = 5.0,
        pixels_per_cell: int = 4,
    ) -> str:
        """Write depth-slice PNGs from an existing nav 3D model run.

        run_dir must be a previously generated nav_trackline run directory
        containing grid.csv.gz (produced by generate_nav_3d_ply).

        Returns the absolute path of the new slice run directory.
        """
        run_dir_path = Path(run_dir)
        grid_path = run_dir_path / "grid.csv.gz"
        if not grid_path.exists():
            raise FileNotFoundError(
                f"grid.csv.gz not found in {run_dir}. "
                "Generate a 3D PLY first — it builds the grid."
            )

        pipeline = self._pipeline_from_grid_csv(str(grid_path), log_fn=self._log)
        if pipeline is None:
            raise ValueError(f"grid.csv.gz in {run_dir} is empty — regenerate the 3D PLY.")

        slice_run_dir = self._next_run_dir(run_dir_path / "slices")
        pipeline.to_raster_slices(
            str(slice_run_dir),
            altitude_step=altitude_step,
            scalar_col=pipeline.scalar_name,
            color_mode="rgb",
            pixels_per_cell=pixels_per_cell,
            legend_label="depth (m)",
        )
        self._save_meta(str(slice_run_dir), {
            "product": "nav_raster_slices",
            "settings": {
                "altitude_step_m": altitude_step,
                "pixels_per_cell": pixels_per_cell,
                "source_run": run_dir_path.name,
            },
        })
        self._log(f"Nav raster slices: {slice_run_dir}")
        return str(slice_run_dir)

    def generate_sensor_slices_from_run(
        self,
        run_dir: str,
        altitude_step: float = 5.0,
        pixels_per_cell: int = 4,
        color_mode: str = "rgb",
        log_scale: bool = False,
        percentile_cap: float = 100.0,
        local_norm: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> str:
        """Write depth-slice PNGs from an existing sensor 3D model run.

        run_dir must be a previously generated sensor_3d/<channel>/run_NNN
        directory containing grid.csv.gz (produced by generate_sensor_3d_ply).

        Returns the absolute path of the new slice run directory.
        """
        run_dir_path = Path(run_dir)
        grid_path = run_dir_path / "grid.csv.gz"
        if not grid_path.exists():
            raise FileNotFoundError(
                f"grid.csv.gz not found in {run_dir}. "
                "Generate a 3D PLY first — it builds the grid."
            )

        pipeline = self._pipeline_from_grid_csv(str(grid_path), log_fn=self._log)
        if pipeline is None:
            raise ValueError(f"grid.csv.gz in {run_dir} is empty — regenerate the 3D PLY.")

        channel = run_dir_path.parent.name  # sensor_3d/{channel}/run_NNN
        slice_run_dir = self._next_run_dir(run_dir_path / "slices")
        pipeline.to_raster_slices(
            str(slice_run_dir),
            altitude_step=altitude_step,
            scalar_col=pipeline.scalar_name,
            color_mode=color_mode,
            log_scale=log_scale,
            percentile_cap=percentile_cap,
            pixels_per_cell=pixels_per_cell,
            local_norm=local_norm,
            vmin=vmin,
            vmax=vmax,
            legend_label=channel,
        )
        self._save_meta(str(slice_run_dir), {
            "product": "sensor_raster_slices",
            "channel": channel,
            "settings": {
                "altitude_step_m": altitude_step,
                "pixels_per_cell": pixels_per_cell,
                "color_mode": color_mode,
                "log_scale": log_scale,
                "percentile_cap": percentile_cap,
                "local_norm": local_norm,
                "vmin": vmin,
                "vmax": vmax,
                "source_run": run_dir_path.name,
            },
        })
        self._log(f"Sensor raster slices: {slice_run_dir}")
        return str(slice_run_dir)

    # ------------------------------------------------------------------
    # QGIS project export
    # ------------------------------------------------------------------

    def generate_qgis_project(
        self,
        output_dir: str,
        project_name: str = "EPR Survey",
    ) -> str:
        """Scan output_dir for all GeoTIFFs and write a .qgs project file.

        Returns the absolute path of the written .qgs file.  The project
        opens in QGIS 3.x with all 2D and depth-slice layers already loaded,
        styled with the viridis colormap, and organized into named groups.

        Args:
            output_dir:   Root outputs directory (same value passed to
                          generate_nav_2d_geotiff / generate_sensor_2d_geotiff).
            project_name: Human-readable project name embedded in the .qgs file.
        """
        from qgis_export import generate_qgis_project as _gen
        path = _gen(output_dir=output_dir, project_name=project_name)
        self._log(f"QGIS project: {path}")
        return path

    # ------------------------------------------------------------------
    # 2D GeoTIFF outputs
    # ------------------------------------------------------------------

    def generate_nav_2d_geotiff(
        self,
        interp_path: str,
        output_dir: str,
        cell_size_m: float = 5.0,
        crs_mode: str = "utm",
    ) -> str:
        """Write a nav trackline GeoTIFF raster (water depth values along track).

        Args:
            interp_path: Path to interp_full.csv.
            output_dir:  Root output directory; file written to <output_dir>/nav/.
            cell_size_m: Raster cell/pixel size in metres.
            crs_mode:    'utm' (native UTM EPSG) or 'wgs84' (reproject to EPSG:4326).

        Returns:
            Absolute path of the written GeoTIFF file.
        """
        df = pd.read_csv(interp_path)
        _check_utm_cols(df)
        if "water_depth" not in df.columns:
            raise ValueError("interp_full.csv is missing 'water_depth' column.")

        epsg = self._utm_epsg_from_df(df)
        run_dir = self._next_run_dir(Path(output_dir) / "nav_depth")
        out_path = str(run_dir / "nav_depth.tif")

        self._write_geotiff(
            x=df["easting"].values,
            y=df["northing"].values,
            values=df["water_depth"].values,
            cell_size=cell_size_m,
            output_path=out_path,
            epsg=epsg,
            crs_mode=crs_mode,
            fill_method="none",
        )
        self._save_meta(out_path, {
            "product": "nav_2d_geotiff",
            "settings": {"cell_size_m": cell_size_m, "crs": crs_mode},
        })
        self._log(f"Nav 2D GeoTIFF: {out_path}")
        return out_path

    def generate_sensor_2d_geotiff(
        self,
        interp_path: str,
        output_dir: str,
        channel: str,
        cell_size_m: float = 5.0,
        crs_mode: str = "utm",
        fill_method: str = "idw",
    ) -> str:
        """Write a 2D GeoTIFF for one sensor channel.

        Args:
            interp_path: Path to interp_full.csv.
            output_dir:  Root output directory; file written to <output_dir>/sensor_2d/<channel>/.
            channel:     Column name of the sensor channel.
            cell_size_m: Raster cell/pixel size in metres.
            crs_mode:    'utm' or 'wgs84'.
            fill_method: 'none' (trackline only), 'idw', or 'rbf'.

        Returns:
            Absolute path of the written GeoTIFF file.
        """
        df = pd.read_csv(interp_path)
        _check_utm_cols(df)
        if channel not in df.columns:
            raise ValueError(
                f"Column '{channel}' not found in interp_full.csv. "
                "Regenerate interp_full.csv if you recently added this sensor channel."
            )

        epsg = self._utm_epsg_from_df(df)
        run_dir = self._next_run_dir(Path(output_dir) / "sensor_2d" / channel)
        out_path = str(run_dir / f"{channel}_2d.tif")

        self._write_geotiff(
            x=df["easting"].values,
            y=df["northing"].values,
            values=df[channel].values,
            cell_size=cell_size_m,
            output_path=out_path,
            epsg=epsg,
            crs_mode=crs_mode,
            fill_method=fill_method,
        )
        self._save_meta(out_path, {
            "product": "sensor_2d_geotiff",
            "channel": channel,
            "settings": {
                "cell_size_m": cell_size_m,
                "crs": crs_mode,
                "fill_method": fill_method,
            },
        })
        self._log(f"Sensor 2D GeoTIFF ({channel}): {out_path}")
        return out_path

    def generate_depth_slice_geotiffs(
        self,
        interp_path: str,
        output_dir: str,
        channel: str,
        altitude_step: float = 5.0,
        cell_size_m: float = 2.0,
        crs_mode: str = "utm",
        fill_method: str = "idw",
    ) -> list[str]:
        """Write one Float32 GeoTIFF per depth band for a sensor channel.

        Each raster is a plan-view (bird's-eye) map of the sensor value at a
        fixed depth band, covering the full survey extent in UTM coordinates.
        Cells with no data are set to NaN.  Output files are named:

            sensor_slices/{channel}/run_NNN/{channel}_z{depth:+08.2f}m.tif

        These load directly into QGIS as a depth-layered series.  Unlike the
        PNG slice outputs from PointCloudPipeline, these files carry true
        floating-point sensor values and a georeference so QGIS can apply any
        colormap and perform spatial analysis.

        Args:
            interp_path:   Path to interp_full.csv (or filtered_interp.csv).
            output_dir:    Root outputs directory; run dir created underneath.
            channel:       Sensor column name in interp_full.csv.
            altitude_step: Vertical thickness of each depth band in metres.
            cell_size_m:   Horizontal resolution of each raster in metres.
            crs_mode:      'utm' or 'wgs84'.
            fill_method:   'none', 'idw', or 'rbf'.

        Returns:
            List of absolute paths to the written GeoTIFF files (one per slice).
        """
        df = pd.read_csv(interp_path)
        _check_utm_cols(df)
        if channel not in df.columns:
            raise ValueError(f"Column '{channel}' not found in interp_full.csv.")
        df = df.dropna(subset=["easting", "northing", "depth", channel])
        if df.empty:
            raise ValueError(f"No valid rows for channel '{channel}'.")

        epsg = self._utm_epsg_from_df(df)
        run_dir = self._next_run_dir(Path(output_dir) / "sensor_slices" / channel)
        run_dir.mkdir(parents=True, exist_ok=True)

        z_vals = df["depth"].values
        z_min  = z_vals.min()
        z_max  = z_vals.max()
        half   = altitude_step / 2.0

        slice_centers = np.arange(z_min + half, z_max + half + 1e-9, altitude_step)
        self._log(
            f"Depth-slice GeoTIFFs ({channel}): {len(slice_centers)} slices "
            f"@ {altitude_step} m step, {cell_size_m} m/cell"
        )

        out_paths: list[str] = []
        for z_center in slice_centers:
            band = df[(df["depth"] >= z_center - half) & (df["depth"] < z_center + half)]
            if band.empty:
                continue

            # Aggregate multiple readings in the same band to one value per xy point
            band_agg = (
                band.groupby(["easting", "northing"])[channel]
                .mean()
                .reset_index()
            )

            fname    = f"{channel}_z{z_center:+08.2f}m.tif"
            out_path = str(run_dir / fname)

            self._write_geotiff(
                x=band_agg["easting"].values,
                y=band_agg["northing"].values,
                values=band_agg[channel].values,
                cell_size=cell_size_m,
                output_path=out_path,
                epsg=epsg,
                crs_mode=crs_mode,
                fill_method=fill_method,
            )
            out_paths.append(out_path)
            self._log(f"  z = {z_center:+.1f} m → {fname}")

        self._save_meta(str(run_dir / "run.meta.json"), {
            "product": "sensor_slice_geotiffs",
            "channel": channel,
            "settings": {
                "altitude_step_m": altitude_step,
                "cell_size_m": cell_size_m,
                "crs": crs_mode,
                "fill_method": fill_method,
                "slice_count": len(out_paths),
            },
        })
        self._log(f"Depth-slice GeoTIFFs ({channel}): {len(out_paths)} files in {run_dir}")
        return out_paths

    # ------------------------------------------------------------------
    # Data quality / QC report
    # ------------------------------------------------------------------

    def generate_qc_report(
        self,
        interp_path: str,
        output_dir: str,
        sensor_files: "list | None" = None,
        channels: "list[str] | None" = None,
        max_gap_s: float = 60.0,
    ) -> list[str]:
        """Write a per-run data-quality report for the frame record.

        Summarises, per sensor channel: value statistics + histogram, source
        coverage window, frames that fell outside coverage (clamped values),
        and source gaps larger than max_gap_s (bridged values).  Coverage/gap
        analysis is only possible when the originating SensorFileConfig list is
        supplied so the raw source timestamps can be re-read; otherwise only
        value statistics are reported.

        Args:
            interp_path:  Path to interp_full.csv (or a filtered interp.csv).
            output_dir:   Root outputs directory; a qc_report/run_NNN is created.
            sensor_files: Optional list of SensorFileConfig used to reload the
                          source timestamps for coverage/gap analysis.
            channels:     Restrict the report to these channels; None ⇒ every
                          configured sensor channel found in the CSV.
            max_gap_s:    Source spacing above which a span counts as a gap.

        Returns:
            List of written file paths (report, json, then histogram PNGs).
        """
        import qc_report
        from sensor_service import SensorService

        df = pd.read_csv(interp_path)
        if "unix_time" not in df.columns:
            raise ValueError("interp CSV has no 'unix_time' column — cannot build QC report.")

        # Map each requested channel to its source unix-time array (or None).
        channel_sources: dict[str, "np.ndarray | None"] = {}
        for cfg in (sensor_files or []):
            try:
                src_df = SensorService.load_sensor_dataframe(cfg)
            except Exception as exc:
                self._log(f"  QC: could not reload {getattr(cfg, 'csv_path', '?')}: {exc}")
                src_df = None
            for ch in getattr(cfg, "channels", []):
                name = ch.display_name
                if channels and name not in channels:
                    continue
                if name not in df.columns:
                    continue
                if src_df is not None and ch.source_column in src_df.columns:
                    present = src_df[src_df[ch.source_column].notna()]
                    channel_sources[name] = present["unix_time"].to_numpy(dtype=float)
                else:
                    channel_sources[name] = None

        # If no sensor configs were supplied, still report stats for any channel
        # the caller named (or every non-coordinate numeric column).
        if not channel_sources:
            skip = {"unix_time", "timestamp", "lat", "lon", "latitude", "longitude",
                    "easting", "northing", "depth", "alt", "altitude", "heading",
                    "pitch", "roll", "utm_zone"}
            for col in df.columns:
                if channels and col not in channels:
                    continue
                if col in skip:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    channel_sources[col] = None

        run_dir = self._next_run_dir(Path(output_dir) / "qc_report")
        written = qc_report.generate_report(
            df, channel_sources, str(run_dir),
            max_gap_s=max_gap_s, log_fn=self._log,
        )
        self._save_meta(str(run_dir / "run.meta.json"), {
            "product": "qc_report",
            "settings": {
                "max_gap_s": max_gap_s,
                "channels": list(channel_sources.keys()),
                "interp_source": Path(interp_path).name,
            },
        })
        self._log(f"QC report: {len(written)} file(s) in {run_dir}")
        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utm_epsg_from_df(df: pd.DataFrame) -> int:
        """Determine the UTM EPSG code from the utm_zone column or data centroid."""
        if "utm_zone" in df.columns:
            zone_str = df["utm_zone"].dropna()
            if len(zone_str):
                return _utm_zone_to_epsg(str(zone_str.iloc[0]))
        if "lat" in df.columns and "lon" in df.columns:
            try:
                import utm as _utm_lib
                lat = float(df["lat"].mean())
                lon = float(df["lon"].mean())
                _, _, zone_num, zone_letter = _utm_lib.from_latlon(lat, lon)
                is_north = zone_letter >= "N"
                return (32600 if is_north else 32700) + zone_num
            except Exception:
                pass
        return 32633  # fallback

    def _write_geotiff(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        cell_size: float,
        output_path: str,
        epsg: int,
        crs_mode: str = "utm",
        fill_method: str = "none",
    ) -> None:
        """Rasterise (x, y, values) onto a regular grid and write a GeoTIFF.

        All computation is done in the UTM coordinate space for metric
        distances.  When crs_mode='wgs84' the raster is warped to EPSG:4326
        as a final step using rasterio.warp.

        Args:
            x, y:        UTM easting/northing coordinates (1-D arrays).
            values:      Scalar values at each (x, y) point.
            cell_size:   Pixel size in metres.
            output_path: Destination file path.
            epsg:        UTM EPSG code of the source coordinates.
            crs_mode:    'utm' or 'wgs84'.
            fill_method: 'none', 'idw', 'rbf', or 'kriging'.
        """
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS

        # Drop NaN/inf rows
        mask = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
        x, y, values = x[mask], y[mask], values[mask]
        if len(x) == 0:
            raise ValueError("No valid (finite) data points to rasterise.")

        pad   = cell_size * 3
        x_min = x.min() - pad
        x_max = x.max() + pad
        y_min = y.min() - pad
        y_max = y.max() + pad

        nx = max(2, int((x_max - x_min) / cell_size) + 1)
        ny = max(2, int((y_max - y_min) / cell_size) + 1)
        self._log(f"  Raster grid: {nx} × {ny} cells @ {cell_size} m/cell")

        grid = np.full((ny, nx), np.nan, dtype=np.float32)

        if fill_method in ("idw", "rbf", "kriging"):
            xi   = np.linspace(x_min + cell_size / 2, x_max - cell_size / 2, nx)
            yi   = np.linspace(y_min + cell_size / 2, y_max - cell_size / 2, ny)
            xg, yg = np.meshgrid(xi, yi)
            pts_query = np.column_stack([xg.ravel(), yg.ravel()])
            pts_known = np.column_stack([x, y])

            if fill_method == "idw":
                from scipy.spatial import cKDTree
                tree = cKDTree(pts_known)
                k = min(len(pts_known), 16)
                dists, idx = tree.query(pts_query, k=k, workers=-1)
                if k == 1:
                    dists = dists[:, np.newaxis]
                    idx   = idx[:, np.newaxis]
                with np.errstate(divide="ignore"):
                    w = np.where(dists == 0, np.inf, dists ** -2.0)
                w_sum = w.sum(axis=1)
                zq = (w * values[idx]).sum(axis=1) / w_sum
            elif fill_method == "kriging":
                from pykrige.ok import OrdinaryKriging
                # Subsample for variogram fitting; kriging solve is O(n^3)
                n = len(x)
                if n > 2000:
                    rng = np.random.default_rng(seed=0)
                    sel = rng.choice(n, 2000, replace=False)
                    ok  = OrdinaryKriging(
                        x[sel], y[sel], values[sel],
                        variogram_model="spherical",
                        verbose=False, enable_plotting=False,
                    )
                else:
                    ok = OrdinaryKriging(
                        x, y, values,
                        variogram_model="spherical",
                        verbose=False, enable_plotting=False,
                    )
                # execute on full grid — pykrige returns a masked array
                zg, _ = ok.execute('grid', xi, yi)
                # rasterio origin is top-left; flip so north is up
                grid = np.asarray(zg).astype(np.float32)[::-1, :]
                grid[~np.isfinite(grid)] = np.nan
            else:
                from scipy.interpolate import RBFInterpolator
                rbf = RBFInterpolator(
                    pts_known, values,
                    kernel="thin_plate_spline",
                    neighbors=min(50, len(pts_known)),
                )
                zq = rbf(pts_query)

            if fill_method in ("idw", "rbf"):
                # rasterio origin is top-left; flip so north is up
                grid = zq.reshape(ny, nx).astype(np.float32)[::-1, :]
        else:
            # Snap track points to nearest pixel (no fill)
            xi_idx = np.clip(((x - x_min) / cell_size).astype(int), 0, nx - 1)
            yi_idx = np.clip(ny - 1 - ((y - y_min) / cell_size).astype(int), 0, ny - 1)
            for i in range(len(x)):
                grid[yi_idx[i], xi_idx[i]] = values[i]

        utm_crs   = CRS.from_epsg(epsg)
        transform = from_origin(x_min, y_max, cell_size, cell_size)

        if crs_mode == "wgs84":
            import os
            import tempfile
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            wgs84 = CRS.from_epsg(4326)

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with rasterio.open(
                    tmp_path, "w", driver="GTiff",
                    height=ny, width=nx, count=1, dtype="float32",
                    crs=utm_crs, transform=transform, nodata=float("nan"),
                ) as dst:
                    dst.write(grid, 1)

                with rasterio.open(tmp_path) as src:
                    t_wgs, w_wgs, h_wgs = calculate_default_transform(
                        src.crs, wgs84, src.width, src.height, *src.bounds
                    )
                    with rasterio.open(
                        output_path, "w", driver="GTiff",
                        height=h_wgs, width=w_wgs, count=1, dtype="float32",
                        crs=wgs84, transform=t_wgs, nodata=float("nan"),
                    ) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=t_wgs,
                            dst_crs=wgs84,
                            resampling=Resampling.bilinear,
                        )
            finally:
                os.remove(tmp_path)
        else:
            with rasterio.open(
                output_path, "w", driver="GTiff",
                height=ny, width=nx, count=1, dtype="float32",
                crs=utm_crs, transform=transform, nodata=float("nan"),
            ) as dst:
                dst.write(grid, 1)
