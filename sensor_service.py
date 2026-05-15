from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models import NavigationConfig, SensorChannel, SensorFileConfig, TimeValueSourceConfig


class SensorService:
    TIMESTAMP_CANDIDATES = [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "unix_timestamp",
        "unix_time",
        "epoch",
    ]

    @staticmethod
    def read_preview(csv_path: str | Path, nrows: int = 20) -> pd.DataFrame:
        return pd.read_csv(csv_path, nrows=nrows)

    @staticmethod
    def read_columns(csv_path: str | Path) -> list[str]:
        df = pd.read_csv(csv_path, nrows=0)
        return [str(column) for column in df.columns]

    @staticmethod
    def normalize_timestamps(series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(dtype="float64")

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(3, int(0.7 * len(series.dropna()))):
            cleaned = numeric.astype("float64")
            finite = cleaned[np.isfinite(cleaned)]
            if finite.empty:
                return cleaned
            median = float(np.nanmedian(finite))
            if median > 1e16:     # nanoseconds  (~1.74e18 for 2026)
                return cleaned / 1e9
            if median > 1e13:     # microseconds (~1.74e15 for 2026)
                return cleaned / 1e6
            if median > 1e10:     # milliseconds (~1.74e12 for 2026)
                return cleaned / 1e3
            return cleaned        # seconds      (~1.74e9  for 2026)

        dt = pd.to_datetime(series, errors="coerce", utc=False)
        unit = np.datetime_data(dt.dtype)[0]  # "ns", "us", "ms", or "s"
        scale = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(unit, 1e9)
        ints = dt.astype("int64")
        ints = ints.where(dt.notna(), np.nan)
        return ints / scale

    @staticmethod
    def build_config(
        csv_path: str | Path,
        timestamp_column: str,
        channels: list[SensorChannel],
    ) -> SensorFileConfig:
        df = pd.read_csv(csv_path)
        timestamps = SensorService.normalize_timestamps(df[timestamp_column]).dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        start_dt = pd.to_datetime(timestamps.min(), unit="s").to_pydatetime()
        end_dt = pd.to_datetime(timestamps.max(), unit="s").to_pydatetime()
        return SensorFileConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            channels=channels,
            start_time=start_dt,
            end_time=end_dt,
        )

    @staticmethod
    def build_time_value_source_config(
        csv_path: str | Path,
        timestamp_column: str,
        value_column: str,
    ) -> TimeValueSourceConfig:
        df = pd.read_csv(csv_path)
        timestamps = SensorService.normalize_timestamps(df[timestamp_column]).dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        values = pd.to_numeric(df[value_column], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"No valid numeric values found in column '{value_column}' for {csv_path}")

        start_dt = pd.to_datetime(timestamps.min(), unit="s").to_pydatetime()
        end_dt = pd.to_datetime(timestamps.max(), unit="s").to_pydatetime()
        return TimeValueSourceConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            value_column=value_column,
            start_time=start_dt,
            end_time=end_dt,
        )

    @staticmethod
    def build_navigation_config(
        latitude_source: TimeValueSourceConfig,
        longitude_source: TimeValueSourceConfig,
        altitude_source: TimeValueSourceConfig | None = None,
    ) -> NavigationConfig:
        return NavigationConfig(
            latitude_source=latitude_source,
            longitude_source=longitude_source,
            altitude_source=altitude_source,
        )

    @staticmethod
    def load_sensor_dataframe(config: SensorFileConfig) -> pd.DataFrame:
        df = pd.read_csv(config.csv_path)
        cols = [config.timestamp_column] + [channel.source_column for channel in config.channels]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing}")

        result = df[cols].copy()
        result["unix_time"] = SensorService.normalize_timestamps(result[config.timestamp_column])
        result = result.dropna(subset=["unix_time"]).sort_values("unix_time")
        result = result.drop_duplicates(subset=["unix_time"])
        return result

    @staticmethod
    def load_time_value_dataframe(config: TimeValueSourceConfig) -> pd.DataFrame:
        df = pd.read_csv(config.csv_path)
        cols = [config.timestamp_column, config.value_column]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing}")

        result = df[cols].copy()
        result["unix_time"] = SensorService.normalize_timestamps(result[config.timestamp_column])
        result["value"] = pd.to_numeric(result[config.value_column], errors="coerce")
        result = result.dropna(subset=["unix_time", "value"]).sort_values("unix_time")
        result = result.drop_duplicates(subset=["unix_time"])
        return result[["unix_time", "value"]]

    @staticmethod
    def interpolate_series(target_unix_time: pd.Series, source_unix_time: pd.Series, source_values: pd.Series) -> np.ndarray:
        sx = pd.to_numeric(source_unix_time, errors="coerce").to_numpy(dtype=float)
        sy = pd.to_numeric(source_values, errors="coerce").to_numpy(dtype=float)
        tx = pd.to_numeric(target_unix_time, errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(sx) & np.isfinite(sy)
        sx = sx[mask]
        sy = sy[mask]
        if len(sx) == 0:
            return np.full_like(tx, np.nan, dtype=float)
        if len(sx) == 1:
            return np.full_like(tx, sy[0], dtype=float)

        order = np.argsort(sx)
        sx = sx[order]
        sy = sy[order]
        return np.interp(tx, sx, sy, left=sy[0], right=sy[-1])
