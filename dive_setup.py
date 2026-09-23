#!/usr/bin/env python3
"""Create a sensor-only dive workspace bundle and build its interp_full.csv.

Generates <root>/<dive>_down.eprproj/workspace.json from the EPR_2026_DATA
file conventions (Renav J17XX_renav_*.csv, DPA J2-17XX_DPA.csv, wide sensor
CSV "EPR csvs/dive17xx.csv") and runs the app pipeline's build_full_interp
step.  No video required — suitable for gas-anomaly-only processing.

Sensor channels carry time_delay_s (response-lag correction, default 0.0;
see models.SensorChannel).

Qt-free.  setup_dive("J1755") -> workspace path.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path("/home/troyboland/epr_imaging/EPR_2026_DATA")
ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
TEMPLATE_WS = ROOT / "J1754_down.eprproj" / "workspace.json"

CHANNELS = [
    ("concentration_uatm_co2", "CO2 Concentration", "uatm"),
    ("concentration_uatm_ch4", "CH4 Concentration", "uatm"),
    ("O2Concentration_uM_", "O2 Concentration", "uM"),
    ("SALINITY_PSU_", "Salinity", "PSU"),
    ("TEMPERATURE_c_", "Temperature", "degC"),
]

# Per-channel sensor response delays in seconds (readings lag the water
# encounter).  All zero until characterised; central place to set them.
GAS_TIME_DELAYS_S = {
    "CO2 Concentration": 0.0,
    "CH4 Concentration": 0.0,
    "O2 Concentration": 0.0,
    "Salinity": 0.0,
    "Temperature": 0.0,
}


def _dive_files(dive: str):
    num = dive[1:]
    renav = sorted(glob.glob(str(DATA / "Renav" / f"{dive}_renav_*.csv")))
    dpa = DATA / "DPA" / f"J2-{num}_DPA.csv"
    sens = DATA / "Sensors" / "EPR csvs" / f"dive{num}.csv"
    if not renav or not dpa.is_file() or not sens.is_file():
        raise FileNotFoundError(f"{dive}: renav={bool(renav)} dpa={dpa.is_file()} "
                                f"sensors={sens.is_file()}")
    return renav[0], dpa, sens


def _renav_range(renav):
    head = pd.read_csv(renav, header=None, nrows=1)
    tail = pd.read_csv(renav, header=None).iloc[-1]
    def iso(date, clock):
        m, d, y = str(date).split("/")
        return f"20{y}-{int(m):02d}-{int(d):02d}T{clock}"
    return iso(head.iloc[0, 0], head.iloc[0, 1]), iso(tail[0], tail[1])


def _dpa_range(dpa):
    d = pd.read_csv(dpa, usecols=["DATE", "TIME"])
    def iso(row):
        return row["DATE"].replace("/", "-") + "T" + row["TIME"]
    return iso(d.iloc[0]), iso(d.iloc[-1])


def setup_dive(dive: str, log=print) -> str:
    renav, dpa, sens = _dive_files(dive)
    ws_dir = ROOT / f"{dive}_down.eprproj"
    (ws_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (ws_dir / "survey").mkdir(exist_ok=True)

    w = json.loads(TEMPLATE_WS.read_text())
    nav0, nav1 = _renav_range(renav)
    dpa0, dpa1 = _dpa_range(dpa)

    nf = w["navigation_file"]
    for key, src in nf.items():
        if isinstance(src, dict) and "csv_path" in src:
            if "DPA" in src["csv_path"]:
                src["csv_path"] = str(dpa)
                src["start_time"], src["end_time"] = dpa0, dpa1
            else:
                src["csv_path"] = str(renav)
                src["start_time"], src["end_time"] = nav0, nav1
    nf["start_time"], nf["end_time"] = dpa0, dpa1

    sf = w["sensor_files"][0]
    sf["csv_path"] = str(sens)
    sf["start_time"], sf["end_time"] = nav0, nav1
    sf["channels"] = [
        {"source_column": col, "display_name": name, "units": units,
         "use_header_name": False,
         "time_delay_s": GAS_TIME_DELAYS_S.get(name, 0.0)}
        for col, name, units in CHANNELS
    ]

    w["video_directory"] = f"/mnt/f/EPR_VIDEO/{dive}/DOWN"
    w["workspace_path"] = str(ws_dir)
    for k in ("segment_history", "job_history", "threshold_history"):
        w[k] = []
    w["pending_job"] = {"job_id": 1, "name": "", "intervals": [],
                        "status": "pending", "settings_snapshot": {}}
    w["task_stack"] = {"tasks": [], "_next_task_id": 1}
    (ws_dir / "workspace.json").write_text(json.dumps(w, indent=1))
    log(f"{dive}: workspace written ({nav0} -> {nav1})")

    from config_service import ConfigService
    from pipeline_service import PipelineService, PipelineConfig
    data = ConfigService.load_workspace(str(ws_dir / "workspace.json"))
    cfg = PipelineConfig(
        video_directory=Path(data["video_directory"]),
        output_directory=ws_dir / "inputs",
        job_id=0,
        video_filename_time_format=data.get("filename_datetime_format", ""),
        videos=[],
        selected_intervals=[],
        navigation_file=data["navigation_file"],
        sensor_files=data.get("sensor_files") or [],
        selected_steps=["build_full_interp"],
        workspace_directory=str(ws_dir / "inputs"),
    )
    PipelineService().run(cfg)
    n = sum(1 for _ in open(ws_dir / "inputs" / "interp_full.csv")) - 1
    log(f"{dive}: interp_full.csv built ({n:,} rows)")
    return str(ws_dir)


if __name__ == "__main__":
    dives = sys.argv[1:] or ["J1755", "J1758", "J1759", "J1760", "J1761"]
    for dv in dives:
        setup_dive(dv)
    print("DIVE_SETUP_DONE")
