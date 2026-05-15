# Video + Sensor Processing Tool

Desktop GUI app for building a processing package from video, navigation, and sensor data.

## Current capabilities

- scans a directory of video clips and parses start times from filenames with robust fallback logic
- reads actual video durations from file metadata
- imports a navigation CSV with timestamp / lat / lon / optional altitude mapping
- imports one or more sensor CSV files with multiple channels per file
- lets the user choose timestamp and value columns per CSV, plus display names and units
- shows a shared timeline of video, navigation, and sensor coverage
- exports a configuration `.json`
- runs an end-to-end processing pipeline from the GUI:
  - extracts frames for each selected interval
  - builds a `master.csv` with aligned frame, nav, and sensor data
  - generates GeoTIFF rasters for each sensor channel
  - optionally annotates extracted frames

## Expected deliverables per segment

Each processed interval is written as:

```text
output/segment_001/
├── frames/
├── master.csv
├── sensors/
│   ├── temperature.tif
│   └── ...
├── frames_annotated/         # optional
```

## Robust filename parsing

The scanner first tries the user-specified `strftime` format. If that fails, it automatically tries common datetime patterns such as:

- `20260315_120000`
- `20260315-120000`
- `2026-03-15_12-00-00`
- `2026-03-15T12:00:00`
- `20260315120000`

If none match, it falls back to the file modified time and records that source in the project summary.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
