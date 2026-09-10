#!/usr/bin/env python3
"""BIIGLE REST API bridge: push survey frames + seed annotations to biigle.de
(or any BIIGLE instance) and pull annotation reports back.

Endpoints (confirmed against biigle/core routes/api.php + controllers/requests on
github.com/biigle/core master, and the community wrapper/examples at
github.com/biigle/community-resources — create-volume/create-volume.py and
annotations_to_biigle/export_to_biigle.py). All paths are relative to
{base_url}/api/v1/ with HTTP Basic auth (email, API token):

  GET    label-trees                      accessible label trees [{id, name, ...}]
  GET    label-trees/{id}                 tree incl. "labels": [{id, name, color, parent_id}]
  POST   label-trees                      {name, visibility_id (1=public, 2=private), project_id?}
  POST   label-trees/{id}/labels          {name, color (hex, '#' optional), parent_id?} -> [label]
  POST   projects/{id}/volumes            {name, url, media_type ('image'|'video'), files: [...],
                                           handle?} (multipart 'metadata_csv' also accepted here)
  GET    volumes/{id}/files               file ids of a volume
  GET    volumes/{id}/filenames           {"<image_id>": "<filename>", ...}
  POST   images/{id}/annotations          single annotation {shape_id, points, label_id, confidence}
  POST   image-annotations                bulk create, list of {image_id, shape_id, label_id,
                                           confidence, points}, max 100 per request
  POST   volumes/{id}/metadata            multipart 'file' + 'parser' (for image CSVs the parser is
                                           'Biigle\\Services\\MetadataParsing\\ImageCsvParser';
                                           CSV columns: filename, taken_at, lng, lat, gps_altitude,
                                           distance_to_ground, area, yaw)
  POST   volumes/{id}/reports             {type_id, ...options (export_area, newest_label,
                                           separate_label_trees, only_labels, ...)} -> report dict
  GET    reports/{id}                     downloads the finished report file (owner only)

Shape ids (biigle/core database/migrations/2015_02_09_090836_populate_required_rows.php,
2017_09_20 add_ellipse_shape, 2020_11_02 add_whole_frame_annotation_shape):
Point=1, LineString=2, Polygon=3, Circle=4, Rectangle=5, Ellipse=6, WholeFrame=7.
Circle points payload is [cx, cy, r]; all pixel coords in FULL-resolution frame space.

Report type ids are migration seed order (no API endpoint lists them): 1 ImageAnnotations\\Area,
2 ImageAnnotations\\Basic, 3 ImageAnnotations\\Csv, 4 ImageAnnotations\\Extended,
5 ImageAnnotations\\Full, 6 ImageLabels\\Basic, 7 ImageLabels\\Csv, 8 VideoAnnotations\\Csv,
9 ImageAnnotations\\Abundance, 10 VideoLabels\\Csv, then location/iFDO/COCO types.

Volume `url`: either a public https:// URL serving the frame files ("remote volume") or a
storage-disk URL like 'local://frames/J1754' / 's3://bucket/prefix' configured on the instance.

OPEN QUESTIONS (not confirmed, handled defensively):
  * Report type ids 1-10 follow biigle.de's fixed migration order; a self-hosted instance that
    diverged could differ, and there is no report-types listing endpoint to verify against.
  * Report generation is queued server-side; GET reports/{id} 404s until the file exists.
    No status endpoint was found — download_report() polls.
  * GET label-trees returns trees the user can ACCESS (not strictly ones they own), so
    ensure_label_tree() reuses the first accessible tree with a matching name.
  * Whether biigle.de throttles clients (HTTP 429) is undocumented; we self-throttle ~4 req/s.

Credentials: args, else env BIIGLE_EMAIL / BIIGLE_TOKEN (also the community wrapper's
BIIGLE_API_EMAIL / BIIGLE_API_TOKEN), else a .biigle.env KEY=VALUE file next to this module.

Qt-free.  Smoke test without credentials:
  python3 biigle_bridge.py --dry-run push <workspace> --project-id 1 --volume-url https://...
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import requests

SHAPES = {"point": 1, "line": 2, "polygon": 3, "circle": 4, "rectangle": 5,
          "ellipse": 6, "whole-frame": 7}

REPORT_TYPE_IMAGE_ANNOTATION_CSV = 3   # ImageAnnotations\Csv (see docstring caveat)

# Default label set for a hydrothermal vent survey: (name, hex color without '#').
EPR_LABELS = [
    ("bacterial mat", "f5f5dc"),
    ("tubeworm bush", "e8402a"),
    ("clam/mussel bed", "f2c744"),
    ("diffuse flow", "6ec6e6"),
    ("sulfide structure", "8a5a2b"),
    ("basalt (sheet)", "5a5f66"),
    ("basalt (pillow)", "2f3542"),
    ("sediment patch", "c9b28a"),
    ("fauna (other)", "b05fd4"),
    ("candidate — unreviewed", "39d353"),
]

_ENV_FILE = Path(__file__).resolve().parent / ".biigle.env"
_THROTTLE_S = 0.25          # >= this between requests: stay under ~5 req/s
_BULK_LIMIT = 100           # server-enforced cap on POST image-annotations


def _read_env_file(path=_ENV_FILE) -> dict:
    """Parse KEY=VALUE lines (comments/blank lines ignored) from a .biigle.env file."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


class BiigleApi:
    """Thin requests.Session wrapper for the BIIGLE REST API (paths without 'api/v1').

    get/post/put/delete return parsed JSON (dict/list; {} for empty bodies) and raise
    RuntimeError with the response body on non-2xx.  With dry_run=True no request is
    made: each call is logged (payload truncated) and a canned minimal dict with a
    fake id comes back so full flows are exercisable without credentials.
    """

    def __init__(self, email=None, token=None, base_url=None, dry_run=False, log=print):
        env = dict(_read_env_file())
        env.update({k: v for k, v in os.environ.items() if k.startswith("BIIGLE")})
        self.email = email or env.get("BIIGLE_EMAIL") or env.get("BIIGLE_API_EMAIL")
        self.token = token or env.get("BIIGLE_TOKEN") or env.get("BIIGLE_API_TOKEN")
        self.base_url = (base_url or env.get("BIIGLE_BASE_URL") or "https://biigle.de").rstrip("/")
        self.dry_run = dry_run
        self.log = log
        self._fake_id = itertools.count(90001)
        self._last_request = 0.0
        if not dry_run and not (self.email and self.token):
            raise RuntimeError("BIIGLE credentials missing: pass email/token, set BIIGLE_EMAIL/"
                               "BIIGLE_TOKEN, or create %s" % _ENV_FILE)
        self.session = requests.Session()
        self.session.auth = (self.email or "dry-run", self.token or "dry-run")
        self.session.headers["Accept"] = "application/json"

    # -- core ------------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return "%s/api/v1/%s" % (self.base_url, path.lstrip("/"))

    def _canned(self, method: str, path: str) -> dict:
        if method == "GET":                       # empty collections/maps for listings
            return {}
        return {"id": next(self._fake_id), "dry_run": True}

    def _call(self, method: str, path: str, **kwargs):
        if self.dry_run:
            payload = kwargs.get("json") or kwargs.get("data") or {}
            text = json.dumps(payload, default=str)
            if len(text) > 300:
                text = text[:300] + "... (%d bytes)" % len(text)
            files = kwargs.get("files")
            extra = " files=%s" % list(files) if files else ""
            self.log("[dry-run] %-6s %s %s%s" % (method, path, text, extra))
            return self._canned(method, path)
        wait = self._last_request + _THROTTLE_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        r = self.session.request(method, self._url(path), **kwargs)
        if not (200 <= r.status_code < 300):
            body = r.text[:1000]
            raise RuntimeError("BIIGLE %s %s failed: HTTP %d: %s" % (method, path,
                                                                     r.status_code, body))
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    def get(self, path, **kwargs):
        return self._call("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self._call("POST", path, **kwargs)

    def put(self, path, **kwargs):
        return self._call("PUT", path, **kwargs)

    def delete(self, path, **kwargs):
        return self._call("DELETE", path, **kwargs)


# -- label trees ---------------------------------------------------------------------

def ensure_label_tree(api: BiigleApi, name: str, labels) -> dict:
    """Ensure a label tree `name` exists with all of `labels` [(name, color_hex), ...].

    Idempotent: reuses the first accessible tree with that exact name (see docstring
    caveat: "accessible", not strictly "owned") and only creates missing labels.
    Returns {label_name: label_id}.
    """
    trees = api.get("label-trees")
    tree = next((t for t in trees if t.get("name") == name), None) if isinstance(trees, list) \
        else None
    if tree is None:
        tree = api.post("label-trees", json={"name": name, "visibility_id": 2,
                                             "description": "EPR imaging pipeline label tree"})
        existing = {}
    else:
        detail = api.get("label-trees/%d" % tree["id"])
        existing = {l["name"]: l["id"] for l in detail.get("labels", [])}
    out = dict(existing)
    for label_name, color in labels:
        if label_name in out:
            continue
        made = api.post("label-trees/%d/labels" % tree["id"],
                        json={"name": label_name, "color": color.lstrip("#")})
        if isinstance(made, list):              # server returns a list of created labels
            made = made[0] if made else {}
        out[label_name] = made.get("id")
    return out


# -- volumes -------------------------------------------------------------------------

def create_frame_volume(api: BiigleApi, project_id: int, name: str, files, url: str,
                        metadata_csv=None) -> dict:
    """Create an image volume in a project and (optionally) attach frame metadata.

    files: list of frame filenames present at `url`.  url: 'remote' https:// URL
    serving the files, or a storage-disk URL ('local://...', 's3://...') configured
    on the BIIGLE instance.  metadata_csv: optional path to a BIIGLE-format CSV
    (columns filename, taken_at, lng, lat, gps_altitude, distance_to_ground, area,
    yaw) uploaded via POST volumes/{id}/metadata with the ImageCsvParser.
    Returns the volume dict (has 'id').
    """
    volume = api.post("projects/%d/volumes" % int(project_id),
                      json={"name": name, "url": url, "media_type": "image",
                            "files": list(files)})
    if metadata_csv:
        parser = "Biigle\\Services\\MetadataParsing\\ImageCsvParser"
        if api.dry_run:
            api.post("volumes/%s/metadata" % volume["id"], data={"parser": parser},
                     files={"file": str(metadata_csv)})
        else:
            with open(metadata_csv, "rb") as fh:
                api.post("volumes/%s/metadata" % volume["id"], data={"parser": parser},
                         files={"file": (Path(metadata_csv).name, fh, "text/csv")})
    return volume


# -- seed annotations ----------------------------------------------------------------

def push_seed_annotations(api: BiigleApi, volume_id, seeds_csv, label_id,
                          confidence=0.5, log=print) -> int:
    """Push circle seed annotations from <workspace>/survey/biigle/seed_candidates.csv.

    CSV columns: frame_filename, cx_px, cy_px, r_px, white_frac (full-resolution pixel
    coords).  Maps frame_filename -> image id via GET volumes/{id}/filenames, then
    bulk-creates one Circle ([cx, cy, r]) per row via POST image-annotations (max 100
    per request, self-throttled).  Returns the number of annotations pushed.
    """
    filenames = api.get("volumes/%s/filenames" % volume_id)   # {"<id>": "<filename>"}
    by_name = {v: int(k) for k, v in filenames.items()} if isinstance(filenames, dict) else {}
    rows = list(csv.DictReader(open(seeds_csv, newline="")))
    annotations, skipped = [], 0
    for row in rows:
        fname = row["frame_filename"]
        image_id = by_name.get(fname)
        if image_id is None:
            if api.dry_run:                     # no server to ask: fabricate stable ids
                image_id = by_name.setdefault(fname, 80000 + len(by_name))
            else:
                skipped += 1
                continue
        annotations.append({
            "image_id": image_id,
            "shape_id": SHAPES["circle"],
            "label_id": label_id,
            "confidence": float(confidence),
            "points": [float(row["cx_px"]), float(row["cy_px"]), float(row["r_px"])],
        })
    if skipped:
        log("push_seed_annotations: %d rows skipped (filename not in volume)" % skipped)
    for i in range(0, len(annotations), _BULK_LIMIT):
        api.post("image-annotations", json=annotations[i:i + _BULK_LIMIT])
    log("push_seed_annotations: pushed %d circle annotations to volume %s"
        % (len(annotations), volume_id))
    return len(annotations)


# -- reports -------------------------------------------------------------------------

def request_annotation_report(api: BiigleApi, volume_id,
                              type_id=REPORT_TYPE_IMAGE_ANNOTATION_CSV) -> dict:
    """Request an annotation report (default ImageAnnotations\\Csv, type_id=3 — see the
    docstring caveat about type ids).  Generation is queued server-side; the returned
    report dict's 'id' is downloadable via download_report() once ready."""
    return api.post("volumes/%s/reports" % volume_id, json={"type_id": int(type_id)})


def download_report(api: BiigleApi, report_id_or_url, out_path, timeout_s=300,
                    poll_s=10, log=print) -> str:
    """Download a finished report to out_path (GET reports/{id}, or a full URL).

    Best effort: the report is generated by a queued job with no status endpoint we
    could confirm, so a 404 means "not ready yet" and we poll up to timeout_s."""
    url = str(report_id_or_url)
    if not url.startswith("http"):
        url = api._url("reports/%s" % url)
    if api.dry_run:
        log("[dry-run] GET    %s -> %s" % (url, out_path))
        return str(out_path)
    deadline = time.monotonic() + timeout_s
    while True:
        r = api.session.get(url)
        if r.status_code == 404 and time.monotonic() < deadline:
            log("download_report: not ready (404), retrying in %ds" % poll_s)
            time.sleep(poll_s)
            continue
        if not (200 <= r.status_code < 300):
            raise RuntimeError("BIIGLE GET %s failed: HTTP %d: %s"
                               % (url, r.status_code, r.text[:1000]))
        Path(out_path).write_bytes(r.content)
        return str(out_path)


# -- CLI -----------------------------------------------------------------------------

_FAKE_SEEDS = [("frame_000123.jpg", 1912.0, 1044.5, 62.0, 0.41),
               ("frame_000123.jpg", 355.5, 902.0, 38.5, 0.22),
               ("frame_000456.jpg", 2760.0, 388.0, 91.0, 0.63)]


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Push survey data to a BIIGLE instance.")
    ap.add_argument("--dry-run", action="store_true", help="log requests, make none")
    ap.add_argument("--email"), ap.add_argument("--token")
    ap.add_argument("--base-url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="label tree + volume + seed annotations")
    p.add_argument("workspace")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--volume-url", required=True)
    p.add_argument("--volume-name", default=None)
    p.add_argument("--tree-name", default="EPR Hydrothermal Vent Survey")
    p.add_argument("--confidence", type=float, default=0.5)
    a = ap.parse_args(argv)

    ws = Path(a.workspace)
    seeds = ws / "survey" / "biigle" / "seed_candidates_filtered.csv"
    if not seeds.is_file():
        seeds = ws / "survey" / "biigle" / "seed_candidates.csv"
    if not seeds.is_file():
        if not a.dry_run:
            print("missing contract file: %s" % seeds, file=sys.stderr)
            return 2
        seeds = Path(os.getenv("TMPDIR", "/tmp")) / "biigle_fake_seed_candidates.csv"
        with open(seeds, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_filename", "cx_px", "cy_px", "r_px", "white_frac"])
            w.writerows(_FAKE_SEEDS)
        print("no seed_candidates.csv in workspace; synthesized 3 fake rows -> %s" % seeds)

    files = sorted({r["frame_filename"] for r in csv.DictReader(open(seeds, newline=""))})
    api = BiigleApi(email=a.email, token=a.token, base_url=a.base_url, dry_run=a.dry_run)
    label_map = ensure_label_tree(api, a.tree_name, EPR_LABELS)
    print("label tree '%s': %d labels" % (a.tree_name, len(label_map)))
    name = a.volume_name or "%s frames" % ws.stem
    volume = create_frame_volume(api, a.project_id, name, files, a.volume_url)
    print("volume '%s' id=%s (%d files)" % (name, volume.get("id"), len(files)))
    n = push_seed_annotations(api, volume.get("id"), seeds,
                              label_map.get("candidate — unreviewed"),
                              confidence=a.confidence)
    print("done: %d seed annotations" % n)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
