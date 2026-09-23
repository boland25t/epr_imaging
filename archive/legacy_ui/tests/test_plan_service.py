"""Tests for plan_service — the Task Stack's plan builder.

These run WITHOUT a QApplication, which is the point of extracting the planner
out of MainWindow: the orchestration rules are now directly testable.

Covered:
  * scope resolution for every target kind (full / job / jobs / all_jobs)
  * the dedupe rule that stops "All jobs" fanning out to a saved job twice
  * per-channel fan-out (one task × N channels × M jobs → N*M steps)
  * task ordering is preserved
  * skip reporting for jobs with no intervals and unrunnable targets
  * the workspace-level invariants (build_interp / anomaly_detect never fan out)
  * settings → kwargs mapping, including the fill-label translation
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plan_service
from models import Job, SelectedTimeRange, Task, TaskStack


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_job(job_id: int, name: str = "", n_intervals: int = 1) -> Job:
    intervals = [
        SelectedTimeRange(
            start_time=datetime(2026, 1, 15, 20, i, 0),
            end_time=datetime(2026, 1, 15, 20, i, 30),
        )
        for i in range(n_intervals)
    ]
    return Job(job_id=job_id, name=name or f"Job {job_id}", intervals=intervals)


def make_ctx(pending=None, history=None, channels=("CO2", "CH4"), videos=(), **kw):
    defaults = dict(
        workspace_path="/ws",
        pending_job=pending,
        job_history=list(history or []),
        sensor_files=[],
        videos=list(videos),
        interp_full_path=lambda: "/ws/interp_full.csv",
        outputs_root=lambda: "/ws/outputs",
        filtered_interp_for_job=lambda job: f"/ws/job_{job.job_id}/filtered_interp.csv",
        job_output_dirname=lambda job: f"job_{job.job_id}",
        available_channels=lambda: list(channels),
        build_interp_config=lambda task: object(),      # non-None ⇒ runnable
        build_sampling_config=lambda task, job, scope_id: object(),
        raster_channel_units=lambda ch: "uatm",
    )
    defaults.update(kw)
    return plan_service.PlanContext(**defaults)


def stack_of(*tasks) -> TaskStack:
    s = TaskStack()
    for t in tasks:
        s.add(t)
    return s


def task(task_type, target=None, settings=None, channels=None, task_id=1, depends_on=None):
    return Task(
        task_id=task_id,
        task_type=task_type,
        target=target or {"kind": "full"},
        settings=settings or {},
        channels=channels or [],
        depends_on=depends_on,
    )


# --------------------------------------------------------------------------
# scope resolution
# --------------------------------------------------------------------------
def test_full_target_yields_single_workspace_scope():
    ctx = make_ctx()
    scopes = plan_service.resolve_task_scopes(ctx, task("nav_3d"), [])
    assert len(scopes) == 1
    assert scopes[0].scope_id == "full"
    assert scopes[0].interp_path == "/ws/interp_full.csv"
    assert scopes[0].output_dir == "/ws/outputs"
    assert scopes[0].job is None


def test_job_target_resolves_to_that_job():
    job = make_job(7)
    ctx = make_ctx(pending=job)
    scopes = plan_service.resolve_task_scopes(
        ctx, task("nav_3d", {"kind": "job", "job_id": 7}), []
    )
    assert [s.scope_id for s in scopes] == ["job_7"]
    assert scopes[0].job is job
    assert scopes[0].output_dir == str(Path("/ws/job_7/outputs"))


def test_jobs_target_resolves_each_named_job():
    a, b = make_job(1), make_job(2)
    ctx = make_ctx(pending=a, history=[b])
    scopes = plan_service.resolve_task_scopes(
        ctx, task("nav_3d", {"kind": "jobs", "jobs": [{"job_id": 1}, {"job_id": 2}]}), []
    )
    assert [s.scope_id for s in scopes] == ["job_1", "job_2"]


def test_all_jobs_dedupes_pending_and_its_saved_copy():
    """After Save Job the pending job and its history copy share an id.

    "All jobs" must not fan out to the same job twice.
    """
    pending = make_job(3, "Dive A")
    saved_copy = make_job(3, "Dive A")          # same id, as after _job_save
    other = make_job(4, "Dive B")
    ctx = make_ctx(pending=pending, history=[saved_copy, other])

    assert [jid for jid, _ in plan_service.available_jobs(ctx)] == [3, 4]

    scopes = plan_service.resolve_task_scopes(ctx, task("nav_3d", {"kind": "all_jobs"}), [])
    assert [s.scope_id for s in scopes] == ["job_3", "job_4"]


def test_job_without_intervals_is_skipped_with_a_reason():
    empty = Job(job_id=9, name="Empty", intervals=[])
    ctx = make_ctx(pending=empty)
    skips: list = []
    scopes = plan_service.resolve_task_scopes(
        ctx, task("nav_3d", {"kind": "job", "job_id": 9}), skips
    )
    assert scopes == []
    assert len(skips) == 1 and "no intervals" in skips[0]


def test_build_interp_never_fans_out_per_job():
    """interp_full.csv is workspace-level even when the task targets jobs."""
    a, b = make_job(1), make_job(2)
    ctx = make_ctx(pending=a, history=[b])
    scopes = plan_service.resolve_task_scopes(
        ctx, task("build_interp", {"kind": "all_jobs"}), []
    )
    assert [s.scope_id for s in scopes] == ["full"]


# --------------------------------------------------------------------------
# plan building
# --------------------------------------------------------------------------
def test_per_channel_task_fans_out_over_channels():
    ctx = make_ctx(channels=("CO2", "CH4", "O2"))
    result = plan_service.build_plan(ctx, stack_of(task("sensor_3d")))
    assert len(result) == 3
    assert [s["channel"] for s in result.steps] == ["CO2", "CH4", "O2"]
    assert all(s["method"] == "generate_sensor_3d_ply" for s in result.steps)


def test_fan_out_is_channels_times_jobs():
    a, b = make_job(1), make_job(2)
    ctx = make_ctx(pending=a, history=[b], channels=("CO2", "CH4"))
    result = plan_service.build_plan(
        ctx, stack_of(task("sensor_2d", {"kind": "all_jobs"}))
    )
    assert len(result) == 4                      # 2 channels × 2 jobs
    assert {s["scope_id"] for s in result.steps} == {"job_1", "job_2"}


def test_explicit_task_channels_override_discovery():
    ctx = make_ctx(channels=("CO2", "CH4", "O2"))
    result = plan_service.build_plan(
        ctx, stack_of(task("sensor_3d", channels=["CH4"]))
    )
    assert [s["channel"] for s in result.steps] == ["CH4"]


def test_task_order_is_preserved():
    ctx = make_ctx(channels=("CO2",))
    result = plan_service.build_plan(ctx, stack_of(
        task("build_interp", task_id=1),
        task("nav_3d", task_id=2),
        task("qc_report", task_id=3),
    ))
    assert [s["product_type"] for s in result.steps] == [
        "build_interp", "nav_3d", "qc_report",
    ]


def test_unrunnable_target_is_reported_as_a_skip():
    ctx = make_ctx()                              # no jobs at all
    result = plan_service.build_plan(
        ctx, stack_of(task("nav_3d", {"kind": "job", "job_id": 42}))
    )
    assert result.steps == []
    assert any("no runnable scope" in s for s in result.skips)


def test_job_interp_on_full_scope_is_skipped():
    """job_interp writes one CSV per interval, so it needs a job scope."""
    ctx = make_ctx()
    result = plan_service.build_plan(ctx, stack_of(task("job_interp")))
    assert result.steps == []
    assert any("target a Job" in s for s in result.skips)


def test_job_interp_emits_epoch_intervals():
    job = make_job(5, n_intervals=3)
    ctx = make_ctx(pending=job)
    result = plan_service.build_plan(
        ctx, stack_of(task("job_interp", {"kind": "job", "job_id": 5}))
    )
    assert len(result) == 1
    ivs = result.steps[0]["kwargs"]["intervals"]
    assert len(ivs) == 3
    assert all(isinstance(a, int) and isinstance(b, int) and a < b for a, b in ivs)


def test_build_interp_skipped_when_config_unavailable():
    ctx = make_ctx(build_interp_config=lambda task: None)
    result = plan_service.build_plan(ctx, stack_of(task("build_interp")))
    assert result.steps == []


def test_anomaly_detect_is_workspace_level_even_when_targeting_a_job():
    job = make_job(2)
    ctx = make_ctx(pending=job)
    result = plan_service.build_plan(
        ctx, stack_of(task("anomaly_detect", {"kind": "job", "job_id": 2},
                           {"run_detector": False, "run_catalog": True}))
    )
    assert len(result) == 1
    stepd = result.steps[0]
    assert stepd["scope_id"] == "full"
    assert stepd["kwargs"]["run_detector"] is False
    assert stepd["kwargs"]["run_catalog"] is True
    assert stepd["kwargs"]["interp_path"] == "/ws/interp_full.csv"


def test_photogrammetry_carries_depends_on_and_real_job_id():
    job = make_job(6)
    ctx = make_ctx(pending=job)
    result = plan_service.build_plan(ctx, stack_of(
        task("photogrammetry", {"kind": "job", "job_id": 6},
             {"engine": "COLMAP", "chunk_size": 250}, task_id=9, depends_on=3)
    ))
    stepd = result.steps[0]
    assert stepd["engine"] == "colmap"
    assert stepd["depends_on_task_id"] == 3
    assert stepd["kwargs"]["job_id"] == 6          # not 0
    assert stepd["kwargs"]["chunk_size"] == 250


def test_photogrammetry_full_dataset_uses_job_id_zero():
    ctx = make_ctx()
    result = plan_service.build_plan(ctx, stack_of(task("photogrammetry")))
    assert result.steps[0]["kwargs"]["job_id"] == 0


@pytest.mark.parametrize("label,expected", [
    ("IDW fill", "idw"),
    ("Kriging fill", "kriging"),
    ("RBF fill", "rbf"),
    ("Trackline only (no fill)", "none"),
    ("nonsense", "idw"),                            # unknown → default
])
def test_fill_label_translation(label, expected):
    ctx = make_ctx(channels=("CO2",))
    result = plan_service.build_plan(
        ctx, stack_of(task("sensor_3d", settings={"fill": label}))
    )
    assert result.steps[0]["kwargs"]["fill_method"] == expected


def test_sensor_slices_uses_full_scale_source_for_job_scopes():
    """Per-job slices borrow the full-dataset colour range for comparability."""
    job = make_job(1)
    ctx = make_ctx(pending=job, channels=("CO2",))
    result = plan_service.build_plan(
        ctx, stack_of(task("sensor_slices", {"kind": "job", "job_id": 1}))
    )
    kw = result.steps[0]["kwargs"]
    assert "_scale_source_glob" in kw
    assert "outputs/sensor_3d/CO2" in kw["_scale_source_glob"].replace("\\", "/")


def test_sensor_slices_manual_range_overrides_scale_source():
    job = make_job(1)
    ctx = make_ctx(pending=job, channels=("CO2",))
    result = plan_service.build_plan(ctx, stack_of(
        task("sensor_slices", {"kind": "job", "job_id": 1},
             {"manual_range": True, "vmin": 2.0, "vmax": 9.0})
    ))
    kw = result.steps[0]["kwargs"]
    assert kw["vmin"] == 2.0 and kw["vmax"] == 9.0
    assert "_scale_source_glob" not in kw


def test_netcdf_pulls_units_from_context():
    ctx = make_ctx(channels=("CO2",), raster_channel_units=lambda ch: f"units-of-{ch}")
    result = plan_service.build_plan(ctx, stack_of(task("sensor_netcdf")))
    assert result.steps[0]["kwargs"]["units"] == "units-of-CO2"


def test_empty_stack_yields_empty_plan():
    result = plan_service.build_plan(make_ctx(), TaskStack())
    assert not result and result.steps == [] and result.skips == []
