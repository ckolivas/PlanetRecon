"""Owned reconstruction jobs: process lifecycle, cancel, checkpoints, events."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from planetrecon import constants as C
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.runtime import apply_thread_limits

if TYPE_CHECKING:
    from planetrecon.result import ReconstructionResult


JobState = str  # queued, running, completed, failed, cancelled


@dataclass
class JobEvent:
    job_id: str
    seq: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


def _put_event(q, event: JobEvent, cancel_event=None) -> None:
    # Optional UI updates must never stall processing or cancellation. The
    # terminal event carries the final image even when previews were dropped.
    if event.kind in ("preview", "progress"):
        try:
            q.put_nowait(event)
        except queue.Full:
            pass
        return
    while cancel_event is None or not cancel_event.is_set():
        try:
            q.put(event, timeout=0.1)
            return
        except queue.Full:
            continue


def _worker_main(
    job_id: str,
    source_path: str,
    config_dict: dict,
    event_q,
    cancel_event,
    checkpoint_dir: str | None,
) -> None:
    apply_thread_limits(config_dict.get("threads"))
    # Spawn imports this module before entering the worker. Keep numerical
    # imports here so the requested thread limit precedes BLAS initialisation.
    from planetrecon.io.source import open_source
    from planetrecon.pipeline.baseline import stack_source

    seq = 0
    source = None

    def emit(kind: str, payload: dict) -> None:
        nonlocal seq
        seq += 1
        _put_event(event_q, JobEvent(job_id, seq, kind, payload), cancel_event)

    try:
        config = ReconstructionConfig.from_dict(config_dict)
        source = open_source(
            source_path,
            endian_convention=config.endian_convention,
            endian_override=config.endian_override,
            bayer_override=config.bayer_override,
            recover_complete_frames=config.recover_complete_frames,
            crop=config.crop,
        )
        emit("progress", {"stage": "scan", "fraction": 0.0, "backend": config.device})

        def on_event(result: ReconstructionResult, info: dict) -> None:
            preview = result.copy_preview()
            emit(
                "preview",
                {
                    "stage": result.stage,
                    "n_used": result.n_used,
                    "n_rejected": result.n_rejected,
                    "backend": result.backend,
                    "incomplete": result.incomplete,
                    "image": preview.image,
                    "coverage": preview.coverage,
                    "layer_coverage": preview.layer_coverage,
                    "channel_order": result.channel_order,
                    "reference_epoch": result.reference_epoch,
                    "warnings": result.warnings,
                    "fraction": float(info.get("n_processed", 0))
                    / max(float(info.get("n_total", 1)), 1.0),
                },
            )
            emit(
                "progress",
                {
                    "stage": result.stage,
                    "fraction": float(info.get("n_processed", 0))
                    / max(float(info.get("n_total", 1)), 1.0),
                    "n_used": result.n_used,
                    "backend": result.backend,
                },
            )
            if checkpoint_dir:
                _write_checkpoint(checkpoint_dir, job_id, config, result)

        result = stack_source(
            source,
            config,
            on_event=on_event,
            should_cancel=cancel_event.is_set,
        )
        source.close()
        source = None
        if cancel_event.is_set():
            emit("cancelled", {"n_used": result.n_used})
            return
        emit(
            "completed",
            {
                "n_used": result.n_used,
                "n_rejected": result.n_rejected,
                "backend": result.backend,
                "warnings": result.warnings,
                "channel_order": result.channel_order,
                "reference_epoch": result.reference_epoch,
                "image": result.image,
                "coverage": result.coverage,
                "layer_coverage": result.layer_coverage,
                "validity": result.validity,
                "units": result.units,
                "incomplete": result.incomplete,
                "provenance": result.provenance,
            },
        )
    except Exception as exc:
        emit(
            "error",
            {"message": str(exc), "traceback": traceback.format_exc()},
        )
    finally:
        if source is not None:
            source.close()
        if cancel_event.is_set():
            # A caller cancelling a job does not need its queued previews.
            event_q.cancel_join_thread()


def _write_checkpoint(
    directory: str,
    job_id: str,
    config: ReconstructionConfig,
    result: ReconstructionResult,
) -> None:
    import numpy as np

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / f"{job_id}.npz"
    meta = {
        "job_id": job_id,
        "config": config.to_dict(),
        "operator": config.baseline_operator_version,
        "schema": C.JOB_SCHEMA,
        "schema_version": C.JOB_SCHEMA_VERSION,
        "n_used": result.n_used,
        "n_rejected": result.n_rejected,
        "units": result.units,
        "channel_order": result.channel_order,
        "reference_epoch": result.reference_epoch,
        "incomplete": result.incomplete,
        "provenance": result.provenance,
        "layer_names": sorted(result.layer_coverage),
        "result": result.metadata(),
    }
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, image=result.image, coverage=result.coverage,
                        validity=result.validity, metadata=json.dumps(meta, sort_keys=True),
                        **{f"layer_coverage__{name}": value for name, value in result.layer_coverage.items()})
    tmp.replace(dest)


def load_checkpoint(path: Path, config: ReconstructionConfig) -> dict:
    import numpy as np

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data:
            raise ValueError("legacy non-atomic checkpoint is unsupported")
        meta = json.loads(str(data["metadata"]))
        arrays = {key: data[key] for key in ("image", "coverage", "validity")}
        arrays["layer_coverage"] = {name: data[f"layer_coverage__{name}"]
                                    for name in meta.get("layer_names", [])}
    if meta.get("schema") != C.JOB_SCHEMA or meta.get("schema_version") != C.JOB_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    stored = ReconstructionConfig.from_dict(meta["config"])
    if stored.to_json() != config.to_json():
        raise ValueError("checkpoint config does not match the running job")
    if meta.get("operator") != config.baseline_operator_version:
        raise ValueError("checkpoint operator version is incompatible")
    return {**arrays, "meta": meta}


@dataclass
class JobHandle:
    job_id: str
    process: multiprocessing.Process
    events: multiprocessing.Queue
    cancel_event: multiprocessing.Event
    state: JobState = "queued"
    started: float = field(default_factory=time.time)
    last_seq: int = 0

    def poll(self, timeout: float = 0.0) -> list[JobEvent]:
        out = []
        deadline = time.time() + timeout
        def accept(event):
            if event.seq <= self.last_seq and event.kind == "preview":
                return
            self.last_seq = max(self.last_seq, event.seq)
            out.append(event)
            terminal = {"completed": "completed", "cancelled": "cancelled",
                        "failed": "failed", "error": "failed"}
            if event.kind in terminal:
                self.state = terminal[event.kind]

        while True:
            remaining = deadline - time.time()
            if remaining < 0 and timeout > 0:
                break
            try:
                event = self.events.get(timeout=max(remaining, 0.0) if timeout else 0.0)
            except queue.Empty:
                break
            accept(event)
        if self.state == "queued" and self.process.is_alive():
            self.state = "running"
        if self.state in ("queued", "running") and not self.process.is_alive():
            # Recheck after observing exit: the child may have flushed its last
            # event between the earlier queue read and is_alive().
            while True:
                try:
                    accept(self.events.get_nowait())
                except queue.Empty:
                    break
            if self.state in ("queued", "running") and self.process.exitcode is not None:
                if self.cancel_event.is_set():
                    accept(JobEvent(self.job_id, self.last_seq + 1, "cancelled"))
                else:
                    accept(JobEvent(self.job_id, self.last_seq + 1, "error", {
                        "message": f"worker exited without a result (code {self.process.exitcode})"
                    }))
        return out

    def cancel(self, grace_s: float = 2.0) -> None:
        self.cancel_event.set()
        deadline = time.monotonic() + grace_s
        while self.process.is_alive() and time.monotonic() < deadline:
            self.poll()
            self.process.join(timeout=0.02)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)
        self.state = "cancelled"

    def close(self) -> None:
        if self.state in ("completed", "failed"):
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.cancel()
        else:
            self.process.join()
        try:
            self.events.close()
        except Exception:
            pass


def start_stack_job(
    source_path: str | Path,
    config: ReconstructionConfig,
    *,
    job_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    queue_size: int = 8,
) -> JobHandle:
    ctx = multiprocessing.get_context("spawn")
    job_id = job_id or f"job-{os.getpid()}-{int(time.time() * 1000)}"
    event_q = ctx.Queue(maxsize=max(2, int(queue_size)))
    cancel_event = ctx.Event()
    proc = ctx.Process(
        target=_worker_main,
        args=(
            job_id,
            str(source_path),
            config.to_dict(),
            event_q,
            cancel_event,
            None if checkpoint_dir is None else str(checkpoint_dir),
        ),
        name=f"planetrecon-job-{job_id}",
        daemon=True,
    )
    handle = JobHandle(job_id, proc, event_q, cancel_event, state="queued")
    proc.start()
    handle.state = "running"
    return handle
