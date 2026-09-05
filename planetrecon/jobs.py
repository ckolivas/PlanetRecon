"""Owned reconstruction jobs: process lifecycle, cancel, checkpoints, events."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import signal
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from planetrecon import constants as C
from planetrecon.io.source import open_source
from planetrecon.pipeline.baseline import stack_source
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.result import ReconstructionResult
from planetrecon.runtime import apply_thread_limits


JobState = str  # queued, running, completed, failed, cancelled


@dataclass
class JobEvent:
    job_id: str
    seq: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


def _put_event(q, event: JobEvent, coalesce_preview: bool = True) -> None:
    if event.kind == "preview" and coalesce_preview:
        try:
            q.put_nowait(event)
            return
        except queue.Full:
            try:
                dumped = q.get_nowait()
                if dumped.kind not in ("preview", "progress"):
                    q.put_nowait(dumped)
                    q.put(event)
                    return
            except queue.Empty:
                pass
            try:
                q.put_nowait(event)
            except queue.Full:
                return
            return
    q.put(event)


def _worker_main(
    job_id: str,
    source_path: str,
    config_dict: dict,
    event_q,
    cancel_event,
    checkpoint_dir: str | None,
) -> None:
    apply_thread_limits(config_dict.get("threads"))
    seq = 0

    def emit(kind: str, payload: dict) -> None:
        nonlocal seq
        seq += 1
        _put_event(event_q, JobEvent(job_id, seq, kind, payload))

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
                    "channel_order": result.channel_order,
                    "warnings": result.warnings,
                    "fraction": float(info.get("n_used", 0))
                    / max(float(info.get("n_total", 1)), 1.0),
                },
            )
            emit(
                "progress",
                {
                    "stage": result.stage,
                    "fraction": float(info.get("n_used", 0))
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
                "image": result.image,
                "coverage": result.coverage,
            },
        )
    except Exception as exc:
        emit(
            "error",
            {"message": str(exc), "traceback": traceback.format_exc()},
        )


def _write_checkpoint(
    directory: str,
    job_id: str,
    config: ReconstructionConfig,
    result: ReconstructionResult,
) -> None:
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
    }
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, image=result.image, coverage=result.coverage)
    tmp.replace(dest)
    (path / f"{job_id}.json").write_text(json.dumps(meta, sort_keys=True) + "\n")


def load_checkpoint(path: Path, config: ReconstructionConfig) -> dict:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".json").read_text())
    stored = ReconstructionConfig.from_dict(meta["config"])
    if stored.to_json() != config.to_json():
        raise ValueError("checkpoint config does not match the running job")
    if meta.get("operator") != config.baseline_operator_version:
        raise ValueError("checkpoint operator version is incompatible")
    return {"image": data["image"], "coverage": data["coverage"], "meta": meta}


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
        while True:
            remaining = deadline - time.time()
            if remaining < 0 and timeout > 0:
                break
            try:
                event = self.events.get(timeout=max(remaining, 0.0) if timeout else 0.0)
            except queue.Empty:
                break
            if event.seq <= self.last_seq and event.kind == "preview":
                continue
            self.last_seq = max(self.last_seq, event.seq)
            out.append(event)
            if event.kind in ("completed", "failed", "cancelled", "error"):
                if event.kind == "completed":
                    self.state = "completed"
                elif event.kind == "cancelled":
                    self.state = "cancelled"
                else:
                    self.state = "failed"
        if self.state == "queued" and self.process.is_alive():
            self.state = "running"
        if self.state in ("queued", "running") and not self.process.is_alive():
            if self.process.exitcode not in (0, None) and self.state != "cancelled":
                self.state = "failed"
        return out

    def cancel(self, grace_s: float = 2.0) -> None:
        self.cancel_event.set()
        self.process.join(timeout=grace_s)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)
        self.state = "cancelled"

    def close(self) -> None:
        if self.process.is_alive():
            self.cancel()
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
