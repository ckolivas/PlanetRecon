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


def _put_event(q, event: JobEvent, cancel_event=None) -> bool:
    # Optional UI updates must never stall processing or cancellation. The
    # terminal event carries the final image even when previews were dropped.
    if event.kind in ("preview", "progress", "snapshot", "source"):
        try:
            q.put_nowait(event)
        except queue.Full:
            return False
        return True
    while cancel_event is None or not cancel_event.is_set():
        try:
            q.put(event, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def result_payload(result):
    return {**result.metadata(), "image": result.image, "coverage": result.coverage,
            "validity": result.validity, "layer_coverage": result.layer_coverage}


def result_from_payload(payload):
    from planetrecon.result import ReconstructionResult

    return ReconstructionResult(**{k: v for k, v in payload.items()
                                   if k in ReconstructionResult.__dataclass_fields__})


def _watch_parent() -> None:
    """Stop owned work after a hard parent crash, on spawn-supported platforms."""
    import threading
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    def watch():
        while True:
            if not parent.is_alive():
                # No live UI can receive events. Do not wait for queue feeders.
                os._exit(1)
            time.sleep(0.25)
    threading.Thread(target=watch, name="planetrecon-parent-watch", daemon=True).start()


def _worker_main(job_id, source_path, config_dict, event_q, cancel_event, *args):
    _watch_parent()
    from planetrecon.memory import cpu_memory_limit
    try:
        with cpu_memory_limit(config_dict.get('max_ram_bytes')):
            _worker_run(job_id, source_path, config_dict, event_q, cancel_event, *args)
    except Exception as exc:
        # A memory failure may prevent the normal handler from allocating an
        # event. The process ceiling has now been restored for terminal reporting.
        _put_event(event_q, JobEvent(job_id, 2**63-1, 'error', {
            'message': str(exc) or 'CPU process memory limit exceeded'}), cancel_event)


def _worker_run(
    job_id: str,
    source_path: str,
    config_dict: dict,
    event_q,
    cancel_event,
    checkpoint_dir: str | None,
    snapshot_request=None,
    inspect_only: bool = False,
    resume_from: str | None = None,
    state_checkpoint: str | None = None,
) -> None:
    apply_thread_limits(config_dict.get("threads"))
    # Spawn imports this module before entering the worker. Keep numerical
    # imports here so the requested thread limit precedes BLAS initialisation.
    from planetrecon.io.source import open_source
    from planetrecon.pipeline.baseline import stack_source

    seq = 0
    source = None

    def emit(kind: str, payload: dict) -> bool:
        nonlocal seq
        seq += 1
        return _put_event(event_q, JobEvent(job_id, seq, kind, payload), cancel_event)

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
        emit("progress", {"stage": "scan", "fraction": 0.0})
        if (snapshot_request is not None or inspect_only) and not cancel_event.is_set():
            import numpy as np
            from planetrecon.detector import is_bayer, nearest_debayer_preview

            raw = source.read_raw(0)
            color = source.color_mode()
            if color == "BGR":
                raw = raw[..., ::-1]
            step = max(1, int(np.ceil(max(raw.shape[:2]) / 512)))
            bayer = is_bayer(color)
            preview = nearest_debayer_preview(raw, color, stride=step) if bayer else np.array(raw[::step, ::step], copy=True)
            payload = {"source_metadata": source.metadata().as_dict(),
                       "input_image": preview,
                       "input_max": float(np.max(raw, where=np.isfinite(raw), initial=0)),
                       "input_stride": step, "input_view": "nearest-neighbour Bayer RGB" if bayer else color}
            emit("completed" if inspect_only else "source", payload)
            if inspect_only:
                return
        if config.geometry_mode != "none":
            emit("progress", {"stage": "pose estimation", "fraction": None, "backend": "cpu",
                              "device_report": {"reason": "Geometry processing uses CPU float64"}})

        def on_event(result: ReconstructionResult, info: dict) -> None:
            if result.stage == 'preprocessing':
                emit('progress', {'stage': 'preprocessing: quality and planet size',
                    'fraction': info.get('n_processed', 0) / max(info.get('n_total', 1), 1),
                    'n_used': 0, 'backend': 'cpu'})
                return
            if snapshot_request is not None and snapshot_request.is_set() and result.n_used:
                snapshot_request.clear()
                if not emit("snapshot", result_payload(result)):
                    snapshot_request.set()
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
                    "validity": preview.validity,
                    "spatial_stride": preview.spatial_stride,
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
                    "device_report": result.provenance.get("device_report", {}),
                    "warnings": result.warnings,
                },
            )
            if checkpoint_dir:
                _write_checkpoint(checkpoint_dir, job_id, config, result)

        result = stack_source(
            source,
            config,
            on_event=on_event,
            should_cancel=cancel_event.is_set,
            resume_from=resume_from,
            state_checkpoint=state_checkpoint,
        )
        source.close()
        source = None
        if cancel_event.is_set():
            emit("cancelled", {"n_used": result.n_used})
            return
        emit("completed", result_payload(result))

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
    try:
        np.savez_compressed(tmp, image=result.image, coverage=result.coverage,
                            validity=result.validity, metadata=json.dumps(meta, sort_keys=True),
                            **{f"layer_coverage__{name}": value for name, value in result.layer_coverage.items()})
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


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
    snapshot_request: Any = None

    def poll(self, timeout: float = 0.0) -> list[JobEvent]:
        out = []
        # Cancellation may abandon the queue feeder halfway through a large
        # snapshot. Never enter recv() on that pipe once cancellation is set.
        if self.cancel_event.is_set():
            if not self.process.is_alive() and self.state not in ("completed", "failed", "cancelled"):
                self.state = "cancelled"
                self.last_seq += 1
                return [JobEvent(self.job_id, self.last_seq, "cancelled")]
            return out
        deadline = time.time() + timeout
        def accept(event):
            if event.job_id != self.job_id or event.seq <= self.last_seq:
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
    inspect_only: bool = False,
    resume_from: str | Path | None = None,
    state_checkpoint: str | Path | None = None,
) -> JobHandle:
    ctx = multiprocessing.get_context("spawn")
    job_id = job_id or f"job-{os.getpid()}-{int(time.time() * 1000)}"
    if inspect_only and (resume_from is not None or state_checkpoint is not None):
        raise ValueError('input inspection cannot use accumulator checkpoints')
    if state_checkpoint is not None and checkpoint_dir is not None:
        state, snapshot = Path(state_checkpoint), Path(checkpoint_dir) / f'{job_id}.npz'
        if state.resolve() == snapshot.resolve() or (state.exists() and snapshot.exists() and state.samefile(snapshot)):
            raise ValueError('resumable state and result checkpoint must have different paths')
    from planetrecon.event_transport import FileEventQueue

    event_q = FileEventQueue.create(ctx, maxsize=max(2, int(queue_size)))
    cancel_event = ctx.Event()
    snapshot_request = ctx.Event()
    proc = ctx.Process(
        target=_worker_main,
        args=(
            job_id,
            str(source_path),
            config.to_dict(),
            event_q,
            cancel_event,
            None if checkpoint_dir is None else str(checkpoint_dir),
            snapshot_request,
            inspect_only,
            None if resume_from is None else str(resume_from),
            None if state_checkpoint is None else str(state_checkpoint),
        ),
        name=f"planetrecon-job-{job_id}",
        daemon=True,
    )
    handle = JobHandle(job_id, proc, event_q, cancel_event, state="queued", snapshot_request=snapshot_request)
    try:
        proc.start()
    except BaseException:
        event_q.close()
        raise
    handle.state = "running"
    return handle
