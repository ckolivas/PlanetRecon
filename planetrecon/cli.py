"""Command-line entry points for Prompts 1–2, Q2, and W00 evidence."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from planetrecon.runtime import apply_thread_limits

from planetrecon import constants as C


def main(argv: list[str] | None = None) -> int:
    if argv is None and getattr(sys, "frozen", False) and len(sys.argv) == 1:
        argv = ["gui"]
    parser = argparse.ArgumentParser(
        prog="planetrecon",
        description="PlanetRecon Gate-1 simulator, known-transfer estimators, and Q2 MFBD (R9)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "CPU BLAS/FFT thread cap (default: PLANETRECON_THREADS or 32). "
            "GPU is used only after a live operator probe succeeds."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate one seed/regime HDF5 truth file")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--dr0", type=float, required=True, help="D/r0 (8 or 4)")
    g.add_argument("--out", type=Path, default=Path("out"))
    g.add_argument("--n-frames", type=int, default=C.N_FRAMES)
    g.add_argument("--j", type=int, default=C.DEFAULT_J)
    g.add_argument("--padding", type=int, default=C.DEFAULT_PADDING_DET_PX)

    v = sub.add_parser(
        "validate-dev",
        help="run the Prompt 1 development validation suite",
    )
    v.add_argument("--out", type=Path, default=Path("out"))
    v.add_argument(
        "--no-generate",
        action="store_true",
        help="do not generate missing development HDF5 files",
    )

    fr = sub.add_parser(
        "freeze-reg",
        help="choose E1/E2a λ_rel on development seeds and write the scan",
    )
    fr.add_argument("--out", type=Path, default=Path("out"))

    ev = sub.add_parser(
        "evaluate",
        help="Prompt 2 reconstructions, G1/G2/G3, and classification tables",
    )
    ev.add_argument("--out", type=Path, default=Path("out"))
    ev.add_argument(
        "--family",
        choices=("dev", "eval", "ext"),
        default="eval",
        help="seed family: development 1001-1003, evaluation 2001-2012, or extension 2013-2024",
    )
    ev.add_argument(
        "--no-generate",
        action="store_true",
        help="do not generate missing truth files",
    )
    ev.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process workers for truth-file generation",
    )
    ev.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="process workers for reconstructions",
    )

    rec = sub.add_parser("reconstruct", help="run Prompt 2 estimators on one truth file")
    rec.add_argument("--path", type=Path, required=True)
    rec.add_argument("--out", type=Path, default=Path("out/prompt2"))

    fp = sub.add_parser(
        "freeze-prior",
        help="choose E2b TV μ on development seeds and lock E2*",
    )
    fp.add_argument("--out", type=Path, default=Path("out"))

    q2 = sub.add_parser(
        "q2",
        help="Q2 all-frame D/D-tail reconstructions and closure tables",
    )
    q2.add_argument("--out", type=Path, default=Path("out"))
    q2.add_argument(
        "--family",
        choices=("dev", "eval", "ext"),
        default="dev",
        help="seed family; development also runs held-out prediction by default",
    )
    q2.add_argument(
        "--dr0",
        type=float,
        nargs="+",
        default=list(C.Q2_SCOPE_DR0),
        help="seeing regimes (default: 4, the strong-G1 moderate-seeing scope)",
    )
    q2.add_argument("--no-generate", action="store_true")
    q2.add_argument("--workers", type=int, default=1)
    q2.add_argument("--eval-workers", type=int, default=1)
    q2.add_argument(
        "--frame-workers",
        type=int,
        default=C.Q2_FRAME_WORKERS,
        help="threads for per-frame PSF/phase updates inside one reconstruction",
    )
    hold = q2.add_mutually_exclusive_group()
    hold.add_argument(
        "--holdout",
        action="store_true",
        help="fit a 90/10 held-out diagnostic in addition to all-frame D",
    )
    hold.add_argument(
        "--no-holdout",
        action="store_true",
        help="skip the held-out diagnostic (default for eval/ext)",
    )

    evd = sub.add_parser(
        "evidence",
        help="write W00/W01 evidence manifests without regenerating R9 tables",
    )
    evd.add_argument("--results", type=Path, default=Path("results"))

    st = sub.add_parser("stack", help="W07 baseline stack of a SER, AVI or observed HDF5 crop")
    st.add_argument("--path", type=Path, required=True)
    st.add_argument("--out", type=Path, default=Path("out/stack"))
    st.add_argument("--export", choices=("png16", "tiff16", "tiff32"), default=None,
                    help="also save a scientific image in the output directory")
    st.add_argument("--checkpoint", type=Path, default=None,
                    help="atomically update a full-resolution NPZ snapshot after each batch")
    st.add_argument("--resume", type=Path, help="resume baseline/geometry accumulator state with matching configuration")
    st.add_argument("--state-checkpoint", type=Path, help="atomically save resumable accumulator state after each batch")
    st.add_argument("--device", choices=("cpu", "auto", "gpu"), default="auto")
    st.add_argument("--batch", type=int, default=32)
    st.add_argument("--cuda-memory-mib", type=int, help="CUDA tensor allocator cap in MiB; excludes driver/library memory")
    st.add_argument("--crop", choices=("feature", "bland"), default="feature")
    st.add_argument("--bayer", "--color", default=None, help="raw colour override: mono or a Bayer pattern such as RGGB")
    for name in ("bias", "dark", "flat"):
        st.add_argument(f"--{name}", default=None, help=f"detector-shape {name} NPY table")
    st.add_argument("--gain", type=float, default=None, help="electrons per ADU")
    st.add_argument("--read-noise", type=float, default=None, help="read noise in electrons (metadata)")
    st.add_argument("--saturate", type=float, default=None, help="saturation threshold in raw ADU")
    st.add_argument(
        "--geometry",
        choices=("none", "field", "surface", "combined", "saturn"),
        default="none",
        help="none=translation; field/surface/combined/saturn use declared geometry",
    )
    st.add_argument("--field-rate-deg-s", type=float, default=None)
    st.add_argument("--surface-rate-deg-s", type=float, default=None)
    st.add_argument("--field-angle0-deg", type=float, default=0.0)
    st.add_argument("--flattening", type=float, default=0.0)
    st.add_argument("--radius", type=float, default=None, help="equatorial radius in pixels")
    st.add_argument("--center-x", type=float, default=None, help="disc centre x in detector pixel-centre coordinates")
    st.add_argument("--center-y", type=float, default=None, help="disc centre y in detector pixel-centre coordinates")
    st.add_argument("--pole-pa-deg", type=float, default=0.0)
    st.add_argument("--sub-obs-lat-deg", type=float, default=None)
    st.add_argument("--sub-obs-lon-deg", type=float, default=0.0)
    st.add_argument("--exposure", type=float, default=0.0, help="integration time in seconds")
    st.add_argument("--cadence", type=float, default=None, help="seconds between frame starts")
    st.add_argument("--reference-epoch", type=float, default=0.0)
    st.add_argument("--ring-inner", type=float, default=None, help="Saturn ring inner radius in pixels")
    st.add_argument("--ring-outer", type=float, default=None, help="Saturn ring outer radius in pixels")
    st.add_argument("--ring-transmission", type=float, default=0.35)
    st.add_argument("--sun-lon-deg", type=float, default=None)
    st.add_argument("--sun-lat-deg", type=float, default=None)
    st.add_argument("--moon-x", type=float, default=None)
    st.add_argument("--moon-y", type=float, default=None)
    st.add_argument("--moon-radius", type=float, default=None)
    st.add_argument("--moon-vx", type=float, default=0.0)
    st.add_argument("--moon-vy", type=float, default=0.0)

    ex = sub.add_parser("export", help="export a saved result or full-resolution checkpoint")
    ex.add_argument("--path", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--encoding", choices=("png16", "tiff16", "tiff32"), default="tiff32")
    for command in (st, ex):
        command.add_argument("--black", type=float, default=None, help="fixed integer black level in result units")
        command.add_argument("--white", type=float, default=None, help="fixed integer white level in result units")
        command.add_argument("--display-gamma", type=float, default=None,
                             help="label as display-rendered and apply power 1/gamma (integer only)")
        command.add_argument("--overwrite", action="store_true", help="allow replacement of an existing exported image")

    prb = sub.add_parser("probe-device", help="probe CPU/GPU backends and print the selection")
    prb.add_argument("--device", choices=("cpu", "auto", "gpu"), default="auto")

    gui = sub.add_parser("gui", help="Qt6 shell with progressive baseline reconstruction")
    gui.add_argument("--path", type=Path, default=None)
    smoke = sub.add_parser("gui-smoke", help="bounded Qt/owned-worker/scientific-save packaging check")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    applied_threads = apply_thread_limits(args.threads)
    export_config = None
    if args.cmd in ("stack", "export"):
        from planetrecon.export import ExportConfig

        encoding = args.encoding if args.cmd == "export" else args.export
        if encoding:
            try:
                export_config = ExportConfig(encoding, args.black, args.white, args.display_gamma)
            except ValueError as exc:
                parser.error(str(exc))
        elif any(v is not None for v in (args.black, args.white, args.display_gamma)) or args.overwrite:
            parser.error("export mapping and overwrite options require --export")
        if args.cmd == "stack" and args.checkpoint:
            if args.checkpoint.suffix.lower() != ".npz":
                parser.error("--checkpoint must name an NPZ snapshot")
            if (args.checkpoint.resolve() == args.path.resolve() or
                    (args.checkpoint.exists() and args.checkpoint.samefile(args.path))):
                parser.error("checkpoint cannot replace the input capture")
    if args.cmd == "stack" and (args.resume or args.state_checkpoint):
        if args.state_checkpoint:
            for other in (args.out / "stack.npz", args.checkpoint):
                if other is not None and (args.state_checkpoint.resolve() == other.resolve() or
                        (args.state_checkpoint.exists() and other.exists() and args.state_checkpoint.samefile(other))):
                    parser.error("resumable state and scientific result checkpoints must have different paths")
    if args.cmd == "export":
        from planetrecon.export import export_result
        from planetrecon.result import load_snapshot

        try:
            report = export_result(load_snapshot(args.path), args.out, export_config, overwrite=args.overwrite)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"wrote {report.path} metadata={report.sidecar} "
              f"incomplete={report.metadata['result']['incomplete']} counts={report.metadata['counts']}")
        return 0
    if args.cmd == "generate":
        from planetrecon.simulate import generate_one

        generate_one(
            args.seed,
            args.dr0,
            args.out,
            n_frames=args.n_frames,
            exposure_samples_j=args.j,
            padding_detector_px=args.padding,
        )
        return 0
    if args.cmd == "validate-dev":
        from planetrecon.validate import run_development_suite

        return run_development_suite(args.out, generate=not args.no_generate)
    if args.cmd == "freeze-reg":
        from planetrecon.gate import run_freeze

        run_freeze(args.out)
        return 0
    if args.cmd == "evaluate":
        from planetrecon.gate import run_family

        seeds = {
            "dev": C.DEV_SEEDS,
            "eval": C.EVAL_SEEDS,
            "ext": C.EXT_SEEDS,
        }[args.family]
        run_family(
            args.out,
            seeds,
            family_name=args.family,
            generate=not args.no_generate,
            workers=args.workers,
            eval_workers=args.eval_workers,
        )
        return 0
    if args.cmd == "reconstruct":
        from planetrecon.evaluate import evaluate_file

        result = evaluate_file(args.path, out_dir=args.out)
        print(result.get("metrics_path", "ok"))
        return 0
    if args.cmd == "freeze-prior":
        from planetrecon.q2 import run_freeze_prior

        run_freeze_prior(args.out)
        return 0
    if args.cmd == "q2":
        from planetrecon.q2 import run_q2_family

        seeds = {
            "dev": C.DEV_SEEDS,
            "eval": C.EVAL_SEEDS,
            "ext": C.EXT_SEEDS,
        }[args.family]
        if args.no_holdout:
            holdout = False
        elif args.holdout:
            holdout = True
        else:
            holdout = args.family == "dev"
        run_q2_family(
            args.out,
            seeds,
            family_name=args.family,
            dr0s=tuple(float(v) for v in args.dr0),
            generate=not args.no_generate,
            workers=args.workers,
            eval_workers=args.eval_workers,
            holdout=holdout,
            frame_workers=args.frame_workers,
        )
        return 0
    if args.cmd == "evidence":
        from planetrecon.evidence import write_evidence_manifests

        written = write_evidence_manifests(args.results)
        for path in written:
            print(path)
        return 0
    if args.cmd == "probe-device":
        from planetrecon.backends import select_backend

        backend, report = select_backend(args.device, threads=applied_threads)
        print(f"requested={report.requested} selected={report.selected} name={report.name}")
        print(f"fallback={report.fallback} reason={report.reason}")
        for warning in report.warnings:
            print(f"warning: {warning}")
        print(f"backend={backend.name} precision={backend.precision}")
        return 0 if not (args.device == "gpu" and report.fallback) else 2
    if args.cmd == "stack":
        from planetrecon.io import open_source
        from planetrecon.pipeline.baseline import stack_source
        from planetrecon.reconstruction import ReconstructionConfig
        from planetrecon.result import save_snapshot

        args.out.mkdir(parents=True, exist_ok=True)
        export_path = args.out / ("stack.png" if args.export == "png16" else "stack.tif")
        if export_config and export_path.exists() and not args.overwrite:
            parser.error(f"{export_path} exists; use --overwrite to replace it")
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        def _rad(deg):
            return None if deg is None else float(deg) * math.pi / 180.0

        cfg = ReconstructionConfig(
            device=args.device,
            threads=applied_threads,
            batch_frames=args.batch,
            max_vram_bytes=None if args.cuda_memory_mib is None else args.cuda_memory_mib * 1024**2,
            crop=args.crop,
            bayer_override=args.bayer,
            bias_path=args.bias, dark_path=args.dark, flat_path=args.flat,
            gain_e_per_adu=args.gain, read_noise_e=args.read_noise, saturate_adu=args.saturate,
            geometry_mode=args.geometry,
            field_rate_rad_s=_rad(args.field_rate_deg_s),
            surface_rate_rad_s=_rad(args.surface_rate_deg_s),
            field_angle0_rad=_rad(args.field_angle0_deg) or 0.0,
            flattening=args.flattening,
            equatorial_radius_px=args.radius,
            field_center_x=args.center_x,
            field_center_y=args.center_y,
            pole_pa_rad=_rad(args.pole_pa_deg) or 0.0,
            sub_obs_lat_rad=_rad(args.sub_obs_lat_deg),
            sub_obs_lon0_rad=_rad(args.sub_obs_lon_deg) or 0.0,
            exposure_s=args.exposure,
            cadence_s=args.cadence,
            reference_epoch_s=args.reference_epoch,
            ring_inner_radius_px=args.ring_inner,
            ring_outer_radius_px=args.ring_outer,
            ring_transmission=args.ring_transmission,
            sun_lon_rad=_rad(args.sun_lon_deg),
            sun_lat_rad=_rad(args.sun_lat_deg),
            moon_x=args.moon_x,
            moon_y=args.moon_y,
            moon_radius_px=args.moon_radius,
            moon_vx_px_s=args.moon_vx,
            moon_vy_px_s=args.moon_vy,
        )
        with open_source(
            args.path,
            crop=args.crop,
            bayer_override=args.bayer,
        ) as source:
            result = stack_source(source, cfg, on_event=(
                (lambda snapshot, info: save_snapshot(args.checkpoint, snapshot)) if args.checkpoint else None),
                resume_from=args.resume, state_checkpoint=args.state_checkpoint)
        npz = args.out / "stack.npz"
        save_snapshot(npz, result)
        if export_config:
            from planetrecon.export import export_result

            try:
                report = export_result(result, export_path, export_config, overwrite=args.overwrite)
            except (OSError, ValueError) as exc:
                parser.error(f"export failed: {exc}; full result retained at {npz}")
            print(f"wrote {report.path} metadata={report.sidecar} counts={report.metadata['counts']}")
        print(
            f"wrote {npz} backend={result.backend} n_used={result.n_used} "
            f"rejected={result.n_rejected} incomplete={result.incomplete}"
        )
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    if args.cmd == "gui":
        from planetrecon.gui.app import main as gui_main

        argv = [] if args.path is None else [str(args.path)]
        return gui_main(argv)
    if args.cmd == "gui-smoke":
        from planetrecon.gui.smoke import run_smoke

        return run_smoke(args.out, args.device)
    parser.error("unknown command")
    return 2
