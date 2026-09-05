"""Command-line entry points for Prompts 1–2, Q2, and W00 evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from planetrecon.runtime import apply_thread_limits

from planetrecon import constants as C


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planetrecon",
        description="PlanetRecon Gate-1 simulator, known-transfer estimators, and Q2 MFBD (R9)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "CPU BLAS/FFT thread cap (default: PLANETRECON_THREADS or 8). "
            "Applied before importing numerical libraries. GPU devices are not used."
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

    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    apply_thread_limits(args.threads)
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
    parser.error("unknown command")
    return 2
