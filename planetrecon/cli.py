"""Command-line entry points for Prompts 1 and 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from planetrecon import constants as C
from planetrecon.simulate import generate_one
from planetrecon.validate import run_development_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planetrecon",
        description="PlanetRecon Gate-1 simulator and known-transfer estimators (R9)",
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

    args = parser.parse_args(argv)
    if args.cmd == "generate":
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
    parser.error("unknown command")
    return 2
