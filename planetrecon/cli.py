"""Command-line entry points for Prompt 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from planetrecon import constants as C
from planetrecon.simulate import generate_one
from planetrecon.validate import run_development_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planetrecon",
        description="PlanetRecon Gate-1 simulator (R9 Prompt 1)",
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
    parser.error("unknown command")
    return 2
