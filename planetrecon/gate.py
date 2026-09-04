"""Prompt 2 CLI orchestration: freeze λ, evaluate families, write tables."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from planetrecon import constants as C
from planetrecon.evaluate import (
    REG,
    aggregate_family,
    evaluate_file,
    family_paths,
    format_classification,
    freeze_regularisation,
    resource_rows,
    write_hash_file,
    write_resource_csv,
)
from planetrecon.hdf5io import schema_errors
from planetrecon.rank import ranking_config_hash
from planetrecon.simulate import generate_one


def _generate_one(args):
    seed, dr0, out_dir = args
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=2):
            return str(generate_one(seed, dr0, Path(out_dir)))
    except ImportError:
        return str(generate_one(seed, dr0, Path(out_dir)))


def ensure_truth_files(
    out_dir: Path,
    seeds: tuple[int, ...],
    generate: bool = True,
    workers: int = 1,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = family_paths(out_dir, seeds)
    missing = []
    for path in wanted:
        if not path.exists() or schema_errors(path):
            missing.append(path)
    if missing and not generate:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"missing Gate-eligible truth files: {names}")
    if missing:
        jobs = []
        for path in missing:
            # gate1_Dr0-{int}_seed-{int:05d}.h5
            seed = int(path.stem.split("seed-")[1])
            dr0 = float(path.stem.split("Dr0-")[1].split("_")[0])
            jobs.append((seed, dr0, str(out_dir)))
        print(f"Generating {len(jobs)} truth files...", flush=True)
        if workers <= 1:
            for job in jobs:
                _generate_one(job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_generate_one, job) for job in jobs]
                for fut in as_completed(futs):
                    print(f"  wrote {fut.result()}", flush=True)
    return wanted


def run_family(
    out_dir: Path,
    seeds: tuple[int, ...],
    family_name: str,
    generate: bool = True,
    workers: int = 1,
    eval_workers: int = 1,
) -> dict:
    out_dir = Path(out_dir)
    prompt2_dir = out_dir / "prompt2"
    prompt2_dir.mkdir(parents=True, exist_ok=True)
    hash_path = write_hash_file(prompt2_dir)
    print(f"ranking_hash {hash_path.read_text().strip()}", flush=True)
    paths = ensure_truth_files(out_dir, seeds, generate=generate, workers=workers)
    results = []
    if eval_workers <= 1:
        for path in paths:
            print(f"Evaluating {path.name}", flush=True)
            results.append(evaluate_file(path, out_dir=prompt2_dir, reg=REG))
    else:
        with ProcessPoolExecutor(max_workers=eval_workers) as pool:
            futs = {
                pool.submit(evaluate_file, path, prompt2_dir, ("feature", "bland"), REG): path
                for path in paths
            }
            for fut in as_completed(futs):
                res = fut.result()
                print(f"  done seed={res['seed']} D/r0={res['dr0']}", flush=True)
                results.append(res)
    results.sort(key=lambda r: (float(r["dr0"]), int(r["seed"])))
    tables = {}
    for crop in ("feature", "bland"):
        agg = aggregate_family(results, crop=crop)
        text = format_classification(agg, crop)
        dest = prompt2_dir / f"classification_{family_name}_{crop}.txt"
        dest.write_text(text)
        (prompt2_dir / f"classification_{family_name}_{crop}.json").write_text(
            json.dumps(agg, indent=2, sort_keys=True) + "\n"
        )
        print(text)
        tables[crop] = agg
        for estimator in ("E1", "E2a", "A1o"):
            rows = resource_rows(results, crop, estimator)
            write_resource_csv(
                rows,
                prompt2_dir / f"resource_{family_name}_{crop}_{estimator}.csv",
            )
    summary = {
        "family": family_name,
        "ranking_hash": ranking_config_hash(),
        "n_files": len(results),
        "tables": tables,
        "regularisation": {
            "E1_LAMBDA_REL": C.E1_LAMBDA_REL,
            "E2A_LAMBDA_REL": C.E2A_LAMBDA_REL,
        },
    }
    (prompt2_dir / f"summary_{family_name}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def run_freeze(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    prompt2_dir = out_dir / "prompt2"
    prompt2_dir.mkdir(parents=True, exist_ok=True)
    write_hash_file(prompt2_dir)
    paths = family_paths(out_dir, C.DEV_SEEDS)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "development truth files missing; run `planetrecon validate-dev` first: "
            + ", ".join(p.name for p in missing)
        )
    result = freeze_regularisation(paths)
    dest = prompt2_dir / "frozen_regularisation.json"
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"chosen λ_rel = {result['chosen_lambda_rel']}")
    print(f"grid medians: {result['candidates']}")
    print(f"wrote {dest}")
    return result
