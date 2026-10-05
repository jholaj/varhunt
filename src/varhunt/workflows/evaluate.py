"""How well the detector works, injected signals in quiet stars and catalogued variables."""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from varhunt.config import Resources, Run, Settings
from varhunt.evaluation.benchmark import MAGNITUDES, MATCH_ARCSEC, TYPES, Outcome, evaluate, report
from varhunt.evaluation.injection import Trial, completeness_table, plan, run_trial
from varhunt.output.plots import completeness_figure
from varhunt.output.report import QUIET, read_results
from varhunt.sources.catalogs import catalogued_sample
from varhunt.sources.mast import download, local_path, newest_light_curve

log = logging.getLogger(__name__)

QUERY_THREADS = 8


def inject(run: Run, resources: Resources, trials: int, seed: int) -> list[Trial]:
    quiet = [r for r in read_results(run.stars_csv) if r.status == QUIET]
    if not quiet:
        return []
    paths = [local_path(resources.cache_dir, r.file) for r in quiet]
    injections = plan(paths, trials, run.settings, seed)
    with ProcessPoolExecutor(resources.workers) as pool:
        outcome = list(pool.map(partial(run_trial, settings=run.settings), injections, chunksize=4))
    run.completeness_md.write_text(
        f"# Injection and recovery, {len(outcome)} trials on {len(quiet)} quiet stars\n\n"
        "Share of injected sinusoids found at the injected frequency and passing the orbit "
        "check (trials per cell in brackets).\n\n" + completeness_table(outcome) + "\n"
    )
    completeness_figure(outcome, run.root / "completeness.png")
    return outcome


def benchmark(
    settings: Settings, resources: Resources, per_type: int, seed: int, path: Path
) -> list[Outcome]:
    variables = [
        variable
        for kind, (periods, _) in TYPES.items()
        for variable in catalogued_sample(kind, periods, MAGNITUDES, per_type, seed)
    ]
    log.info("%d catalogued variables, looking up their light curves", len(variables))
    with ThreadPoolExecutor(QUERY_THREADS) as pool:
        products = list(
            pool.map(lambda v: newest_light_curve(v.ra, v.dec, MATCH_ARCSEC), variables)
        )
    paths = download([p for p in products if p is not None], resources.cache_dir, resources.workers)
    pairs = [
        (variable, paths[product])
        for variable, product in zip(variables, products, strict=True)
        if product in paths
    ]
    with ProcessPoolExecutor(resources.workers) as pool:
        outcomes = list(pool.map(partial(evaluate, settings=settings), *zip(*pairs, strict=True)))
    path.write_text(
        f"# Recall on {len(outcomes)} catalogued variables with QLP data\n\n"
        f"Random VSX stars of {MAGNITUDES[0]} to {MAGNITUDES[1]} mag, seed {seed}.\n\n"
        + report(outcomes)
        + "\n"
    )
    return outcomes
