"""Many fields in a row, and the sure candidates gathered from any number of runs."""

import logging
from pathlib import Path

from varhunt.checks.pixels import cached_cutouts
from varhunt.config import RUN_FILE, Field, Resources, Run, Settings
from varhunt.output.report import CANDIDATE, Result, read_results, write_sure, write_survey
from varhunt.sources.mast import available_sectors, best_sector, local_path
from varhunt.workflows.confirm import confirm
from varhunt.workflows.search import search
from varhunt.workflows.vet import vet

log = logging.getLogger(__name__)

DONE_FILE = "survey.done"


def survey(
    fields: list[Field], settings: Settings, resources: Resources, root: Path
) -> list[tuple[Run, list[Result]]]:
    """Search field after field, vet and confirm the strong candidates, keep only their data."""
    done: list[tuple[Run, list[Result]]] = []
    for field in fields:
        name = f"{field.ra:.3f}{field.dec:+.3f}"
        if (root / name / DONE_FILE).exists():
            run = Run.load(root / name)
            done.append((run, read_results(run.stars_csv)))
            continue
        sectors = available_sectors(field)
        if not sectors:
            log.info("field %s has no QLP data", name)
            continue
        run = Run(
            root=root / name, field=field, sector=best_sector(field, sectors), settings=settings
        )
        log.info("field %s, sector %d", name, run.sector)
        try:
            strong = {r.tic for r in search(run, resources) if r.could_be_sure}
            if strong:
                vet(run, resources, strong)
                confirm(run, resources)
        except Exception:
            # One field failing on the network must not stop the survey, it is retried next time.
            log.exception("field %s failed", name)
            continue
        results = read_results(run.stars_csv)
        _prune(resources, results)
        (run.root / DONE_FILE).touch()
        done.append((run, results))
        write_survey(root / "survey.md", done)
    write_survey(root / "survey.md", done)
    return done


def sure(roots: list[Path], path: Path) -> list[tuple[Run, Result]]:
    """Every sure candidate of the runs in or below the given directories, strongest first."""
    runs = sorted({file.parent for root in roots for file in _run_files(root)})
    found = [
        (run, result)
        for run in (Run.load(directory) for directory in runs)
        if run.stars_csv.exists()
        for result in read_results(run.stars_csv)
        if result.sure
    ]
    found.sort(key=lambda item: -item[1].strength)
    write_sure(path, found, len(runs))
    return found


def _run_files(root: Path) -> list[Path]:
    return [path for path in (root / RUN_FILE, *root.glob(f"*/{RUN_FILE}")) if path.exists()]


def _prune(resources: Resources, results: list[Result]) -> None:
    for result in results:
        if result.status != CANDIDATE:
            local_path(resources.cache_dir, result.file).unlink(missing_ok=True)
        for path in cached_cutouts(result.ra, result.dec, resources.cache_dir):
            path.unlink(missing_ok=True)
