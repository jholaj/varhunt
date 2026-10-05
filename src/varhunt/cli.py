"""Command line entry point."""

import argparse
import logging
import os
from pathlib import Path

from varhunt.config import Field, Resources, Run, Settings, default_cache_dir
from varhunt.sources.mast import available_sectors
from varhunt.workflows.confirm import confirm
from varhunt.workflows.evaluate import benchmark, inject
from varhunt.workflows.search import search
from varhunt.workflows.survey import sure, survey
from varhunt.workflows.vet import vet
from varhunt.workflows.vsx import prepare

log = logging.getLogger("varhunt")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    resources = Resources(cache_dir=args.cache_dir, workers=args.workers)
    return args.handler(args, resources)


def _sectors(args: argparse.Namespace, resources: Resources) -> int:
    sectors = available_sectors(Field(args.ra, args.dec, args.radius))
    print(", ".join(map(str, sectors)) if sectors else "no QLP light curves here")
    return 0


def _search(args: argparse.Namespace, resources: Resources) -> int:
    field = Field(args.ra, args.dec, args.radius)
    sector = args.sector
    if sector is None:
        sectors = available_sectors(field)
        if not sectors:
            log.error("no QLP light curves around RA %s, Dec %s", args.ra, args.dec)
            return 1
        sector = sectors[-1]
    overrides = {
        name: getattr(args, name)
        for name in ("min_snr", "min_period", "max_period")
        if getattr(args, name) is not None
    }
    root = args.out or Path("runs") / f"s{sector:04d}_{args.ra:.3f}{args.dec:+.3f}"
    run = Run(root=root, field=field, sector=sector, settings=Settings(**overrides))
    search(run, resources)
    print(f"summary: {run.summary_md}")
    return 0


def _vet(args: argparse.Namespace, resources: Resources) -> int:
    run = Run.load(args.run)
    checked = vet(run, resources, set(args.tic))
    for result in checked:
        print(f"TIC {result.tic}: {result.vet_verdict} ({result.vet_offset_px:.2f} px)")
    print(f"summary: {run.summary_md}")
    return 0


def _confirm(args: argparse.Namespace, resources: Resources) -> int:
    run = Run.load(args.run)
    for result in confirm(run, resources):
        print(f"TIC {result.tic}: sure" if result.sure else f"TIC {result.tic}: not sure yet")
    print(f"summary: {run.summary_md}")
    return 0


def _survey(args: argparse.Namespace, resources: Resources) -> int:
    fields = [
        Field(float(ra), float(dec), float(radius))
        for ra, dec, radius in (line.split(",") for line in args.fields.read_text().split())
    ]
    done = survey(fields, Settings(), resources, args.root)
    sure = sum(r.sure for _, results in done for r in results)
    print(f"{len(done)} fields, {sure} sure candidates, report: {args.root / 'survey.md'}")
    return 0


def _sure(args: argparse.Namespace, resources: Resources) -> int:
    found = sure(args.roots, args.out)
    print(f"{len(found)} sure candidates: {args.out}")
    return 0


def _vsx(args: argparse.Namespace, resources: Resources) -> int:
    try:
        draft = prepare(Run.load(args.run), resources, args.tic, args.out, args.discoverer)
    except LookupError as error:
        log.error("%s", error)
        return 1
    print(f"draft: {draft}")
    return 0


def _inject(args: argparse.Namespace, resources: Resources) -> int:
    run = Run.load(args.run)
    trials = inject(run, resources, args.trials, args.seed)
    if not trials:
        log.error("no quiet stars in %s to inject into", run.root)
        return 1
    print(f"completeness: {run.completeness_md}")
    return 0


def _benchmark(args: argparse.Namespace, resources: Resources) -> int:
    outcomes = benchmark(Settings(), resources, args.per_type, args.seed, args.out)
    exact = sum(o.exact for o in outcomes)
    print(f"{exact} of {len(outcomes)} periods within 1 %, details: {args.out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    common.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    common.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(
        prog="varhunt", description="Hunt for uncatalogued variable stars in TESS light curves."
    )
    commands = parser.add_subparsers(required=True)

    sectors = commands.add_parser("sectors", parents=[common], help="list QLP sectors of a field")
    _add_field(sectors)
    sectors.set_defaults(handler=_sectors)

    hunt = commands.add_parser("search", parents=[common], help="search a field for variables")
    _add_field(hunt)
    hunt.add_argument("--sector", type=int, help="TESS sector, the latest one by default")
    hunt.add_argument("--out", type=Path, help="run directory")
    hunt.add_argument("--min-snr", type=float, help="amplitude S/N threshold")
    hunt.add_argument("--min-period", type=float, help="shortest period searched (d)")
    hunt.add_argument("--max-period", type=float, help="longest period searched (d)")
    hunt.set_defaults(handler=_search)

    check = commands.add_parser("vet", parents=[common], help="pixel check of candidates")
    check.add_argument("run", type=Path, help="run directory")
    check.add_argument("--tic", type=int, nargs="*", default=[], help="only these TIC IDs")
    check.set_defaults(handler=_vet)

    evidence = commands.add_parser(
        "confirm", parents=[common], help="other sectors, catalogues and Gaia for candidates"
    )
    evidence.add_argument("run", type=Path, help="run directory")
    evidence.set_defaults(handler=_confirm)

    many = commands.add_parser(
        "survey", parents=[common], help="search, vet and confirm many fields"
    )
    many.add_argument("fields", type=Path, help="file with one ra,dec,radius line per field")
    many.add_argument("--root", type=Path, default=Path("runs/survey"))
    many.set_defaults(handler=_survey)

    gather = commands.add_parser("sure", parents=[common], help="collect the sure candidates")
    gather.add_argument("roots", type=Path, nargs="+", help="run or survey directories")
    gather.add_argument("--out", type=Path, default=Path("runs/sure.md"))
    gather.set_defaults(handler=_sure)

    submit = commands.add_parser("vsx", parents=[common], help="prepare a VSX submission")
    submit.add_argument("run", type=Path, help="run directory of the candidate")
    submit.add_argument("tic", type=int, help="TIC ID of the candidate")
    submit.add_argument("--discoverer", default="", help="name for the VSX Discoverer field")
    submit.add_argument("--out", type=Path, default=Path("runs/vsx"))
    submit.set_defaults(handler=_vsx)

    trial = commands.add_parser("inject", parents=[common], help="measure detection completeness")
    trial.add_argument("run", type=Path, help="run directory")
    trial.add_argument("--trials", type=int, default=500)
    trial.add_argument("--seed", type=int, default=0)
    trial.set_defaults(handler=_inject)

    recall = commands.add_parser(
        "benchmark", parents=[common], help="check the detector on catalogued variables"
    )
    recall.add_argument("--per-type", type=int, default=12, help="stars per VSX type")
    recall.add_argument("--seed", type=int, default=1)
    recall.add_argument("--out", type=Path, default=Path("benchmark.md"))
    recall.set_defaults(handler=_benchmark)
    return parser


def _add_field(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("ra", type=float, help="field centre right ascension (deg)")
    parser.add_argument("dec", type=float, help="field centre declination (deg)")
    parser.add_argument("--radius", type=float, default=0.2, help="field radius (deg)")
