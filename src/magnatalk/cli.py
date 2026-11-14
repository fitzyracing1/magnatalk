"""magnatalk command-line interface.

    magnatalk pick job.yaml      choose a chip+provider for a job spec
    magnatalk list               show the chip catalog
    magnatalk log                show the decision log
    magnatalk validate           validate the chip catalog
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .catalog import CatalogError, load_catalog, validate_catalog_dict
from .decision_log import annotate_decision, append_decision, default_log_path, read_decisions
from .jobspec import JobSpecError, load_job
from .providers import get_adapter
from .router import NoFeasibleChipError, route


def _add_catalog_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--catalog", metavar="PATH", default=None,
                   help="path to a chips.yaml (default: the bundled catalog)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magnatalk",
        description="Pick AI chips on purpose: route each job to the right GPU and provider.",
    )
    parser.add_argument("--version", action="version", version=f"magnatalk {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pick = sub.add_parser("pick", help="choose a chip+provider for a job spec")
    p_pick.add_argument("job", metavar="JOB_FILE", help="job spec (.yaml/.yml/.json)")
    _add_catalog_arg(p_pick)
    p_pick.add_argument("--json", action="store_true", help="emit the decision as JSON")
    p_pick.add_argument("--top", type=int, default=3, metavar="N",
                        help="number of runners-up to show (default 3)")
    p_pick.add_argument("--no-log", action="store_true",
                        help="do not append the decision to the log")
    p_pick.add_argument("--log-file", default=None, metavar="PATH",
                        help="decision log path (default: $MAGNATALK_LOG_FILE or "
                             "./magnatalk_decisions.jsonl)")
    p_pick.add_argument("--launch-plan", action="store_true",
                        help="also print the provider's DRY-RUN launch plan "
                             "(prints only; never launches or bills anything)")

    p_list = sub.add_parser("list", help="show the chip catalog")
    _add_catalog_arg(p_list)
    p_list.add_argument("--json", action="store_true", help="emit the catalog as JSON")

    p_log = sub.add_parser("log", help="show the decision log")
    p_log.add_argument("-n", type=int, default=20, metavar="N",
                       help="show the last N decisions (default 20)")
    p_log.add_argument("--log-file", default=None, metavar="PATH")
    p_log.add_argument("--json", action="store_true", help="emit raw JSONL records")
    p_log.add_argument("--annotate", metavar="DECISION_ID", default=None,
                       help="set the 'actual' field on a logged decision")
    p_log.add_argument("--actual", metavar="JSON", default=None,
                       help='JSON for --annotate, e.g. \'{"cost_usd": 3.10}\'')

    p_val = sub.add_parser("validate", help="validate the chip catalog")
    _add_catalog_arg(p_val)

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.catalog) if args.catalog else None
    from .catalog import default_catalog_path

    p = path or default_catalog_path()
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        print(f"error: cannot read catalog {p}: {e}", file=sys.stderr)
        return 1
    errors = validate_catalog_dict(raw)
    if errors:
        print(f"catalog {p} is INVALID:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    n_chips = len(raw["chips"])
    n_offers = sum(len(c.get("offers", [])) for c in raw["chips"])
    print(f"catalog {p} is valid: {n_chips} chips, {n_offers} offers")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    chips = load_catalog(args.catalog)
    if args.json:
        out = []
        for c in chips:
            out.append({
                "id": c.id, "name": c.name, "vendor": c.vendor,
                "memory_gb": c.memory_gb,
                "memory_bandwidth_gbps": c.memory_bandwidth_gbps,
                "fp16_tflops": c.fp16_tflops, "bf16_tflops": c.bf16_tflops,
                "fp8_tflops": c.fp8_tflops,
                "supported_precisions": list(c.supported_precisions),
                "offers": [
                    {"provider": o.provider, "price_per_hour_usd": o.price_per_hour_usd,
                     "price_date": o.price_date, "source": o.source}
                    for o in c.offers
                ],
            })
        print(json.dumps(out, indent=2))
        return 0

    hdr = f"{'CHIP':<28}{'MEM':>8}{'BW GB/s':>9}{'FP16':>8}{'FP8':>8}  OFFERS ($/hr)"
    print(hdr)
    print("-" * len(hdr))
    for c in chips:
        offers = ", ".join(
            f"{o.provider} "
            + (f"${o.price_per_hour_usd:.2f}" if o.price_per_hour_usd is not None else "?")
            for o in c.offers
        ) or "(none)"
        fp16 = f"{c.fp16_tflops:g}" if c.fp16_tflops else "-"
        fp8 = f"{c.fp8_tflops:g}" if c.fp8_tflops else "-"
        print(f"{c.name:<28}{c.memory_gb:>6g}GB{c.memory_bandwidth_gbps:>9g}"
              f"{fp16:>8}{fp8:>8}  {offers}")
    print("\nPrices are published list prices with date+source in the catalog; "
          "run `magnatalk list --json` for sources.")
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    chips = load_catalog(args.catalog)
    job = load_job(args.job)
    try:
        decision = route(job, chips, top_n=args.top)
    except NoFeasibleChipError as e:
        print(f"no feasible chip: {e}", file=sys.stderr)
        return 2

    record = None
    if not args.no_log:
        record = append_decision(decision, args.log_file)

    if args.json:
        payload = decision.to_dict()
        if record:
            payload["id"] = record["id"]
            payload["timestamp"] = record["timestamp"]
        print(json.dumps(payload, indent=2))
    else:
        d = decision
        price = (f"${d.estimated_cost_per_hour_usd:.2f}/hr"
                 if d.estimated_cost_per_hour_usd is not None else "price unknown")
        print(f"PICK: {d.chip.name} on {d.offer.provider}  ({price})")
        if d.estimated_job_cost_usd is not None:
            print(f"      estimated job cost: ${d.estimated_job_cost_usd:.2f} "
                  f"for up to {job.max_runtime_hours:g} h")
        print(f"\nWHY : {d.reason}")
        if d.runners_up:
            print("\nRUNNERS-UP:")
            for r in d.runners_up:
                print(f"  {r.score:.2f}  {r.chip.name} on {r.offer.provider}  ({r.note})")
        if record:
            print(f"\nlogged as {record['id']} -> "
                  f"{args.log_file or default_log_path()}")

    if args.launch_plan:
        adapter = get_adapter(decision.offer.provider)
        print()
        if adapter is None:
            print(f"(no adapter for provider '{decision.offer.provider}'; nothing to plan)")
        else:
            print(adapter.plan_launch(decision).render())
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    if args.annotate:
        if not args.actual:
            print("error: --annotate requires --actual '<json>'", file=sys.stderr)
            return 1
        try:
            actual = json.loads(args.actual)
        except json.JSONDecodeError as e:
            print(f"error: --actual is not valid JSON: {e}", file=sys.stderr)
            return 1
        try:
            rec = annotate_decision(args.annotate, actual, args.log_file)
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 1
        print(f"annotated {rec['id']}: actual = {json.dumps(rec['actual'])}")
        return 0

    records = read_decisions(args.log_file)
    if not records:
        print(f"no decisions logged yet ({args.log_file or default_log_path()})")
        return 0
    shown = records[-args.n:]
    if args.json:
        for rec in shown:
            print(json.dumps(rec, sort_keys=True))
        return 0
    for rec in shown:
        cost = rec.get("estimated_cost_per_hour_usd")
        cost_txt = f"${cost:.2f}/hr" if cost is not None else "?/hr"
        actual = " [actual recorded]" if rec.get("actual") else ""
        print(f"{rec['timestamp']}  {rec['id']}  {rec['job']['name']}: "
              f"{rec['chip']} @ {rec['provider']} {cost_txt}{actual}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pick":
            return cmd_pick(args)
        if args.command == "list":
            return cmd_list(args)
        if args.command == "log":
            return cmd_log(args)
        if args.command == "validate":
            return cmd_validate(args)
    except (CatalogError, JobSpecError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
