from __future__ import annotations

import argparse

from llmstyler.mix import build_mix
from llmstyler.pipeline import train_style
from llmstyler.restyle import restyle
from llmstyler.runbook import make_runbook


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmstyler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-mix", help="Build a base chat mix dataset")
    build.add_argument("config")
    build.add_argument("--push-to-hub", action="store_true", default=None)
    build.add_argument("--no-push-to-hub", action="store_false", dest="push_to_hub")

    restyle_parser = subparsers.add_parser("restyle", help="Restyle marked dataset rows")
    restyle_parser.add_argument("config")
    restyle_parser.add_argument("--estimate-only", action="store_true")
    restyle_parser.add_argument("--push-to-hub", action="store_true", default=None)
    restyle_parser.add_argument("--no-push-to-hub", action="store_false", dest="push_to_hub")

    runbook = subparsers.add_parser(
        "runstep",
        aliases=["make-runbook"],
        help="Generate and launch a Runbook training step",
    )
    runbook.add_argument("config")
    runbook.add_argument("--output-dir", default=None)
    runbook.add_argument("--run-output", default=None)
    runbook.add_argument("--no-run", action="store_false", dest="run")

    train = subparsers.add_parser(
        "train-style",
        help="Run the full style dataset and training pipeline from one OmegaConf config",
    )
    train.add_argument("config")
    train.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    train.add_argument("--push-to-hub", action="store_true", default=None)
    train.add_argument("--no-push-to-hub", action="store_false", dest="push_to_hub")
    train.add_argument("--no-run", action="store_false", dest="run")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--force", action="store_true", help="Rerun steps even when cached")

    args, unknown_args = parser.parse_known_args()
    if args.command == "train-style":
        args.overrides.extend(unknown_args)
    elif unknown_args:
        parser.error("unrecognized arguments: " + " ".join(unknown_args))

    if args.command == "build-mix":
        build_mix(args.config, push=args.push_to_hub)
    elif args.command == "restyle":
        restyle(args.config, estimate_only=args.estimate_only, push=args.push_to_hub)
    elif args.command in {"runstep", "make-runbook"}:
        make_runbook(
            args.config,
            output_dir=args.output_dir,
            run=args.run,
            run_output=args.run_output,
        )
    elif args.command == "train-style":
        train_style(
            args.config,
            overrides=args.overrides,
            run=args.run,
            push=args.push_to_hub,
            dry_run=args.dry_run,
            force=args.force,
        )
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
