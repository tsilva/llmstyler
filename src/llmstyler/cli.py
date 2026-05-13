from __future__ import annotations

import argparse

from llmstyler.mix import build_mix
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

    runbook = subparsers.add_parser("make-runbook", help="Generate a Runbook training notebook")
    runbook.add_argument("config")
    runbook.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.command == "build-mix":
        build_mix(args.config, push=args.push_to_hub)
    elif args.command == "restyle":
        restyle(args.config, estimate_only=args.estimate_only, push=args.push_to_hub)
    elif args.command == "make-runbook":
        make_runbook(args.config, output_dir=args.output_dir)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
