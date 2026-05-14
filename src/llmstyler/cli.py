from __future__ import annotations

import argparse
import sys

from llmstyler.ease import create_pipeline_config, estimate_config, inspect_config, run_doctor
from llmstyler.mix import build_mix
from llmstyler.pipeline import train_style
from llmstyler.restyle import restyle
from llmstyler.runbook import make_runbook


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmstyler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new = subparsers.add_parser("new", help="Create a complete style pipeline config")
    new.add_argument("id", help="Pipeline id, used for local paths and artifact names")
    new.add_argument("--owner", required=True, help="Hugging Face owner/org for publishable repos")
    new.add_argument("--output", default=None, help="Config path to write")
    new.add_argument("--name", default=None, help="Human-readable pipeline name")
    new.add_argument("--version", default="v1")
    new.add_argument(
        "--base-model",
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="Base instruct model for the training step",
    )
    new.add_argument("--style-id", default=None)
    new.add_argument(
        "--style-prompt",
        default=None,
        help="Rewrite prompt text or path to a prompt file",
    )
    new.add_argument("--style-prompt-file", default=None, help="Path to a rewrite prompt file")
    new.add_argument("--rows", type=int, default=1000, help="Approximate rows to sample")
    new.add_argument("--restyle-rate", type=float, default=0.2)
    new.add_argument("--force", action="store_true", help="Overwrite an existing config")

    doctor = subparsers.add_parser("doctor", help="Check local setup before running a pipeline")
    doctor.add_argument("config", nargs="?")
    doctor.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    doctor.add_argument("--publish", action="store_true", help="Also check Hugging Face auth")
    doctor.add_argument("--no-run", action="store_false", dest="run")

    estimate = subparsers.add_parser("estimate", help="Estimate restyle calls and cost")
    estimate.add_argument("config")
    estimate.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")

    run = subparsers.add_parser(
        "run",
        help="Run a full style pipeline locally by default; pass --publish to upload artifacts",
    )
    run.add_argument("config")
    run.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    run.add_argument("--publish", action="store_true", default=False)
    run.add_argument("--push-to-hub", action="store_true", dest="publish")
    run.add_argument("--no-publish", action="store_false", dest="publish")
    run.add_argument("--no-push-to-hub", action="store_false", dest="publish")
    run.add_argument("--no-run", action="store_false", dest="run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true", help="Rerun steps even when cached")

    inspect = subparsers.add_parser("inspect", help="Summarize pipeline outputs and cache status")
    inspect.add_argument("config")
    inspect.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")

    build = subparsers.add_parser("build-mix", help="Build a base chat mix dataset")
    build.add_argument("config")
    build.add_argument("--push-to-hub", action="store_true", default=None)
    build.add_argument("--no-push-to-hub", action="store_false", dest="push_to_hub")
    build.add_argument("--force", action="store_true", help="rebuild the datamixxer artifact")

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
    if args.command in {"doctor", "estimate", "inspect", "run", "train-style"}:
        args.overrides.extend(unknown_args)
    elif unknown_args:
        parser.error("unrecognized arguments: " + " ".join(unknown_args))

    if args.command == "new":
        create_pipeline_config(
            pipeline_id=args.id,
            owner=args.owner,
            output_path=args.output,
            name=args.name,
            version=args.version,
            base_model=args.base_model,
            style_id=args.style_id,
            style_prompt=args.style_prompt,
            style_prompt_file=args.style_prompt_file,
            rows=args.rows,
            restyle_rate=args.restyle_rate,
            force=args.force,
        )
    elif args.command == "doctor":
        sys.exit(
            run_doctor(
                args.config,
                overrides=args.overrides,
                publish=args.publish,
                run=args.run,
            )
        )
    elif args.command == "estimate":
        estimate_config(args.config, overrides=args.overrides)
    elif args.command == "inspect":
        inspect_config(args.config, overrides=args.overrides)
    elif args.command == "run":
        launch_training = args.run and args.publish
        if args.run and not args.publish:
            print("Skipping remote training launch; pass --publish to upload the dataset first.")
        train_style(
            args.config,
            overrides=args.overrides,
            run=launch_training,
            push=args.publish,
            dry_run=args.dry_run,
            force=args.force,
        )
    elif args.command == "build-mix":
        build_mix(args.config, push=args.push_to_hub, force=args.force)
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
