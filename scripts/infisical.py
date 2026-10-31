"""Deprecated path for the keyless agent launcher; never reads Infisical."""

import sys
from pathlib import Path

from llmstyler.agent import AgentModeError, launch

ROOT = Path(__file__).resolve().parents[1]


def main(arguments):
    launch(arguments, root=ROOT)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except AgentModeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None
