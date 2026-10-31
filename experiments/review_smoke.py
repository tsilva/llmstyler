"""Retired agent smoke entrypoint; no credential or application operations."""

import sys


def main():
    print(
        "The retired direct-OpenRouter judge is disabled. A replacement needs its own reviewed AgentBridge workflow; no credential retrieval or review request was performed.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
