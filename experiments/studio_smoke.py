"""Retired agent smoke entrypoint; no credential or application operations."""

import sys


def main():
    print(
        "Studio session automation is disabled: agents must not read owner auth storage or mint sessions. Use Studio through its human sign-in flow; offline fixtures do not require a real session.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
