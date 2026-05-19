"""Check that commit messages follow the Conventional Commits format."""

import re
import sys


def main() -> int:
    msg = open(sys.argv[1]).read().strip()  # noqa: SIM115
    types = "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
    pattern = rf"^({types})(\(.+\))?!?:\s.+"
    if re.match(pattern, msg):
        return 0
    print(f"Bad commit message:\n  {msg}")
    print()
    print("Expected: type(scope): description")
    print("Types: feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
