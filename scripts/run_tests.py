#!/usr/bin/env python3
"""Run every offline prose plugin test command with failure propagation."""
from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "unittest", "-v", "test_core.py", "test_sentence_context.py", "test_lifecycle_logging.py"],
    [sys.executable, "-m", "pytest", "--import-mode=importlib", "-q", "test_lifecycle_logging.py"],
]


def main() -> int:
    for command in COMMANDS:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
