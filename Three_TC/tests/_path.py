"""Shared sys.path shim so the test scripts can be run directly."""
import os, sys

HERE      = os.path.dirname(os.path.abspath(__file__))   # .../Three_TC/tests
THREE_TC  = os.path.dirname(HERE)                        # .../Three_TC
REPO_ROOT = os.path.dirname(THREE_TC)                    # repo root

for p in (THREE_TC, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
