"""Workaround for sandboxed tempfile.mkdtemp creating inaccessible directories.

Use this to run pip in restricted environments:
    python .pip_fix.py install ...
"""
import os
import sys
import tempfile
import uuid


def _mkdtemp(suffix=None, prefix=None, dir=None):
    if dir is None:
        dir = tempfile.gettempdir()
    os.makedirs(dir, exist_ok=True)
    name = (prefix or "tmp") + uuid.uuid4().hex + (suffix or "")
    path = os.path.join(dir, name)
    os.makedirs(path, exist_ok=False)
    return path


tempfile.mkdtemp = _mkdtemp
tempfile._mkdtemp_inner = _mkdtemp  # best effort

from pip._internal.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
