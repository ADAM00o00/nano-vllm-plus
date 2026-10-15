# -*- coding: utf-8 -*-
"""Rewrite the .tmp/ paths inside the worklogs to the archived experiments/ tree.

ASCII-only source: the anchors are read from the documents themselves, so this
file never has to hold non-ASCII literals.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["docs/sm89_worklog.md", "docs/cmp_vllm_vs_nanovllm_worklog.md"]
REWRITES = [
    # the .tmp/ scratch tree became experiments/
    (".tmp/", "experiments/"),
    # the scripts/ + logs/ split was flattened, then the scripts got grouped by
    # workstream; worklogs now name just the file and the README maps name to
    # folder, except for the sync helper which is named in full.
    ("experiments/scripts/", "experiments/"),
    ("experiments/sync_to_wsl.sh", "experiments/shared/sync_to_wsl.sh"),
]

total = 0
for rel in FILES:
    p = ROOT / rel
    text = p.read_text(encoding="utf-8")
    for old, new in REWRITES:
        n = text.count(old)
        if n:
            text = text.replace(old, new)
        total += n
    p.write_text(text, encoding="utf-8", newline="")
    sys.stdout.write(f"{rel}: rewritten\n")
sys.stdout.write(f"total rewrites: {total}\n")
