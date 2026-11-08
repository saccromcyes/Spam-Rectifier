"""Publish this repository to a Hugging Face Space (Docker SDK).

Creates the Space if needed, syncs the LLM key into the Space's secrets, and uploads the code
with a README that carries the Space metadata. Used by .github/workflows/deploy.yml.

    HF_TOKEN=... HF_SPACE=username/spam-rectifier [LLM_API_KEY=...] python scripts/deploy_space.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_HEADER = """---
title: SpamRectifier
emoji: 📨
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Naive Bayes vs embeddings vs LLM on real SMS spam
---

"""
# Variables that are safe to expose as Space settings; the API key goes in as a secret.
PUBLIC_VARS = ["LLM_BASE_URL", "LLM_MODEL"]


def main() -> int:
    token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE")
    if not token or not space:
        print("HF_TOKEN and HF_SPACE are required", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    if os.environ.get("LLM_API_KEY"):
        api.add_space_secret(space, "LLM_API_KEY", os.environ["LLM_API_KEY"])
    for name in PUBLIC_VARS:
        if os.environ.get(name):
            api.add_space_variable(space, name, os.environ[name])

    with tempfile.TemporaryDirectory() as tmp:
        # Only files tracked by git are published (no local caches, .env or databases).
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
        for rel in files:
            if rel.startswith((".github/", "tests/", "docs/")):
                continue
            dest = Path(tmp) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_HEADER + readme.read_text())

        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        api.upload_folder(
            folder_path=tmp,
            repo_id=space,
            repo_type="space",
            commit_message=f"Deploy {sha} from GitHub",
            delete_patterns=["*"],
        )
    print(f"Deployed to https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
