"""Deploy GroundTruth to a free Hugging Face Docker Space.

    hf auth login                                  # once, with a token that has write access
    python deploy/push_space.py <username>/groundtruth

Then add GROQ_API_KEY under the Space's Settings -> Variables and secrets.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["Dockerfile", "requirements.txt", "groundtruth", "api", "ui", "data", "deploy", "eval", "tests", ".env.example"]
FRONTMATTER = """---
title: GroundTruth
emoji: ✅
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
short_description: RAG that verifies every sentence of its own answers
---

"""


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy/push_space.py <username>/<space-name>")
    repo_id = sys.argv[1]
    api = HfApi()
    print(f"Logged in as {api.whoami()['name']}")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in INCLUDE:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "cache.json"))
            elif src.exists():
                shutil.copy2(src, stage / name)
        # Spaces read their config from README front matter.
        (stage / "README.md").write_text(FRONTMATTER + (ROOT / "README.md").read_text())
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage,
                          commit_message="Deploy GroundTruth")

    print(f"\nPushed. Build logs: https://huggingface.co/spaces/{repo_id}?logs=build")
    print(f"App URL (after the ~10 min first build): https://{repo_id.replace('/', '-').lower()}.hf.space")
    print("Remember: Settings -> Variables and secrets -> New secret -> GROQ_API_KEY")


if __name__ == "__main__":
    main()
