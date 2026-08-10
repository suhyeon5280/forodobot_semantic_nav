"""
Upload a recorded session to Hugging Face Hub as a private dataset.

The session directory name is already a date/time stamp (created by /dataset/start,
e.g. dataset/sessions/2026-07-20_15-45-12), so it's reused as-is for the repo name.

Usage:
  python3 upload_to_hf.py dataset/sessions/2026-07-20_15-45-12
  python3 upload_to_hf.py dataset/sessions/2026-07-20_15-45-12 --repo-name my-custom-name
  python3 upload_to_hf.py dataset/sessions/2026-07-20_15-45-12 --public

Auth: run `huggingface-cli login` once beforehand, or set the HF_TOKEN env var.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Path to a dataset/sessions/<timestamp> folder")
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Override the repo name (defaults to the session folder's date/time name)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset repo public (default: private)",
    )
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        sys.exit(f"Not a directory: {session_dir}")

    session_name = args.repo_name or os.path.basename(session_dir.rstrip("/"))
    repo_name = f"frodobot-drive-{session_name}"

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    whoami = api.whoami()
    username = whoami["name"]
    repo_id = f"{username}/{repo_name}"

    print(f"Creating dataset repo: {repo_id} (private={not args.public})")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    print(f"Uploading {session_dir} -> {repo_id} ...")
    api.upload_folder(
        folder_path=session_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
