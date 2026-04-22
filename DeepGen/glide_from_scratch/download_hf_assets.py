from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo snapshot.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. openai/clip-vit-base-patch32")
    parser.add_argument("--local-dir", required=True, help="Directory to store downloaded files")
    parser.add_argument("--revision", default=None, help="Optional branch, tag, or commit")
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional file glob filters, e.g. tokenizer.json merges.txt vocab.json",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Optional file glob exclusions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.repo_id} to {local_dir.resolve()}")


if __name__ == "__main__":
    main()
