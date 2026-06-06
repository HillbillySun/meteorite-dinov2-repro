#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


UPLOAD_PATTERNS = [
    "README.md",
    "config.json",
    "configuration_meteorite_dinov2.py",
    "modeling_meteorite_dinov2.py",
    "pytorch_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the packaged meteorite model to ModelScope.")
    parser.add_argument("--repo-id", required=True, help="ModelScope repository ID, e.g. user/model-name.")
    parser.add_argument("--model-dir", type=Path, default=Path("hf_direct_full_model"))
    parser.add_argument("--revision", default="master")
    parser.add_argument("--commit-message", default="Upload callable DINOv2-B meteorite classifier")
    parser.add_argument("--private", action="store_true", help="Create or update a private repository.")
    parser.add_argument("--license", default="MIT")
    parser.add_argument("--token-env", default="MODELSCOPE_API_TOKEN")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    required = {
        "config.json",
        "configuration_meteorite_dinov2.py",
        "modeling_meteorite_dinov2.py",
        "pytorch_model.bin",
        "README.md",
    }
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Model directory is incomplete; missing: {missing}")

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f"Missing ModelScope token. Export it first: export {args.token_env}='your-token'"
        )

    try:
        from modelscope import HubApi
    except ImportError as exc:
        raise RuntimeError(
            "Uploading requires the 'modelscope' package. "
            "Install the repository requirements or run: pip install modelscope"
        ) from exc

    visibility = "private" if args.private else "public"
    api = HubApi()
    print(f"[modelscope_upload] repo={args.repo_id}")
    print(f"[modelscope_upload] model_dir={model_dir}")
    print(f"[modelscope_upload] visibility={visibility}")

    repo_url = api.create_repo(
        repo_id=args.repo_id,
        token=token,
        visibility=visibility,
        repo_type="model",
        license=args.license,
        exist_ok=True,
        create_default_config=False,
    )
    print(f"[modelscope_upload] repository ready: {repo_url}")

    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=model_dir,
        path_in_repo="",
        allow_patterns=UPLOAD_PATTERNS,
        token=token,
        revision=args.revision,
        commit_message=args.commit_message,
        max_workers=args.max_workers,
        use_cache=True,
    )
    print(f"[modelscope_upload] commit={result}")
    print(f"[modelscope_upload] finished: https://modelscope.cn/models/{args.repo_id}")


if __name__ == "__main__":
    main()
