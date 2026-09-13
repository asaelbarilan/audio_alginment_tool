"""Upload a local dataset folder (see DATASET_SPEC.md) to the S3 bucket, under a
top-level prefix matching the dataset name -- the same layout BucketDataset in
align_tag_server.py reads back:

    python -m hebrew_training.upload_dataset --dataset ivrit-ai

uploads data/datasets/ivrit-ai/{metadata.json, manifest.jsonl, audio/*} to
s3://$S3_BUCKET/ivrit-ai/...

Credentials and endpoint come from the environment (S3_ENDPOINT, S3_REGION,
S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET) exactly like align_tag_server.py:
loaded from .env via python-dotenv if present, never read or printed by this script.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name -- both the local folder under --datasets-folder and the "
        "S3 prefix it is uploaded under.",
    )
    parser.add_argument(
        "--datasets-folder",
        type=Path,
        default=Path("data/datasets"),
        help="Local folder holding one subfolder per dataset (default: data/datasets).",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATASETS_BUCKET") or os.environ.get("S3_BUCKET", ""),
        help="Target bucket. Defaults to DATASETS_BUCKET or S3_BUCKET from the "
        "environment / .env, same fallback align_tag_server.py uses.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel upload threads (default: 8)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without touching S3.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload every file even when the bucket already has one the same "
        "size at that key (default: skip those, since audio files never change "
        "content at a fixed size in practice and this dataset can be large).",
    )
    return parser.parse_args()


def s3_client(endpoint: str, region: str, access_key: str, secret_key: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # This endpoint rejects virtual-hosted-style requests (SSL errors on a
        # bucket name it does not have a wildcard cert for); path-style always works.
        config=Config(s3={"addressing_style": "path"}),
    )


def iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def existing_sizes(client, bucket: str, prefix: str) -> dict[str, int]:
    """Current object sizes under the prefix, so a re-run can skip unchanged files."""
    sizes: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def upload_one(client, bucket: str, key: str, path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    return key


def main() -> None:
    args = parse_args()

    dataset_dir = args.datasets_folder / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"no such dataset folder: {dataset_dir}")
    for required in ("metadata.json", "manifest.jsonl"):
        if not (dataset_dir / required).is_file():
            raise SystemExit(f"{dataset_dir} is missing {required}; see DATASET_SPEC.md")

    if not args.bucket:
        raise SystemExit(
            "no bucket: pass --bucket or set S3_BUCKET / DATASETS_BUCKET in the "
            "environment or .env"
        )

    files = iter_files(dataset_dir)
    if not files:
        raise SystemExit(f"{dataset_dir} has no files to upload")

    prefix = f"{args.dataset}/"
    total_bytes = sum(p.stat().st_size for p in files)
    print(
        f"{len(files)} files, {total_bytes / 1e6:.1f} MB, "
        f"{dataset_dir} -> s3://{args.bucket}/{prefix}"
    )

    if args.dry_run:
        for path in files:
            key = prefix + str(path.relative_to(dataset_dir)).replace(os.sep, "/")
            print(f"  would upload {key} ({path.stat().st_size} bytes)")
        return

    missing = [
        var
        for var in ("S3_ENDPOINT", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
        if not os.environ.get(var)
    ]
    if missing:
        raise SystemExit(
            f"missing from the environment / .env: {', '.join(missing)} "
            "(needed to reach S3; this script never reads .env itself)"
        )

    client = s3_client(
        endpoint=os.environ["S3_ENDPOINT"],
        region=os.environ.get("S3_REGION", "us-east-1"),
        access_key=os.environ["S3_ACCESS_KEY_ID"],
        secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )

    remote_sizes = {} if args.force else existing_sizes(client, args.bucket, prefix)

    jobs: list[tuple[str, Path]] = []
    skipped = 0
    for path in files:
        key = prefix + str(path.relative_to(dataset_dir)).replace(os.sep, "/")
        if not args.force and remote_sizes.get(key) == path.stat().st_size:
            skipped += 1
            continue
        jobs.append((key, path))

    if skipped:
        print(f"skipping {skipped} file(s) already present with the same size")
    if not jobs:
        print("nothing to upload")
        return

    uploaded = 0
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(upload_one, client, args.bucket, key, path): key
            for key, path in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                failed.append((key, str(exc)))
                continue
            uploaded += 1
            if uploaded % 25 == 0 or uploaded == len(jobs):
                print(f"  {uploaded}/{len(jobs)} uploaded")

    print(f"done: {uploaded} uploaded, {skipped} skipped, {len(failed)} failed")
    if failed:
        for key, err in failed:
            print(f"  FAILED {key}: {err}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
