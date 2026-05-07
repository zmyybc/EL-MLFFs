#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path


SPLITS = {
    "2M": {
        "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "md5": "953474cb93f0b08cdc523399f03f7c36",
        "compressed_bytes": 3_400_000_000,
        "uncompressed_bytes": 17_000_000_000,
    },
    "val_id": {
        "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "md5": "f57f7f5c1302637940f2cc858e789410",
        "compressed_bytes": 1_700_000_000,
        "uncompressed_bytes": 8_300_000_000,
    },
    "val_ood_ads": {
        "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "md5": "431ab0d7557a4639605ba8b67793f053",
        "compressed_bytes": 1_700_000_000,
        "uncompressed_bytes": 8_200_000_000,
    },
    "val_ood_cat": {
        "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "md5": "532d6cd1fe541a0ddb0aa0f99962b7db",
        "compressed_bytes": 1_700_000_000,
        "uncompressed_bytes": 8_300_000_000,
    },
    "val_ood_both": {
        "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "md5": "5731862978d80502bbf7017d68c2c729",
        "compressed_bytes": 1_900_000_000,
        "uncompressed_bytes": 9_500_000_000,
    },
}

DEFAULT_SPLITS = ["2M", "val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"]
USER_AGENT = "Mozilla/5.0 (compatible; OC20Downloader/1.0)"
CHUNK_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official OC20 S2EF train/validation splits and extract them."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/bn/bangchen/EL-MLFFs/data/oc20"),
        help="Output root containing raw/, extracted/, and metadata/ directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=sorted(SPLITS.keys()),
        help="OC20 S2EF splits to download.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download tar files without extraction.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification after download.",
    )
    return parser.parse_args()


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    raw_dir = root / "raw"
    extracted_dir = root / "extracted"
    metadata_dir = root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, extracted_dir, metadata_dir


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "wget",
            "-c",
            "--progress=dot:giga",
            "--tries=5",
            "--user-agent",
            USER_AGENT,
            "-o",
            "/dev/stderr",
            "-O",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.rename(destination)


def extract_tar(archive_path: Path, destination_dir: Path) -> None:
    marker = destination_dir / ".extracted"
    if marker.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(destination_dir)
    marker.write_text(f"extracted_from={archive_path.name}\n", encoding="utf-8")


def write_metadata(metadata_dir: Path, splits: list[str]) -> None:
    payload = {
        "source": "OC20 official S2EF split downloads",
        "source_doc": "https://fair-chem.github.io/catalysts/datasets/oc20.html",
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {split: SPLITS[split] for split in splits},
    }
    with (metadata_dir / "oc20_s2ef_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_readme(root: Path, splits: list[str]) -> None:
    lines = [
        "# OC20 S2EF Data Staging",
        "",
        "This directory contains official OC20 S2EF train/validation tarballs and extracted contents.",
        "",
        "Downloaded splits:",
    ]
    for split in splits:
        spec = SPLITS[split]
        lines.append(
            f"- `{split}`: {spec['url']} (MD5 `{spec['md5']}`, "
            f"compressed ~{human_bytes(spec['compressed_bytes'])}, "
            f"uncompressed ~{human_bytes(spec['uncompressed_bytes'])})"
        )
    lines.extend(
        [
            "",
            "Layout:",
            "- `raw/`: original `.tar` files",
            "- `extracted/`: extracted per-split directories",
            "- `metadata/oc20_s2ef_manifest.json`: split metadata",
            "",
            "Note: this repository currently trains from `extxyz`, not directly from OC20 S2EF archives/LMDBs.",
            "If you want to train this codebase on OC20 next, a dataset conversion or OC20-native loader still needs to be added.",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir, extracted_dir, metadata_dir = ensure_dirs(args.root)

    print(f"Preparing OC20 S2EF data under: {args.root}")
    total_compressed = sum(SPLITS[split]["compressed_bytes"] for split in args.splits)
    total_uncompressed = sum(SPLITS[split]["uncompressed_bytes"] for split in args.splits)
    print(
        f"Planned download: {human_bytes(total_compressed)} compressed, "
        f"~{human_bytes(total_uncompressed)} extracted"
    )

    for split in args.splits:
        spec = SPLITS[split]
        archive_path = raw_dir / f"{split}.tar"
        print(f"[{split}] {spec['url']}")
        if archive_path.exists():
            print(f"  archive exists: {archive_path}")
        else:
            download_file(spec["url"], archive_path)

        if not args.skip_md5:
            actual_md5 = md5sum(archive_path)
            if actual_md5 != spec["md5"]:
                raise RuntimeError(
                    f"MD5 mismatch for {archive_path.name}: expected {spec['md5']}, got {actual_md5}"
                )
            print(f"  md5 ok: {actual_md5}")

        if not args.download_only:
            extract_dir = extracted_dir / split
            print(f"  extracting to {extract_dir}")
            extract_tar(archive_path, extract_dir)

    write_metadata(metadata_dir, args.splits)
    write_readme(args.root, args.splits)
    print("OC20 S2EF staging finished.")


if __name__ == "__main__":
    main()
