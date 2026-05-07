"""
Download pre-trained checkpoints and datasets for EL-MLFFs.

Usage:
    python download_artifacts.py --output-dir ./el-mlffs

The script downloads from a public URL. If the URL is unavailable,
please contact the authors or use your own trained checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# NOTE: Replace with the actual URL after uploading to a public storage
# Examples: HuggingFace, Zenodo, Google Drive, or institutional file server
DEFAULT_ARTIFACTS_URL = "https://github.com/zmyybc/EL-MLFFs/releases/download/v1.0.0/elmlffs_release.tar.gz"

EXPECTED_MD5 = "710f12731f4ae3e7f10c1d6c88ca2b72"  # elmlffs_release.tar.gz


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download with progress."""
    print(f"Downloading from {url}")
    print(f"Saving to {dest}")
    urlretrieve(url, dest)
    print("Download complete.")


def verify_md5(path: Path, expected: str) -> bool:
    """Check MD5 hash."""
    if expected == "PLACEHOLDER":
        return True
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    ok = actual == expected
    print(f"MD5: {actual} {'OK' if ok else 'MISMATCH'}")
    return ok


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract archive (zip or tar.gz)."""
    print(f"Extracting {zip_path} to {extract_to}")
    suffix = zip_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    elif suffix == ".gz" or zip_path.name.endswith(".tar.gz"):
        with tarfile.open(zip_path, "r:gz") as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")
    print("Extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EL-MLFFs artifacts.")
    parser.add_argument("--url", default=DEFAULT_ARTIFACTS_URL, help="URL to artifacts zip")
    parser.add_argument("--output-dir", type=Path, default=Path("el-mlffs"), help="Directory to extract into")
    parser.add_argument("--zip-path", type=Path, default=Path("el-mlffs_artifacts.zip"), help="Temporary zip path")
    parser.add_argument("--skip-verify", action="store_true", help="Skip MD5 verification")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only extract existing zip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_file(args.url, args.zip_path)
        if not args.skip_verify and not verify_md5(args.zip_path, EXPECTED_MD5):
            print("ERROR: MD5 mismatch. The downloaded file may be corrupted.")
            return

    if args.zip_path.exists():
        extract_zip(args.zip_path, args.output_dir)
    else:
        print(f"ERROR: Zip file not found at {args.zip_path}")
        return

    print("\nArtifacts ready:")
    print(f"  Checkpoints: {args.output_dir / 'checkpoints'}")
    print(f"  Data: {args.output_dir / 'data'}")
    print("\nYou can now run training or inference scripts.")


if __name__ == "__main__":
    main()
