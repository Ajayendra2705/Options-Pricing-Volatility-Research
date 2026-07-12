"""
SHA256 Manifest Generator
=========================
Compute SHA256 hash for every file in data/raw/, write manifest.
Ensures raw data immutability is provably reproducible offline.

Usage:
    python scripts/sha256_manifest.py
    python scripts/sha256_manifest.py --verify   # verify existing manifest
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = DATA_RAW / "manifest.json"


def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(data_dir: Path) -> dict:
    """Generate manifest for all files in data_dir."""
    manifest = {
        "generated": datetime.now().isoformat(),
        "root": str(data_dir),
        "files": {},
    }

    files = sorted(f for f in data_dir.rglob("*") if f.is_file() and f.name != "manifest.json")
    if not files:
        print(f"No files found in {data_dir}")
        return manifest

    for f in files:
        rel = f.relative_to(data_dir).as_posix()
        size = f.stat().st_size
        digest = sha256_file(f)
        manifest["files"][rel] = {
            "sha256": digest,
            "size_bytes": size,
        }
        print(f"  {rel}: {digest[:16]}... ({size:,} bytes)")

    return manifest


def verify_manifest(data_dir: Path, manifest_path: Path) -> bool:
    """Verify files against existing manifest. Returns True if all match."""
    if not manifest_path.exists():
        print(f"ERROR: No manifest at {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ok = True
    for rel, info in manifest["files"].items():
        fpath = data_dir / rel
        if not fpath.exists():
            print(f"  MISSING: {rel}")
            all_ok = False
            continue

        actual = sha256_file(fpath)
        if actual != info["sha256"]:
            print(f"  MISMATCH: {rel}")
            print(f"    expected: {info['sha256'][:16]}...")
            print(f"    actual:   {actual[:16]}...")
            all_ok = False
        else:
            print(f"  OK: {rel}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SHA256 manifest for data/raw/")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifest")
    parser.add_argument("--data-dir", type=str, default=str(DATA_RAW))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"

    if args.verify:
        print(f"Verifying manifest: {manifest_path}")
        ok = verify_manifest(data_dir, manifest_path)
        if ok:
            print("\n[OK] All files match manifest.")
        else:
            print("\n[FAIL] Manifest verification FAILED.")
            sys.exit(1)
    else:
        print(f"Generating manifest for: {data_dir}")
        manifest = generate_manifest(data_dir)
        with open(manifest_path, "w", newline="\n") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n-> Wrote {manifest_path}")
        print(f"  {len(manifest['files'])} file(s) hashed.")


if __name__ == "__main__":
    main()
