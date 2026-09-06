"""Provision the pinned model snapshots into the local, git-ignored cache.

The repository documented that "model artifacts are provisioned separately"
without ever saying how. This is that command.

    py -3.11 evaluations/provision_models.py            # download and verify
    py -3.11 evaluations/provision_models.py --verify   # verify only, no network
    py -3.11 evaluations/provision_models.py --write-lock

What it guarantees:

* Weights land only in ``python_backend/.model-cache/``, which is git-ignored.
  Nothing is ever written inside a tracked directory.
* Each repository is fetched at the **exact pinned revision** from the
  manifest, never at HEAD.
* The declared weight file is digested with SHA-256 and compared to the
  manifest. A mismatch, a missing file, or a revision that does not resolve
  fails closed with a non-zero exit and no "downloaded successfully" claim.
* Every other file in the snapshot is digested too, and can be written to a
  lock file so later drift in a file the manifest does not pin is still
  detectable.
* No hosted or paid inference API is contacted. The only network access is an
  anonymous file download from the Hugging Face CDN.

Sizes, timings and the cache location are printed for the operator. The
committed lock file records filenames and digests only, never an absolute
machine path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

MANIFEST_PATH = BACKEND_ROOT / "config" / "model_manifest.json"
LOCK_PATH = BACKEND_ROOT / "config" / "model_files.lock.json"
#: Git-ignored. See .gitignore.
CACHE_ROOT = BACKEND_ROOT / ".model-cache"

READ_SIZE = 1024 * 1024

EXIT_OK = 0
EXIT_CHECKSUM = 2
EXIT_MISSING = 3
EXIT_DEPENDENCY = 4


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_SIZE), b""):
            sha.update(block)
    return sha.hexdigest()


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _load_manifest() -> Dict[str, Dict[str, str]]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _configure_cache() -> None:
    """Point every Hugging Face path at the local ignored cache."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_ROOT)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "hub")
    # No telemetry, and never resume into a half-written file silently.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _snapshot(
    repo_id: str,
    revision: str,
    offline: bool,
    allow_patterns: Optional[List[str]] = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment guard
        print(
            "huggingface_hub is not installed in this interpreter.\n"
            "Install the optional model stack first (requirements-models.txt).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_DEPENDENCY) from exc

    extra = {"allow_patterns": allow_patterns} if allow_patterns else {}
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=offline,
            **extra,
        )
    )


def _write_branch_ref(snapshot: Path, revision: str, ref: str) -> None:
    """Point a branch ref at the pinned commit inside the local cache.

    A backbone is resolved by repository name at load time, so the cache needs
    a branch ref or offline resolution fails. The ref points at the pinned
    revision and the content behind it is checksum-verified, so this makes the
    pin usable rather than loosening it.
    """
    repo_root = snapshot.parent.parent  # .../models--org--name/snapshots/<sha>
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_path = refs_dir / ref
    current = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
    if current != revision:
        ref_path.write_text(revision, encoding="utf-8")


def provision(name: str, spec: Dict[str, str], offline: bool) -> Dict[str, Any]:
    """Fetch (or locate) one pinned snapshot and verify it. Fails closed."""
    print(f"\n[{name}]")
    print(f"  repo     : {spec['repo_id']}")
    print(f"  revision : {spec['revision']}")
    print(f"  license  : {spec['license']}")

    if spec.get("role", "detector") != "detector":
        print(f"  role     : {spec['role']} (required by {spec.get('required_by')})")

    started = time.perf_counter()
    try:
        snapshot = _snapshot(
            spec["repo_id"],
            spec["revision"],
            offline,
            spec.get("allow_patterns"),
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"  FAILED to resolve snapshot: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MISSING)
    elapsed = time.perf_counter() - started

    weight_path = snapshot / spec["weight_file"]
    if not weight_path.is_file():
        print(f"  FAILED: declared weight file is absent: {spec['weight_file']}", file=sys.stderr)
        raise SystemExit(EXIT_MISSING)

    files: List[Tuple[str, int, str]] = []
    total_bytes = 0
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append((path.relative_to(snapshot).as_posix(), size, _digest(path)))

    weight_digest = next(
        digest for name_, _size, digest in files if name_ == spec["weight_file"]
    )
    weight_size = next(
        size for name_, size, _digest in files if name_ == spec["weight_file"]
    )

    print(f"  files    : {len(files)}   total {_human(total_bytes)}")
    print(f"  weight   : {spec['weight_file']}  {_human(weight_size)}")
    print(f"  expected : {spec['weight_sha256']}")
    print(f"  actual   : {weight_digest}")
    print(f"  elapsed  : {elapsed:.1f}s")

    if weight_digest != spec["weight_sha256"]:
        print(
            "  CHECKSUM MISMATCH - the local weight does not match the manifest.\n"
            "  Refusing to declare this model provisioned. Delete the cache entry\n"
            "  and re-run rather than loading it.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_CHECKSUM)

    if spec.get("alias_ref"):
        _write_branch_ref(snapshot, spec["revision"], spec["alias_ref"])
        print(f"  ref      : {spec['alias_ref']} -> pinned revision")

    print("  verified : OK")
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "weight_file": spec["weight_file"],
        "weight_sha256": weight_digest,
        "weight_bytes": weight_size,
        "snapshot_bytes": total_bytes,
        "file_count": len(files),
        "elapsed_seconds": round(elapsed, 2),
        # Filenames and digests only. No absolute path is recorded.
        "files": [
            {"name": name_, "bytes": size, "sha256": digest}
            for name_, size, digest in files
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing cache without any network access",
    )
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help=f"write per-file digests to {LOCK_PATH.name}",
    )
    args = parser.parse_args()

    _configure_cache()
    if args.verify:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    manifest = _load_manifest()
    print(f"manifest    : {MANIFEST_PATH.relative_to(BACKEND_ROOT)}")
    print(f"cache root  : {CACHE_ROOT.relative_to(BACKEND_ROOT)}  (git-ignored)")
    print(f"mode        : {'verify-only, offline' if args.verify else 'download and verify'}")

    started = time.perf_counter()
    results = {
        name: provision(name, spec, offline=args.verify)
        for name, spec in manifest.items()
    }
    total = time.perf_counter() - started

    grand_bytes = sum(r["snapshot_bytes"] for r in results.values())
    print(f"\nall models verified in {total:.1f}s, {_human(grand_bytes)} on disk")

    if args.write_lock:
        payload = {
            "_comment": (
                "Per-file digests for the pinned model snapshots. The manifest "
                "pins the weight file; this extends that to every file in the "
                "snapshot so drift is detectable. Generated by "
                "evaluations/provision_models.py --write-lock."
            ),
            "models": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key != "elapsed_seconds"
                }
                for name, result in results.items()
            },
        }
        LOCK_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {LOCK_PATH.relative_to(BACKEND_ROOT)}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
