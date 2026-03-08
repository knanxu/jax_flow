"""Download and convert all MimicGen datasets.

Downloads raw HDF5 files from HuggingFace and converts to lowdim/image format.
Handles ALL_PROXY socks issue automatically.
"""

import os
import sys
from pathlib import Path

# Fix socks proxy issue with httpx
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))

MIMICGEN_TASKS = [
    "stack",
    "stack_three",
    "threading",
    "coffee",
    "kitchen",
    "hammer_cleanup",
    "mug_cleanup",
    "pick_place",
    "nut_assembly",
]

BASE_DIR = Path.home() / ".robomimic" / "mimicgen"


def download_task(task: str) -> bool:
    """Download raw HDF5 for a task from HuggingFace."""
    raw_path = BASE_DIR / "core" / f"{task}_d0.hdf5"
    if raw_path.exists():
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        print(f"  Already downloaded: {raw_path} ({size_mb:.0f} MB)")
        return True

    filename = f"core/{task}_d0.hdf5"
    print(f"  Downloading {filename} ...")

    try:
        path = hf_hub_download(
            repo_id="amandlek/mimicgen_datasets",
            filename=filename,
            repo_type="dataset",
            local_dir=str(BASE_DIR),
        )
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  Downloaded: {path} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    print("=" * 80)
    print("MimicGen Dataset Download + Convert")
    print("=" * 80)

    # Phase 1: Download all raw files
    print("\n--- Phase 1: Download raw HDF5 files ---\n")
    downloaded = []
    failed = []

    for task in MIMICGEN_TASKS:
        print(f"[{task}]")
        if download_task(task):
            downloaded.append(task)
        else:
            failed.append(task)

    print(f"\nDownloaded: {len(downloaded)}/{len(MIMICGEN_TASKS)}")
    if failed:
        print(f"Failed: {failed}")

    # Phase 2: Convert all downloaded files
    print("\n--- Phase 2: Convert to lowdim/image format ---\n")

    # Import convert script
    sys.path.insert(0, str(Path(__file__).parent))
    from convert_mimicgen import convert_task

    converted = 0
    for task in downloaded:
        raw_path = BASE_DIR / "core" / f"{task}_d0.hdf5"
        if raw_path.exists():
            convert_task(raw_path, task, BASE_DIR)
            converted += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Downloaded: {len(downloaded)}/{len(MIMICGEN_TASKS)}")
    print(f"Converted:  {converted}/{len(downloaded)}")
    if failed:
        print(f"Failed:     {failed}")

    print("\nAvailable datasets:")
    for task in downloaded:
        lowdim = BASE_DIR / "core" / task / "ph" / "low_dim_v141.hdf5"
        image = BASE_DIR / "core" / task / "ph" / "image_v141.hdf5"
        ld = "OK" if lowdim.exists() else "MISSING"
        im = "OK" if image.exists() else "MISSING"
        print(f"  {task}: lowdim={ld}, image={im}")


if __name__ == "__main__":
    main()
