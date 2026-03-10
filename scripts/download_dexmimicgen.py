"""Download DexMimicGen Panda dual-arm datasets from HuggingFace.

Downloads raw HDF5 files and converts to lowdim/image format.

Usage:
    # Download all 3 tasks
    python scripts/download_dexmimicgen.py

    # Download specific tasks
    python scripts/download_dexmimicgen.py --tasks two_arm_threading two_arm_transport

    # Force re-download
    python scripts/download_dexmimicgen.py --force
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TASKS = {
    "two_arm_threading": "generated/two_arm_threading.hdf5",
    "two_arm_three_piece_assembly": "generated/two_arm_three_piece_assembly.hdf5",
    "two_arm_transport": "generated/two_arm_transport.hdf5",
}

BASE_DIR = Path.home() / ".dexmimicgen" / "datasets"
HF_REPO = "MimicGen/dexmimicgen_datasets"


def download_file(hf_filename: str, base_dir: Path, force: bool = False) -> Path | None:
    """Download a single file from HuggingFace."""
    local_path = base_dir / hf_filename
    if local_path.exists() and not force:
        print(f"  Already exists: {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{hf_filename}"

    env = os.environ.copy()
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)

    print(f"  Downloading {hf_filename}...")
    try:
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", url, "-O", str(local_path)],
            env=env, check=True, timeout=3600,
        )
        return local_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  wget failed ({e}), trying huggingface_hub...")
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_filename,
                repo_type="dataset",
                local_dir=str(base_dir),
            )
            return local_path
        except Exception as e2:
            print(f"  Error: {e2}")
            if local_path.exists():
                local_path.unlink()
            return None


def main():
    parser = argparse.ArgumentParser(description="Download DexMimicGen Panda datasets")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        help=f"Tasks to download (default: all). Choices: {list(TASKS.keys())}")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--base_dir", type=str, default=str(BASE_DIR))
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    convert_script = Path(__file__).parent / "convert_dexmimicgen.py"

    print(f"DexMimicGen Panda Dataset Downloader")
    print(f"Base dir: {base_dir}")
    print(f"Tasks: {args.tasks}")
    print()

    success = 0
    for task in args.tasks:
        if task not in TASKS:
            print(f"[{task}] Unknown task, skipping")
            continue

        print(f"[{task}]")
        hf_filename = TASKS[task]

        # Download
        raw_path = download_file(hf_filename, base_dir, force=args.force)
        if raw_path is None:
            print(f"  FAILED to download")
            continue

        size_gb = raw_path.stat().st_size / 1024**3
        print(f"  Raw file: {raw_path} ({size_gb:.1f} GB)")

        # Convert
        print(f"  Converting...")
        try:
            cmd = ["python", str(convert_script), "--input", str(raw_path), "--task", task,
                   "--base_dir", str(base_dir)]
            if args.force:
                cmd.append("--force")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            success += 1
        except subprocess.CalledProcessError as e:
            print(f"  Conversion failed: {e.stderr}")

    print(f"\nDone: {success}/{len(args.tasks)} tasks downloaded and converted.")
    return 0 if success == len(args.tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
