"""Download robomimic/mimicgen datasets.

Usage:
    # Download lowdim dataset
    python scripts/download_data.py --task lift --obs_type lowdim

    # Download image dataset
    python scripts/download_data.py --task lift --obs_type image

    # Download from HuggingFace (mimicgen)
    python scripts/download_data.py --task stack --obs_type lowdim --source mimicgen

    # Force re-download
    python scripts/download_data.py --task lift --obs_type lowdim --force
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_flow.data import DatasetManager


def main():
    parser = argparse.ArgumentParser(
        description="Download robomimic/mimicgen datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download lowdim dataset for lift task
  python scripts/download_data.py --task lift --obs_type lowdim

  # Download image dataset for square task
  python scripts/download_data.py --task square --obs_type image

  # Download from mimicgen
  python scripts/download_data.py --task stack --source mimicgen

Available tasks:
  Robomimic: lift, can, square, transport, tool_hang
  MimicGen: stack, stack_three, threading, coffee, kitchen,
            hammer_cleanup, mug_cleanup, pick_place, square, nut_assembly
        """
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., lift, can, square, stack)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="ph",
        choices=["ph", "mh", "mg"],
        help="Dataset type: ph (proficient-human), mh (multi-human), mg (machine-generated)"
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        default="lowdim",
        choices=["lowdim", "image"],
        help="Observation type: lowdim or image"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="robomimic",
        choices=["robomimic", "mimicgen"],
        help="Data source: robomimic or mimicgen"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Dataset Download Tool")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Observation type: {args.obs_type}")
    print(f"Source: {args.source}")
    print(f"Force re-download: {args.force}")
    print("=" * 80)

    # Use DatasetManager to download
    success = DatasetManager.download_dataset(
        task=args.task,
        dataset_type=args.dataset_type,
        obs_type=args.obs_type,
        source=args.source,
        force=args.force,
    )

    if success:
        print("\n" + "=" * 80)
        print("✓ Download successful!")
        print("=" * 80)

        # Show where the data was saved
        hdf5_type = "low_dim" if args.obs_type == "lowdim" else "image"
        if args.source == "robomimic":
            data_path = DatasetManager.ROBOMIMIC_DIR / args.task / args.dataset_type / f"{hdf5_type}_v15.hdf5"
            print(f"Dataset saved to: {data_path}")
        else:
            data_path = DatasetManager.ROBOMIMIC_DIR / "mimicgen" / "core" / args.task / args.dataset_type / f"{hdf5_type}_v141.hdf5"
            print(f"Dataset saved to: {data_path}")

        print("\nYou can now use this dataset in training:")
        print(f"  python scripts/train_bc.py task={args.task}_{args.obs_type}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ Download failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
