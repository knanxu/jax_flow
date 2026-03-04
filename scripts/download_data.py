"""Download robomimic/mimicgen datasets.

Usage:
    # Download lowdim dataset
    python scripts/download_data.py --task lift --dataset_type ph --obs_type lowdim

    # Download image dataset
    python scripts/download_data.py --task lift --dataset_type ph --obs_type image

    # Download from HuggingFace (mimicgen)
    python scripts/download_data.py --task stack --dataset_type ph --obs_type lowdim --source mimicgen
"""

import argparse
import os
from pathlib import Path


def download_robomimic(task, dataset_type, obs_type, data_dir):
    """Download robomimic dataset using robomimic tools."""
    try:
        import robomimic.scripts.download_datasets as download_script
    except ImportError:
        print("Error: robomimic not installed. Install with: pip install robomimic")
        return

    # Map obs_type to hdf5_type
    hdf5_type = "low_dim" if obs_type == "lowdim" else "image"

    print(f"Downloading robomimic dataset:")
    print(f"  Task: {task}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Observation type: {obs_type}")
    print(f"  Target directory: {data_dir}")

    # Use robomimic's download script
    # This will download to ~/.robomimic/datasets/ by default
    # Then we can move/symlink to our data directory
    import sys
    sys.argv = [
        "download_datasets",
        "--tasks", task,
        "--dataset_types", dataset_type,
        "--hdf5_types", hdf5_type,
    ]

    try:
        download_script.main()
        print("\nDownload complete!")
        print(f"Data saved to: ~/.robomimic/datasets/{task}/{dataset_type}/")
        print(f"\nYou can create a symlink:")
        print(f"  ln -s ~/.robomimic/datasets {data_dir}/robomimic")
    except Exception as e:
        print(f"Error during download: {e}")


def download_mimicgen(task, dataset_type, obs_type, data_dir):
    """Download mimicgen dataset from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return

    # Map obs_type to hdf5 filename
    hdf5_type = "low_dim" if obs_type == "lowdim" else "image"

    # MimicGen repo structure: core/{task}/{dataset_type}/{hdf5_type}_v141.hdf5
    filename = f"core/{task}/{dataset_type}/{hdf5_type}_v141.hdf5"

    print(f"Downloading mimicgen dataset from HuggingFace:")
    print(f"  Task: {task}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Observation type: {obs_type}")
    print(f"  Filename: {filename}")

    try:
        file_path = hf_hub_download(
            repo_id="amandlek/mimicgen_datasets",
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir / "mimicgen",
        )
        print(f"\nDownload complete!")
        print(f"Data saved to: {file_path}")
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nAvailable tasks in mimicgen:")
        print("  stack, stack_three, threading, coffee, kitchen,")
        print("  hammer_cleanup, mug_cleanup, pick_place, square, nut_assembly")


def main():
    parser = argparse.ArgumentParser(description="Download robomimic/mimicgen datasets")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., lift, can, square, stack)")
    parser.add_argument("--dataset_type", type=str, default="ph",
                        help="Dataset type: ph (proficient-human), mh (multi-human), mg (machine-generated)")
    parser.add_argument("--obs_type", type=str, default="lowdim",
                        choices=["lowdim", "image"],
                        help="Observation type: lowdim or image")
    parser.add_argument("--source", type=str, default="robomimic",
                        choices=["robomimic", "mimicgen"],
                        help="Data source: robomimic or mimicgen")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Target data directory")

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download based on source
    if args.source == "robomimic":
        download_robomimic(args.task, args.dataset_type, args.obs_type, data_dir)
    else:
        download_mimicgen(args.task, args.dataset_type, args.obs_type, data_dir)


if __name__ == "__main__":
    main()
