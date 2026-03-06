"""Batch download all datasets for robomimic and mimicgen tasks.

This script downloads all lowdim and image datasets.
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_flow.data import DatasetManager


# Task lists
ROBOMIMIC_TASKS = ['lift', 'can', 'square', 'transport', 'tool_hang']
MIMICGEN_TASKS = [
    'stack', 'stack_three', 'threading',
    'coffee', 'kitchen', 'hammer_cleanup',
    'mug_cleanup', 'pick_place', 'nut_assembly'
]


def download_robomimic_datasets(tasks, obs_type='lowdim', dataset_type='ph'):
    """Download robomimic datasets."""
    print(f"\n{'='*80}")
    print(f"Downloading Robomimic {obs_type.upper()} Datasets")
    print(f"{'='*80}")

    success_count = 0
    failed_tasks = []

    for task in tasks:
        print(f"\n[{task}]")

        # Check if already exists
        exists, path = DatasetManager.check_dataset_exists(
            dataset_path=f"data/robomimic/{task}/{dataset_type}/{'low_dim' if obs_type == 'lowdim' else 'image'}_v15.hdf5",
            task=task,
            dataset_type=dataset_type,
            obs_type=obs_type,
        )

        if exists:
            print(f"  ✓ Already exists: {path}")
            success_count += 1
            continue

        # Download
        success = DatasetManager.download_dataset(
            task=task,
            dataset_type=dataset_type,
            obs_type=obs_type,
            source='robomimic',
            force=False,
        )

        if success:
            success_count += 1
        else:
            failed_tasks.append(task)

    return success_count, failed_tasks


def download_mimicgen_datasets(tasks, obs_type='lowdim', dataset_type='ph'):
    """Download mimicgen datasets."""
    print(f"\n{'='*80}")
    print(f"Downloading MimicGen {obs_type.upper()} Datasets")
    print(f"{'='*80}")

    success_count = 0
    failed_tasks = []

    for task in tasks:
        print(f"\n[{task}]")

        # Check if already exists
        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"
        expected_path = DatasetManager.ROBOMIMIC_DIR / "mimicgen" / "core" / task / dataset_type / f"{hdf5_type}_v141.hdf5"

        if expected_path.exists():
            print(f"  ✓ Already exists: {expected_path}")
            success_count += 1
            continue

        # Download
        success = DatasetManager.download_dataset(
            task=task,
            dataset_type=dataset_type,
            obs_type=obs_type,
            source='mimicgen',
            force=False,
        )

        if success:
            success_count += 1
        else:
            failed_tasks.append(task)

    return success_count, failed_tasks


def generate_robomimic_image_datasets(tasks, dataset_type='ph'):
    """Generate robomimic image datasets from demo files."""
    print(f"\n{'='*80}")
    print(f"Generating Robomimic IMAGE Datasets")
    print(f"{'='*80}")

    success_count = 0
    failed_tasks = []

    for task in tasks:
        print(f"\n[{task}]")

        # Check if image dataset already exists
        image_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / "image_v141.hdf5"
        if image_path.exists():
            print(f"  ✓ Already exists: {image_path}")
            success_count += 1
            continue

        # Check if demo file exists
        demo_candidates = [
            DatasetManager.ROBOMIMIC_DIR / task / dataset_type / "demo_v141.hdf5",
            DatasetManager.ROBOMIMIC_DIR / task / dataset_type / "demo_v15.hdf5",
        ]

        demo_path = None
        for candidate in demo_candidates:
            if candidate.exists():
                demo_path = candidate
                break

        if not demo_path:
            print(f"  ✗ Demo file not found, skipping")
            failed_tasks.append(task)
            continue

        # Generate image dataset
        print(f"  Generating from {demo_path.name}...")
        cmd = [
            "python", "scripts/generate_image_dataset.py",
            "--task", task,
            "--dataset_type", dataset_type,
            "--compress",
            "--exclude_next_obs",
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if image_path.exists():
                print(f"  ✓ Generated: {image_path}")
                success_count += 1
            else:
                print(f"  ✗ Generation failed")
                failed_tasks.append(task)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error: {e}")
            failed_tasks.append(task)

    return success_count, failed_tasks


def main():
    """Main function to download all datasets."""

    print("="*80)
    print("Batch Download All Datasets")
    print("="*80)
    print("\nThis will download:")
    print(f"  - Robomimic: {len(ROBOMIMIC_TASKS)} tasks × 2 types = {len(ROBOMIMIC_TASKS)*2} datasets")
    print(f"  - MimicGen: {len(MIMICGEN_TASKS)} tasks × 2 types = {len(MIMICGEN_TASKS)*2} datasets")
    print(f"  - Total: {(len(ROBOMIMIC_TASKS) + len(MIMICGEN_TASKS))*2} datasets")
    print("\nStarting download...")

    # Summary
    total_success = 0
    total_failed = 0
    all_failed_tasks = []

    # 1. Download Robomimic lowdim datasets
    success, failed = download_robomimic_datasets(ROBOMIMIC_TASKS, obs_type='lowdim')
    total_success += success
    total_failed += len(failed)
    all_failed_tasks.extend([(t, 'robomimic', 'lowdim') for t in failed])

    # 2. Generate Robomimic image datasets (from demo files)
    print("\n" + "="*80)
    print("Note: Robomimic image datasets need to be generated from demo files")
    print("="*80)
    success, failed = generate_robomimic_image_datasets(ROBOMIMIC_TASKS)
    total_success += success
    total_failed += len(failed)
    all_failed_tasks.extend([(t, 'robomimic', 'image') for t in failed])

    # 3. Download MimicGen lowdim datasets
    success, failed = download_mimicgen_datasets(MIMICGEN_TASKS, obs_type='lowdim')
    total_success += success
    total_failed += len(failed)
    all_failed_tasks.extend([(t, 'mimicgen', 'lowdim') for t in failed])

    # 4. Download MimicGen image datasets
    success, failed = download_mimicgen_datasets(MIMICGEN_TASKS, obs_type='image')
    total_success += success
    total_failed += len(failed)
    all_failed_tasks.extend([(t, 'mimicgen', 'image') for t in failed])

    # Final summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"✓ Successful: {total_success}")
    print(f"✗ Failed: {total_failed}")

    if all_failed_tasks:
        print("\nFailed tasks:")
        for task, source, obs_type in all_failed_tasks:
            print(f"  - {task} ({source}, {obs_type})")
        print("\nYou can retry failed tasks manually:")
        print("  python scripts/download_data.py --task <task> --obs_type <type> --source <source>")

    print("\n" + "="*80)
    print("All available tasks:")
    print("="*80)
    print("\nRobomimic tasks:")
    for task in ROBOMIMIC_TASKS:
        print(f"  python scripts/train_bc.py task={task}_lowdim")
        print(f"  python scripts/train_bc.py task={task}_image")

    print("\nMimicGen tasks:")
    for task in MIMICGEN_TASKS:
        print(f"  python scripts/train_bc.py task={task}_lowdim")
        print(f"  python scripts/train_bc.py task={task}_image")


if __name__ == "__main__":
    main()
