"""Convert DexMimicGen raw HDF5 files to lowdim/image format.

DexMimicGen HuggingFace files are `panda/{TaskName}_demo.hdf5` containing both
image and lowdim observations. This script splits them into:
  - low_dim_v141.hdf5: only dual-arm lowdim obs keys (no object state)
  - image_v141.hdf5: image + lowdim obs keys (symlink to original)

Usage:
    # Convert a single file
    python scripts/convert_dexmimicgen.py --input ~/.dexmimicgen/datasets/panda/TwoArmThreading_demo.hdf5 --task two_arm_threading

    # Convert all downloaded dexmimicgen files
    python scripts/convert_dexmimicgen.py --all
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# Dual-arm lowdim keys (robot0 + robot1, no object state)
LOWDIM_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
]

# Task name -> HuggingFace filename stem
TASK_MAP = {
    "two_arm_threading": "TwoArmThreading",
    "two_arm_three_piece_assembly": "TwoArmThreePieceAssembly",
    "two_arm_transport": "TwoArmTransport",
}

DEFAULT_BASE_DIR = Path.home() / ".dexmimicgen" / "datasets"


def convert_to_lowdim(input_path: Path, output_path: Path, force: bool = False):
    """Extract lowdim-only dataset from raw DexMimicGen HDF5."""
    if output_path.exists() and not force:
        print(f"  Already exists: {output_path}")
        return True
    if output_path.exists() and force:
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        data_in = fin["data"]
        data_out = fout.create_group("data")

        demo_keys = sorted(data_in.keys(), key=lambda k: int(k.split("_")[1]))
        print(f"  Converting {len(demo_keys)} demos to lowdim...")

        for dk in tqdm(demo_keys, desc="  lowdim", leave=False):
            demo_in = data_in[dk]
            demo_out = data_out.create_group(dk)

            # Copy actions, dones, rewards, states
            for key in ["actions", "dones", "rewards", "states"]:
                if key in demo_in:
                    demo_out.create_dataset(
                        key, data=demo_in[key][:], compression="gzip"
                    )

            # Copy only dual-arm lowdim obs
            obs_out = demo_out.create_group("obs")
            obs_in = demo_in["obs"]
            for key in LOWDIM_KEYS:
                if key in obs_in:
                    obs_out.create_dataset(
                        key, data=obs_in[key][:].astype(np.float32), compression="gzip"
                    )

        # Copy attributes (env_args is critical for env creation)
        for attr_key in data_in.attrs:
            data_out.attrs[attr_key] = data_in.attrs[attr_key]
        data_out.attrs["num_demos"] = len(demo_keys)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return True


def convert_to_image(input_path: Path, output_path: Path, force: bool = False):
    """Create image dataset - symlink to original since it already has everything."""
    if output_path.exists() and not force:
        print(f"  Already exists: {output_path}")
        return True
    if output_path.exists() and force:
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_path.symlink_to(input_path.resolve())
        print(f"  Symlinked: {output_path} -> {input_path}")
        return True
    except Exception as e:
        print(f"  Symlink failed ({e}), copying instead...")
        import shutil
        shutil.copy2(input_path, output_path)
        print(f"  Copied: {output_path}")
        return True


def convert_task(input_path: Path, task: str, base_dir: Path, force: bool = False):
    """Convert a single task's raw HDF5 to lowdim and image formats."""
    print(f"\n[{task}] Converting {input_path.name}")

    # Output paths: ~/.dexmimicgen/datasets/{task}/ph/
    task_dir = base_dir / task / "ph"

    # Lowdim
    lowdim_path = task_dir / "low_dim_v141.hdf5"
    print(f"  -> lowdim: {lowdim_path}")
    convert_to_lowdim(input_path, lowdim_path, force=force)

    # Image (symlink to original)
    image_path = task_dir / "image_v141.hdf5"
    print(f"  -> image: {image_path}")
    convert_to_image(input_path, image_path, force=force)

    return True


def find_all_raw_files(base_dir: Path):
    """Find all raw DexMimicGen HDF5 files (generated/{task}.hdf5)."""
    raw_files = {}
    gen_dir = base_dir / "generated"
    if not gen_dir.exists():
        return raw_files

    for snake_name in TASK_MAP:
        f = gen_dir / f"{snake_name}.hdf5"
        if f.exists():
            raw_files[snake_name] = f

    return raw_files


def main():
    parser = argparse.ArgumentParser(description="Convert DexMimicGen raw HDF5 to lowdim/image format")
    parser.add_argument("--input", type=str, help="Path to raw HDF5 file")
    parser.add_argument("--task", type=str, help="Task name in snake_case (e.g. two_arm_threading)")
    parser.add_argument("--all", action="store_true", help="Convert all downloaded raw files")
    parser.add_argument("--force", action="store_true", help="Force re-convert even if output exists")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR),
                        help="Base directory for dexmimicgen data")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.all:
        raw_files = find_all_raw_files(base_dir)
        if not raw_files:
            print("No raw DexMimicGen files found.")
            print(f"Expected location: {base_dir / 'panda' / '*.hdf5'}")
            return 1

        print(f"Found {len(raw_files)} raw files:")
        for task, path in raw_files.items():
            print(f"  {task}: {path}")

        success = 0
        for task, path in raw_files.items():
            if convert_task(path, task, base_dir, force=args.force):
                success += 1

        print(f"\nConverted {success}/{len(raw_files)} tasks.")
        return 0

    elif args.input:
        if not args.task:
            # Try to infer task name from filename: TwoArmThreading_demo.hdf5 -> two_arm_threading
            input_path = Path(args.input)
            stem = input_path.stem.replace("_demo", "")
            # CamelCase to snake_case
            import re
            args.task = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
            print(f"Inferred task name: {args.task}")

        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return 1

        convert_task(input_path, args.task, base_dir, force=args.force)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
