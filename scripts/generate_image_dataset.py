"""Generate image datasets from robomimic demo files.

This script is a wrapper around robomimic's dataset_states_to_obs.py script,
providing a simpler interface with task-specific defaults.

Key features:
- Automatic camera configuration for each task
- Simplified command-line interface
- Progress tracking and validation

Usage:
    # Generate image dataset for a task (uses task-specific camera config)
    python scripts/generate_image_dataset.py --task square --dataset_type ph

    # Custom camera configuration
    python scripts/generate_image_dataset.py --task lift --dataset_type ph \
        --camera_names agentview robot0_eye_in_hand \
        --camera_height 84 --camera_width 84

    # Generate from custom demo file
    python scripts/generate_image_dataset.py --demo_path /path/to/demo.hdf5

    # With compression (saves disk space)
    python scripts/generate_image_dataset.py --task square --dataset_type ph --compress

Note:
    This script requires robomimic to be installed and a valid demo file.
    The demo file must contain 'states' data for each trajectory.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_flow.data import DatasetManager


# Default camera configurations for each task
# Based on robomimic's extract_obs_from_raw_datasets.sh
CAMERA_CONFIGS = {
    "lift": {
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_height": 84,
        "camera_width": 84,
    },
    "can": {
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_height": 84,
        "camera_width": 84,
    },
    "square": {
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_height": 84,
        "camera_width": 84,
    },
    "transport": {
        "camera_names": ["shouldercamera0", "shouldercamera1",
                        "robot0_eye_in_hand", "robot1_eye_in_hand"],
        "camera_height": 84,
        "camera_width": 84,
    },
    "tool_hang": {
        "camera_names": ["sideview", "robot0_eye_in_hand"],
        "camera_height": 240,
        "camera_width": 240,
    },
}


def generate_image_dataset(
    demo_path: Path,
    output_name: str = "image_v141.hdf5",
    camera_names: list | None = None,
    camera_height: int = 84,
    camera_width: int = 84,
    done_mode: int = 2,
    compress: bool = False,
    exclude_next_obs: bool = False,
) -> bool:
    """Generate image dataset from demo file using robomimic's script.

    Args:
        demo_path: Path to demo HDF5 file
        output_name: Output filename
        camera_names: List of camera names
        camera_height: Image height
        camera_width: Image width
        done_mode: Done signal mode (0/1/2)
        compress: Use gzip compression
        exclude_next_obs: Exclude next_obs (saves space for pure IL)

    Returns:
        True if successful
    """
    if not demo_path.exists():
        print(f"✗ Demo file not found: {demo_path}")
        return False

    print("=" * 80)
    print("Generating Image Dataset")
    print("=" * 80)
    print(f"Demo file: {demo_path}")
    print(f"Output: {output_name}")
    print(f"Cameras: {camera_names}")
    print(f"Resolution: {camera_height}x{camera_width}")
    print(f"Done mode: {done_mode}")
    print(f"Compress: {compress}")
    print(f"Exclude next_obs: {exclude_next_obs}")
    print("=" * 80)

    # Build command for robomimic's script
    cmd = [
        "python", "-m", "robomimic.scripts.dataset_states_to_obs",
        "--dataset", str(demo_path),
        "--output_name", output_name,
        "--done_mode", str(done_mode),
        "--camera_height", str(camera_height),
        "--camera_width", str(camera_width),
    ]

    if camera_names:
        cmd.extend(["--camera_names"] + camera_names)

    if compress:
        cmd.append("--compress")

    if exclude_next_obs:
        cmd.append("--exclude-next-obs")

    print(f"\nRunning robomimic's dataset_states_to_obs.py:")
    print(f"  {' '.join(cmd)}\n")

    try:
        # Run the command and show output in real-time
        subprocess.run(
            cmd,
            check=True,
            text=True,
        )

        # Check if output file was created
        output_path = demo_path.parent / output_name
        if output_path.exists():
            print("\n" + "=" * 80)
            print("✓ Image dataset generated successfully!")
            print("=" * 80)
            print(f"Output: {output_path}")
            print(f"Size: {output_path.stat().st_size / (1024**3):.2f} GB")
            return True
        else:
            print("\n✗ Output file not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during generation: {e}")
        print("\nPossible issues:")
        print("  1. Demo file does not contain 'states' data")
        print("  2. Camera names are incorrect for this task")
        print("  3. robomimic environment setup failed")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate image datasets from robomimic demo files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate image dataset for square task (uses default camera config)
  python scripts/generate_image_dataset.py --task square --dataset_type ph

  # Generate with compression (saves disk space)
  python scripts/generate_image_dataset.py --task square --dataset_type ph --compress

  # Generate with custom camera config
  python scripts/generate_image_dataset.py --task lift --dataset_type ph \\
      --camera_names agentview robot0_eye_in_hand \\
      --camera_height 128 --camera_width 128

  # Generate from custom demo file
  python scripts/generate_image_dataset.py --demo_path /path/to/demo.hdf5

Camera configurations by task (from robomimic):
  lift/can/square: agentview, robot0_eye_in_hand (84x84)
  transport: shouldercamera0, shouldercamera1, robot0_eye_in_hand, robot1_eye_in_hand (84x84)
  tool_hang: sideview, robot0_eye_in_hand (240x240)

Note:
  This script requires:
  - robomimic installed (pip install robomimic)
  - A demo file with 'states' data
  - Sufficient disk space (image datasets are large, ~2-3GB per task)
        """
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task name (lift, can, square, transport, tool_hang)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="ph",
        choices=["ph", "mh", "mg"],
        help="Dataset type (default: ph)"
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        help="Path to demo HDF5 file (overrides --task)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="image_v141.hdf5",
        help="Output filename (default: image_v141.hdf5)"
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        help="Camera names (default: task-specific)"
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="Image height (default: 84)"
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="Image width (default: 84)"
    )
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Done mode: 0=success, 1=end of trajectory, 2=both (default: 2)"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use gzip compression (saves disk space)"
    )
    parser.add_argument(
        "--exclude_next_obs",
        action="store_true",
        help="Exclude next_obs (saves space, only for pure imitation learning)"
    )

    args = parser.parse_args()

    # Determine demo path
    if args.demo_path:
        demo_path = Path(args.demo_path)
        task = None
    elif args.task:
        task = args.task
        # Try multiple possible demo file names (prioritize v15 for compatibility)
        demo_candidates = [
            DatasetManager.ROBOMIMIC_DIR / task / args.dataset_type / "demo_v15.hdf5",
            DatasetManager.ROBOMIMIC_DIR / task / args.dataset_type / "demo_v141.hdf5",
            DatasetManager.ROBOMIMIC_DIR / task / args.dataset_type / "demo.hdf5",
        ]

        demo_path = None
        for candidate in demo_candidates:
            if candidate.exists():
                demo_path = candidate
                break
    else:
        parser.error("Either --task or --demo_path must be specified")

    # Check if demo file exists
    if demo_path is None or not demo_path.exists():
        print("=" * 80)
        print("✗ Demo file not found")
        print("=" * 80)
        if args.demo_path:
            print(f"Specified path: {args.demo_path}")
        else:
            print(f"Searched locations:")
            for candidate in demo_candidates:
                print(f"  - {candidate}")

        print("\nTo download demo file:")
        if task:
            print(f"  python -m robomimic.scripts.download_datasets \\")
            print(f"      --tasks {task} \\")
            print(f"      --dataset_types {args.dataset_type} \\")
            print(f"      --hdf5_types raw")

        print("\nNote: Demo files are large (several GB) and may take time to download")
        return 1

    print(f"Found demo file: {demo_path}")

    # Get camera configuration
    camera_names = args.camera_names
    camera_height = args.camera_height
    camera_width = args.camera_width

    if not camera_names and task and task in CAMERA_CONFIGS:
        config = CAMERA_CONFIGS[task]
        camera_names = config["camera_names"]
        camera_height = config["camera_height"]
        camera_width = config["camera_width"]
        print(f"Using default camera config for {task}")

    if not camera_names:
        print("Warning: No camera names specified, using default")
        camera_names = ["agentview", "robot0_eye_in_hand"]

    # Generate image dataset
    success = generate_image_dataset(
        demo_path=demo_path,
        output_name=args.output_name,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        done_mode=args.done_mode,
        compress=args.compress,
        exclude_next_obs=args.exclude_next_obs,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
