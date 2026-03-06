"""Dataset manager for automatic download and path resolution.

Handles:
- Path resolution (relative paths, ~/.robomimic/, etc.)
- Automatic dataset download if missing
- Version compatibility (v15 vs v141 for image datasets)
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple


class DatasetManager:
    """Manages dataset paths and automatic downloads."""

    # Default robomimic dataset directory
    ROBOMIMIC_DIR = Path.home() / ".robomimic"

    # Dataset version mappings
    VERSION_MAP = {
        "low_dim": "low_dim_v15.hdf5",
        "image": "image_v141.hdf5",
    }

    @staticmethod
    def resolve_dataset_path(
        dataset_path: str,
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
    ) -> Path:
        """Resolve dataset path, checking multiple locations.

        Priority:
        1. Absolute path if exists
        2. Relative to project root if exists
        3. ~/.robomimic/ directory
        4. Return expected path for download

        Args:
            dataset_path: Path from config file
            task: Task name (lift, square, etc.)
            dataset_type: Dataset type (ph, mh, mg)
            obs_type: Observation type (lowdim, image)

        Returns:
            Resolved Path object
        """
        # Try as absolute path
        path = Path(dataset_path).expanduser()
        if path.exists():
            return path

        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        rel_path = project_root / dataset_path
        if rel_path.exists():
            return rel_path

        # Try ~/.robomimic/ directory
        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"
        robomimic_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v15.hdf5"
        if robomimic_path.exists():
            return robomimic_path

        # Try v141 for image datasets
        if obs_type == "image":
            robomimic_path_v141 = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / "image_v141.hdf5"
            if robomimic_path_v141.exists():
                return robomimic_path_v141

        # Return expected path (may not exist yet)
        return path

    @staticmethod
    def check_dataset_exists(
        dataset_path: str,
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
    ) -> Tuple[bool, Optional[Path]]:
        """Check if dataset exists.

        Returns:
            (exists, resolved_path)
        """
        resolved_path = DatasetManager.resolve_dataset_path(
            dataset_path, task, dataset_type, obs_type
        )
        return resolved_path.exists(), resolved_path

    @staticmethod
    def download_dataset(
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
        source: str = "robomimic",
        force: bool = False,
    ) -> bool:
        """Download dataset using robomimic or mimicgen.

        Args:
            task: Task name (lift, square, etc.)
            dataset_type: Dataset type (ph, mh, mg)
            obs_type: Observation type (lowdim, image)
            source: Data source (robomimic, mimicgen)
            force: Force re-download even if exists

        Returns:
            True if successful, False otherwise
        """
        # Check if already exists
        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"
        expected_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v15.hdf5"

        if expected_path.exists() and not force:
            print(f"Dataset already exists at: {expected_path}")
            return True

        print(f"\n{'='*80}")
        print(f"Downloading {source} dataset")
        print(f"{'='*80}")
        print(f"Task: {task}")
        print(f"Dataset type: {dataset_type}")
        print(f"Observation type: {obs_type}")
        print(f"Target: {expected_path}")
        print(f"{'='*80}\n")

        if source == "robomimic":
            return DatasetManager._download_robomimic(task, dataset_type, obs_type)
        elif source == "mimicgen":
            return DatasetManager._download_mimicgen(task, dataset_type, obs_type)
        else:
            print(f"Error: Unknown source '{source}'")
            return False

    @staticmethod
    def _download_robomimic(task: str, dataset_type: str, obs_type: str) -> bool:
        """Download using robomimic's download script."""
        try:
            import robomimic  # noqa: F401
        except ImportError:
            print("Error: robomimic not installed. Install with:")
            print("  pip install robomimic")
            return False

        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"

        # Special handling for image datasets
        if obs_type == "image":
            print("\n" + "=" * 80)
            print("Robomimic Image Dataset Generation Required")
            print("=" * 80)
            print("\nRobomimic image datasets cannot be directly downloaded.")
            print("You have two options:\n")

            print("Option 1: Generate from demo file (Recommended for robomimic tasks)")
            print("-" * 80)
            print("Step 1: Download demo file")
            print(f"  python scripts/download_data.py --task {task} --dataset_type {dataset_type} --obs_type demo")
            print("\nStep 2: Generate image dataset")
            print(f"  python scripts/generate_image_dataset.py --task {task} --dataset_type {dataset_type}")

            print("\nOption 2: Use MimicGen (if task is available)")
            print("-" * 80)
            print(f"  python scripts/download_data.py --task {task} --obs_type image --source mimicgen")
            print("\nMimicGen tasks: stack, square, threading, coffee, kitchen, etc.")
            print("=" * 80)
            return False

        # Use robomimic CLI for lowdim datasets
        cmd = [
            "python", "-m", "robomimic.scripts.download_datasets",
            "--tasks", task,
            "--dataset_types", dataset_type,
            "--hdf5_types", hdf5_type,
        ]

        print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            print("\n✓ Download complete!")

            # Verify download
            expected_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v15.hdf5"
            if expected_path.exists():
                print(f"✓ Dataset verified at: {expected_path}")
                return True
            else:
                print(f"✗ Dataset not found at expected path: {expected_path}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"Error during download: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    @staticmethod
    def _download_mimicgen(task: str, dataset_type: str, obs_type: str) -> bool:
        """Download from HuggingFace mimicgen datasets."""
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError:
            print("Error: huggingface_hub not installed. Install with:")
            print("  pip install huggingface_hub")
            return False

        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"

        # MimicGen repo structure: core/{task}/{dataset_type}/{hdf5_type}_v141.hdf5
        filename = f"core/{task}/{dataset_type}/{hdf5_type}_v141.hdf5"

        print(f"Downloading mimicgen dataset from HuggingFace:")
        print(f"  Filename: {filename}")

        try:
            import os

            # Check proxy settings
            if os.environ.get('http_proxy') or os.environ.get('https_proxy'):
                print(f"  Using proxy from environment variables")

            target_dir = DatasetManager.ROBOMIMIC_DIR / "mimicgen"
            file_path = hf_hub_download(
                repo_id="amandlek/mimicgen_datasets",
                filename=filename,
                repo_type="dataset",
                local_dir=str(target_dir),
            )
            print(f"\n✓ Download complete!")
            print(f"✓ Dataset saved to: {file_path}")

            # Create symlink to standard location for easier access
            standard_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v141.hdf5"
            standard_path.parent.mkdir(parents=True, exist_ok=True)

            if not standard_path.exists():
                try:
                    os.symlink(file_path, standard_path)
                    print(f"✓ Created symlink: {standard_path}")
                except Exception as e:
                    print(f"Note: Could not create symlink: {e}")

            return True

        except Exception as e:
            print(f"Error during download: {e}")
            print("\nAvailable MimicGen tasks:")
            print("  stack, stack_three, threading, coffee, kitchen,")
            print("  hammer_cleanup, mug_cleanup, pick_place, square, nut_assembly")
            print("\nNote: Make sure the task name matches exactly")
            print("\nIf you're behind a proxy, set environment variables:")
            print("  export http_proxy=http://127.0.0.1:7897")
            print("  export https_proxy=http://127.0.0.1:7897")
            return False

    @staticmethod
    def ensure_dataset(
        dataset_path: str,
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
        source: str = "robomimic",
        auto_download: bool = True,
    ) -> Optional[Path]:
        """Ensure dataset exists, download if missing.

        Args:
            dataset_path: Path from config
            task: Task name
            dataset_type: Dataset type
            obs_type: Observation type
            source: Data source
            auto_download: Whether to auto-download if missing

        Returns:
            Resolved path if exists/downloaded, None if failed
        """
        exists, resolved_path = DatasetManager.check_dataset_exists(
            dataset_path, task, dataset_type, obs_type
        )

        if exists:
            print(f"✓ Dataset found at: {resolved_path}")
            return resolved_path

        if not auto_download:
            print(f"✗ Dataset not found at: {resolved_path}")
            print(f"\nTo download manually, run:")
            print(f"  python scripts/download_data.py --task {task} --obs_type {obs_type}")
            return None

        # Auto-download
        print(f"✗ Dataset not found, attempting automatic download...")
        success = DatasetManager.download_dataset(
            task=task,
            dataset_type=dataset_type,
            obs_type=obs_type,
            source=source,
        )

        if success:
            # Re-resolve path after download
            exists, resolved_path = DatasetManager.check_dataset_exists(
                dataset_path, task, dataset_type, obs_type
            )
            if exists:
                return resolved_path

        print(f"\n✗ Failed to download dataset")
        print(f"\nPlease download manually:")
        print(f"  python scripts/download_data.py --task {task} --obs_type {obs_type}")
        return None
