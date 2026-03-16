"""Dataset manager for automatic download and path resolution.

Handles:
- Path resolution (relative paths, ~/.robomimic/, etc.)
- Automatic dataset download if missing
- Version compatibility (v15 vs v141 for image datasets)
"""

import subprocess
from pathlib import Path


class DatasetManager:
    """Manages dataset paths and automatic downloads."""

    # Default robomimic dataset directory
    ROBOMIMIC_DIR = Path.home() / ".robomimic"

    # Dataset version mappings
    VERSION_MAP = {
        "low_dim": "low_dim_v15.hdf5",
        "image": "image_v141.hdf5",
    }

    # MimicGen tasks (stored under ~/.robomimic/mimicgen/core/)
    MIMICGEN_TASKS = {
        "stack", "stack_three", "threading", "coffee", "kitchen",
        "hammer_cleanup", "mug_cleanup", "pick_place", "nut_assembly",
    }

    # DexMimicGen tasks (stored under ~/.dexmimicgen/datasets/)
    DEXMIMICGEN_TASKS = {
        "two_arm_threading", "two_arm_three_piece_assembly", "two_arm_transport",
    }

    # DexMimicGen task name -> HuggingFace filename mapping
    DEXMIMICGEN_HF_MAP = {
        "two_arm_threading": "generated/two_arm_threading.hdf5",
        "two_arm_three_piece_assembly": "generated/two_arm_three_piece_assembly.hdf5",
        "two_arm_transport": "generated/two_arm_transport.hdf5",
    }

    # DexMimicGen base directory
    DEXMIMICGEN_DIR = Path.home() / ".dexmimicgen" / "datasets"

    # Push-T tasks (downloaded from HuggingFace as zarr)
    PUSHT_TASKS = {"pusht"}

    # Kitchen tasks (downloaded from HuggingFace as zip of MJL logs)
    KITCHEN_TASKS = {"kitchen"}

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
        3. ~/.robomimic/mimicgen/core/{task}/ph/ (MimicGen tasks)
        4. ~/.robomimic/{task}/ph/ (Robomimic tasks, v15 then v141)
        5. Return expected path for download

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

        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"

        # Try DexMimicGen path: ~/.dexmimicgen/datasets/{task}/ph/
        if task in DatasetManager.DEXMIMICGEN_TASKS:
            dex_path = DatasetManager.DEXMIMICGEN_DIR / task / dataset_type / f"{hdf5_type}_v141.hdf5"
            if dex_path.exists():
                return dex_path

        # Try MimicGen path: ~/.robomimic/mimicgen/core/{task}/{dataset_type}/
        mimicgen_path = DatasetManager.ROBOMIMIC_DIR / "mimicgen" / "core" / task / dataset_type / f"{hdf5_type}_v141.hdf5"
        if mimicgen_path.exists():
            return mimicgen_path

        # Try Robomimic path: ~/.robomimic/{task}/{dataset_type}/
        robomimic_path = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v15.hdf5"
        if robomimic_path.exists():
            return robomimic_path

        # Try v141 variant
        robomimic_v141 = DatasetManager.ROBOMIMIC_DIR / task / dataset_type / f"{hdf5_type}_v141.hdf5"
        if robomimic_v141.exists():
            return robomimic_v141

        # Return expected path (may not exist yet)
        return path

    @staticmethod
    def check_dataset_exists(
        dataset_path: str,
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
    ) -> tuple[bool, Path | None]:
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
        # Check if already exists via resolve
        exists, resolved = DatasetManager.check_dataset_exists(
            dataset_path="", task=task, dataset_type=dataset_type, obs_type=obs_type
        )
        if exists and not force:
            print(f"Dataset already exists at: {resolved}")
            return True

        print(f"\n{'='*80}")
        print(f"Downloading {source} dataset")
        print(f"{'='*80}")
        print(f"Task: {task}")
        print(f"Dataset type: {dataset_type}")
        print(f"Observation type: {obs_type}")
        print(f"{'='*80}\n")

        if source == "robomimic":
            return DatasetManager._download_robomimic(task, dataset_type, obs_type)
        elif source == "mimicgen":
            return DatasetManager._download_mimicgen(task, dataset_type, obs_type)
        elif source == "dexmimicgen":
            return DatasetManager._download_dexmimicgen(task, dataset_type, obs_type)
        elif source == "pusht":
            return DatasetManager._download_pusht(task, obs_type)
        elif source == "kitchen":
            return DatasetManager._download_kitchen(task)
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
        """Download from HuggingFace mimicgen datasets.

        MimicGen files on HuggingFace are `core/{task}_d0.hdf5` containing both
        image and lowdim data. We download the raw file, then convert to separate
        lowdim/image formats using convert_mimicgen.py.
        """
        import os
        import subprocess

        base_dir = DatasetManager.ROBOMIMIC_DIR / "mimicgen"
        raw_path = base_dir / "core" / f"{task}_d0.hdf5"

        # Step 1: Download raw file if not present
        if not raw_path.exists():
            print(f"Downloading {task}_d0.hdf5 from HuggingFace...")

            # Use wget (more reliable than hf_hub_download with proxies)
            url = f"https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/{task}_d0.hdf5"
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            # Unset ALL_PROXY to avoid socks proxy issues with some tools
            env = os.environ.copy()
            env.pop("ALL_PROXY", None)
            env.pop("all_proxy", None)

            try:
                result = subprocess.run(
                    ["wget", "-c", "-q", "--show-progress", url, "-O", str(raw_path)],
                    env=env,
                    check=True,
                    timeout=3600,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"wget failed ({e}), trying huggingface_hub...")
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore
                    hf_hub_download(
                        repo_id="amandlek/mimicgen_datasets",
                        filename=f"core/{task}_d0.hdf5",
                        repo_type="dataset",
                        local_dir=str(base_dir),
                    )
                except Exception as e2:
                    print(f"Error: {e2}")
                    if raw_path.exists():
                        raw_path.unlink()
                    return False

        if not raw_path.exists():
            print(f"Error: raw file not found at {raw_path}")
            return False

        print(f"Raw file: {raw_path} ({raw_path.stat().st_size / 1024**3:.1f} GB)")

        # Step 2: Convert to lowdim/image format
        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"
        target_path = base_dir / "core" / task / dataset_type / f"{hdf5_type}_v141.hdf5"

        if target_path.exists():
            print(f"Converted file already exists: {target_path}")
            return True

        print(f"Converting to {obs_type} format...")
        try:
            convert_script = Path(__file__).parent.parent.parent / "scripts" / "convert_mimicgen.py"
            result = subprocess.run(
                ["python", str(convert_script), "--input", str(raw_path), "--task", task],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e.stderr}")
            return False

        return target_path.exists()

    @staticmethod
    def _download_dexmimicgen(task: str, dataset_type: str, obs_type: str) -> bool:
        """Download from HuggingFace DexMimicGen datasets and convert.

        DexMimicGen files on HuggingFace (MimicGen/dexmimicgen_datasets) are
        `panda/{CamelCaseName}_demo.hdf5`. We download the raw file, then convert
        to lowdim format using convert_dexmimicgen.py.
        """
        import os

        if task not in DatasetManager.DEXMIMICGEN_HF_MAP:
            print(f"Error: Unknown DexMimicGen task '{task}'")
            print(f"Available: {sorted(DatasetManager.DEXMIMICGEN_TASKS)}")
            return False

        hf_filename = DatasetManager.DEXMIMICGEN_HF_MAP[task]
        raw_path = DatasetManager.DEXMIMICGEN_DIR / hf_filename

        # Step 1: Download raw file if not present
        if not raw_path.exists():
            print(f"Downloading {hf_filename} from HuggingFace...")
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            url = f"https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/resolve/main/{hf_filename}"

            env = os.environ.copy()
            env.pop("ALL_PROXY", None)
            env.pop("all_proxy", None)

            try:
                subprocess.run(
                    ["wget", "-c", "-q", "--show-progress", url, "-O", str(raw_path)],
                    env=env,
                    check=True,
                    timeout=3600,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"wget failed ({e}), trying huggingface_hub...")
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore
                    hf_hub_download(
                        repo_id="MimicGen/dexmimicgen_datasets",
                        filename=hf_filename,
                        repo_type="dataset",
                        local_dir=str(DatasetManager.DEXMIMICGEN_DIR),
                    )
                except Exception as e2:
                    print(f"Error: {e2}")
                    if raw_path.exists():
                        raw_path.unlink()
                    return False

        if not raw_path.exists():
            print(f"Error: raw file not found at {raw_path}")
            return False

        print(f"Raw file: {raw_path} ({raw_path.stat().st_size / 1024**3:.1f} GB)")

        # Step 2: Convert to lowdim/image format
        hdf5_type = "low_dim" if obs_type == "lowdim" else "image"
        target_path = DatasetManager.DEXMIMICGEN_DIR / task / dataset_type / f"{hdf5_type}_v141.hdf5"

        if target_path.exists():
            print(f"Converted file already exists: {target_path}")
            return True

        print(f"Converting to {obs_type} format...")
        try:
            convert_script = Path(__file__).parent.parent.parent / "scripts" / "convert_dexmimicgen.py"
            subprocess.run(
                ["python", str(convert_script), "--input", str(raw_path), "--task", task],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e.stderr}")
            return False

        return target_path.exists()

    @staticmethod
    def _download_pusht(task: str, obs_type: str) -> str | None:
        """Download Push-T dataset from HuggingFace. Returns path."""
        try:
            from jax_flow.data.pusht_dataset import download_pusht_dataset
            return download_pusht_dataset()
        except Exception as e:
            print(f"Error downloading Push-T dataset: {e}")
            return None

    @staticmethod
    def _download_kitchen(task: str) -> str | None:
        """Download Kitchen dataset from HuggingFace. Returns path."""
        try:
            from jax_flow.data.kitchen_dataset import download_kitchen_dataset
            return download_kitchen_dataset()
        except Exception as e:
            print(f"Error downloading Kitchen dataset: {e}")
            return None

    @staticmethod
    def ensure_dataset(
        dataset_path: str,
        task: str,
        dataset_type: str = "ph",
        obs_type: str = "lowdim",
        source: str = "robomimic",
        auto_download: bool = True,
    ) -> Path | None:
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
        # Push-T / Kitchen: skip robomimic path resolution, use own download logic
        if source in ("pusht", "kitchen"):
            # Try dataset_path as-is first
            path = Path(dataset_path).expanduser()
            if path.exists():
                print(f"✓ Dataset found at: {path}")
                return path
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            rel_path = project_root / dataset_path
            if rel_path.exists():
                print(f"✓ Dataset found at: {rel_path}")
                return rel_path

            if not auto_download:
                print(f"✗ Dataset not found at: {path}")
                return None

            print(f"✗ Dataset not found, attempting automatic download ({source})...")
            if source == "pusht":
                result_path = DatasetManager._download_pusht(task, obs_type)
            else:
                result_path = DatasetManager._download_kitchen(task)
            if result_path and Path(result_path).exists():
                print(f"✓ Dataset downloaded to: {result_path}")
                return Path(result_path)
            print(f"\n✗ Failed to download {source} dataset")
            print(f"  pip install jax_flow[{source}]")
            return None

        # Robomimic / MimicGen / DexMimicGen path resolution
        exists, resolved_path = DatasetManager.check_dataset_exists(
            dataset_path, task, dataset_type, obs_type
        )

        if exists:
            print(f"✓ Dataset found at: {resolved_path}")
            return resolved_path

        if not auto_download:
            print(f"✗ Dataset not found at: {resolved_path}")
            print("\nTo download manually, run:")
            print(f"  python scripts/download_data.py --task {task} --obs_type {obs_type}")
            return None

        # Auto-detect source from task name
        if source == "robomimic" and task in DatasetManager.MIMICGEN_TASKS:
            source = "mimicgen"
        elif source == "robomimic" and task in DatasetManager.DEXMIMICGEN_TASKS:
            source = "dexmimicgen"

        # Auto-download (robomimic / mimicgen / dexmimicgen)
        print(f"✗ Dataset not found, attempting automatic download ({source})...")
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

        print("\n✗ Failed to download dataset")
        print("\nPlease download manually:")
        print(f"  python scripts/download_data.py --task {task} --obs_type {obs_type}")
        return None
