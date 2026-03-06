"""Batch create task configuration files for robomimic and mimicgen.

This script generates all necessary task configuration files.
"""

import os
from pathlib import Path

# Task definitions
ROBOMIMIC_TASKS = {
    'lift': {
        'env_name': 'Lift',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'can': {
        'env_name': 'PickPlaceCan',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'square': {
        'env_name': 'NutAssemblySquare',
        'obs_dim': 23,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'transport': {
        'env_name': 'Transport',
        'obs_dim': 82,
        'act_dim': 14,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                     'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object'],
        'image_keys': ['shouldercamera0_image', 'shouldercamera1_image',
                      'robot0_eye_in_hand_image', 'robot1_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                       'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos'],
    },
    'tool_hang': {
        'env_name': 'ToolHang',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['sideview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
}

MIMICGEN_TASKS = {
    'stack': {
        'env_name': 'Stack',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'stack_three': {
        'env_name': 'StackThree',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'threading': {
        'env_name': 'Threading',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'coffee': {
        'env_name': 'Coffee',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'kitchen': {
        'env_name': 'Kitchen',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'hammer_cleanup': {
        'env_name': 'HammerCleanup',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'mug_cleanup': {
        'env_name': 'MugCleanup',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'pick_place': {
        'env_name': 'PickPlace',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
    'nut_assembly': {
        'env_name': 'NutAssembly',
        'obs_dim': 39,
        'act_dim': 7,
        'obs_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'image_keys': ['agentview_image', 'robot0_eye_in_hand_image'],
        'lowdim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    },
}


def create_lowdim_config(task_name, task_info, source='robomimic'):
    """Create lowdim configuration file."""

    if source == 'robomimic':
        path = f"data/robomimic/{task_name}/ph/low_dim_v15.hdf5"
    else:  # mimicgen
        path = f"data/mimicgen/{task_name}/ph/low_dim_v141.hdf5"

    config = f"""# {task_info['env_name']} task with low-dimensional state observations

name: {task_name}_lowdim
env_name: {task_info['env_name']}
dataset_name: {task_name}
obs_type: lowdim

# Dataset configuration
dataset:
  path: {path}
  dataset_type: ph
  horizon: 16
  obs_steps: 2
  act_steps: 8

  # Normalization
  normalize_obs: true
  normalize_act: true
  norm_type: min_max

# Environment configuration
env:
  max_episode_steps: 400

# Observation and action dimensions
obs_dim: {task_info['obs_dim']}
act_dim: {task_info['act_dim']}

# Observation keys to use
obs_keys:
"""

    for key in task_info['obs_keys']:
        config += f"  - {key}\n"

    return config


def create_image_config(task_name, task_info, source='robomimic'):
    """Create image configuration file."""

    if source == 'robomimic':
        path = f"data/robomimic/{task_name}/ph/image_v141.hdf5"
    else:  # mimicgen
        path = f"data/mimicgen/{task_name}/ph/image_v141.hdf5"

    config = f"""# {task_info['env_name']} task with image observations

name: {task_name}_image
env_name: {task_info['env_name']}
dataset_name: {task_name}
obs_type: image

# Dataset configuration
dataset:
  path: {path}
  dataset_type: ph
  horizon: 16
  obs_steps: 2
  act_steps: 8

  # Image observation keys
  image_keys: {task_info['image_keys']}

  # Low-dimensional observation keys (mixed with images)
  lowdim_keys: {task_info['lowdim_keys']}

  # Image processing
  image_size: [84, 84]
  crop_shape: [76, 76]

  # Normalization
  normalize_obs: true
  normalize_act: true
  norm_type: min_max

# Environment configuration
env:
  max_episode_steps: 400
  use_camera_obs: true

# Action dimension
act_dim: {task_info['act_dim']}

# Observation keys to use
obs_keys:
"""

    for key in task_info['image_keys']:
        config += f"  - {key}\n"
    for key in task_info['lowdim_keys']:
        config += f"  - {key}\n"

    return config


def main():
    """Generate all configuration files."""

    config_dir = Path(__file__).parent.parent / 'configs' / 'task'
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Task Configuration Files")
    print("=" * 80)

    total_created = 0

    # Generate Robomimic configs
    print("\n[Robomimic Tasks]")
    for task_name, task_info in ROBOMIMIC_TASKS.items():
        # Lowdim config
        lowdim_path = config_dir / f"{task_name}_lowdim.yaml"
        if not lowdim_path.exists():
            lowdim_config = create_lowdim_config(task_name, task_info, 'robomimic')
            lowdim_path.write_text(lowdim_config)
            print(f"  ✓ Created {lowdim_path.name}")
            total_created += 1
        else:
            print(f"  - Skipped {lowdim_path.name} (already exists)")

        # Image config
        image_path = config_dir / f"{task_name}_image.yaml"
        if not image_path.exists():
            image_config = create_image_config(task_name, task_info, 'robomimic')
            image_path.write_text(image_config)
            print(f"  ✓ Created {image_path.name}")
            total_created += 1
        else:
            print(f"  - Skipped {image_path.name} (already exists)")

    # Generate MimicGen configs
    print("\n[MimicGen Tasks]")
    for task_name, task_info in MIMICGEN_TASKS.items():
        # Lowdim config
        lowdim_path = config_dir / f"{task_name}_lowdim.yaml"
        if not lowdim_path.exists():
            lowdim_config = create_lowdim_config(task_name, task_info, 'mimicgen')
            lowdim_path.write_text(lowdim_config)
            print(f"  ✓ Created {lowdim_path.name}")
            total_created += 1
        else:
            print(f"  - Skipped {lowdim_path.name} (already exists)")

        # Image config
        image_path = config_dir / f"{task_name}_image.yaml"
        if not image_path.exists():
            image_config = create_image_config(task_name, task_info, 'mimicgen')
            image_path.write_text(image_config)
            print(f"  ✓ Created {image_path.name}")
            total_created += 1
        else:
            print(f"  - Skipped {image_path.name} (already exists)")

    print("\n" + "=" * 80)
    print(f"Summary: Created {total_created} new configuration files")
    print("=" * 80)

    # Print next steps
    print("\nNext steps:")
    print("1. Download Robomimic lowdim datasets:")
    print("   python scripts/download_data.py --task <task> --obs_type lowdim")
    print("\n2. Generate Robomimic image datasets:")
    print("   python scripts/generate_image_dataset.py --task <task> --dataset_type ph --compress")
    print("\n3. Download MimicGen datasets:")
    print("   python scripts/download_data.py --task <task> --obs_type lowdim --source mimicgen")
    print("   python scripts/download_data.py --task <task> --obs_type image --source mimicgen")


if __name__ == "__main__":
    main()
