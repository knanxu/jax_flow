# Robomimic and MimicGen Research Summary

This document summarizes the key findings from researching robomimic and mimicgen environments and datasets.

## 1. Robomimic Environment Setup

### Environment Creation from Dataset Metadata

Robomimic uses a metadata-driven approach to create environments:

```python
# From robomimic/utils/file_utils.py
def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset HDF5 file.

    Returns:
        env_meta (dict): Contains 3 keys:
            - 'env_name': name of environment (e.g., "Lift", "Stack")
            - 'type': environment type (EB.EnvType.ROBOSUITE_TYPE, etc.)
            - 'env_kwargs': dictionary of keyword arguments for env constructor
    """
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    return env_meta

# From robomimic/utils/env_utils.py
def create_env_from_metadata(env_meta, env_name=None, render=False,
                             render_offscreen=False, use_image_obs=False,
                             use_depth_obs=False):
    """Creates environment from metadata stored in dataset."""
    env_type = env_meta["type"]
    env_kwargs = env_meta["env_kwargs"]
    env = create_env(env_type=env_type, env_name=env_name,
                     render=render, render_offscreen=render_offscreen,
                     use_image_obs=use_image_obs, use_depth_obs=use_depth_obs,
                     **env_kwargs)
    return env
```

**Key Points:**
- Environment metadata is stored in HDF5 dataset at `data.attrs["env_args"]`
- Metadata includes env name, type, and all constructor kwargs
- Environments are created lazily during training/evaluation
- Supports robosuite, gym, and other environment types

### Observation Types

Robomimic supports two main observation modalities:

**1. Low-Dimensional (State) Observations:**
- Robot proprioception: joint positions, velocities, gripper state
- Object states: positions, orientations, velocities
- End-effector pose: position and quaternion
- Relative vectors: gripper-to-object, object-to-object

**2. Image Observations:**
- RGB images: shape (H, W, 3), stored as uint8
- Depth images: shape (H, W, 1), stored as float32
- Multiple camera views: "agentview", "robot0_eye_in_hand", etc.
- Images are stored in channel-last format (H, W, C)

**Observation Processing:**
```python
# From robomimic/utils/obs_utils.py
def process_obs(obs, obs_modality=None, obs_key=None):
    """
    Process observation for network input.
    - RGB: convert uint8 to float, normalize [0,255] -> [0,1], HWC -> CHW
    - Depth: convert to float, normalize, HWC -> CHW
    - Low-dim: no processing (identity)
    """
    return OBS_MODALITY_CLASSES[obs_modality].process_obs(obs)
```

### Action Spaces

**Action Representation:**
- Actions are continuous vectors
- Stored in dataset normalized to [-1, 1] range
- Support for multiple action components (e.g., arm + gripper)
- Action keys specified in config: `config.train.action_keys`

**Action Normalization:**
```python
# From robomimic/utils/dataset.py
def action_stats_to_normalization_stats(action_stats, action_config):
    """
    Converts action statistics to normalization parameters.

    Normalization methods:
    - "min_max": normalize to [-0.999999, 0.999999]
    - "gaussian": normalize to zero mean, unit variance
    - None: no normalization (unit scale, zero offset)
    """
    # normalized_action = (raw_action - offset) / scale
    # raw_action = scale * normalized_action + offset
```

### Standard Evaluation Protocol

**Rollout Procedure:**
```python
# From robomimic/utils/train_utils.py
def run_rollout(policy, env, horizon, use_goals=False, render=False,
                video_writer=None, video_skip=5, terminate_on_success=False):
    """
    Standard rollout procedure:
    1. Reset environment
    2. Get initial observation (and goal if goal-conditioned)
    3. For each timestep:
       - Get action from policy
       - Step environment
       - Accumulate rewards
       - Check success conditions
       - Optionally render/record video
    4. Return results: return, success_rate, horizon
    """
```

**Evaluation Metrics:**
- **Success Rate**: Binary task completion (from `env.is_success()`)
- **Return**: Cumulative reward over episode
- **Horizon**: Episode length (early termination on success optional)
- **Additional Success Metrics**: Task-specific criteria (e.g., grasp, lift, stack)

**Standard Settings:**
- Horizon: 400-500 steps (task-dependent)
- Number of episodes: 50 for evaluation
- Terminate on success: Optional (speeds up evaluation)
- Video recording: Every N episodes, at 20 FPS

## 2. Robomimic Dataset Format

### HDF5 File Structure

```
dataset.hdf5
├── data/                           # Main data group
│   ├── @total                      # Total number of samples across all demos
│   ├── @env_args                   # JSON string with env metadata
│   ├── demo_0/                     # First demonstration
│   │   ├── @num_samples            # Length of this trajectory
│   │   ├── @model_file             # MJCF XML (robosuite only)
│   │   ├── states                  # MuJoCo states, shape (N, state_dim)
│   │   ├── actions                 # Actions, shape (N, action_dim)
│   │   ├── rewards                 # Rewards, shape (N,)
│   │   ├── dones                   # Done flags, shape (N,)
│   │   ├── obs/                    # Observations at timestep t
│   │   │   ├── robot0_eef_pos      # End-effector position, shape (N, 3)
│   │   │   ├── robot0_eef_quat     # End-effector quaternion, shape (N, 4)
│   │   │   ├── robot0_gripper_qpos # Gripper joint positions, shape (N, 2)
│   │   │   ├── object              # Object state, shape (N, obj_dim)
│   │   │   ├── agentview_image     # RGB image, shape (N, H, W, 3), uint8
│   │   │   └── ...                 # Other observation keys
│   │   └── next_obs/               # Observations at timestep t+1
│   │       └── ...                 # Same keys as obs/
│   ├── demo_1/
│   └── ...
└── mask/                           # Optional filter keys for dataset splits
    ├── train                       # Demo keys for training split
    ├── valid                       # Demo keys for validation split
    └── ...                         # Other filter keys
```

### Key Observations by Task Type

**Robomimic Tasks (Low-Dim):**
- `robot0_eef_pos`: End-effector position (3,)
- `robot0_eef_quat`: End-effector quaternion (4,)
- `robot0_gripper_qpos`: Gripper joint positions (2,)
- `robot0_joint_pos`: Arm joint positions (7,)
- `robot0_joint_vel`: Arm joint velocities (7,)
- `object`: Object state vector (varies by task)

**Robomimic Tasks (Image):**
- `agentview_image`: Third-person camera view (84, 84, 3) or (128, 128, 3)
- `robot0_eye_in_hand_image`: Wrist camera view (84, 84, 3)
- Low-dim observations also included

**Standard Tasks:**
- **Lift**: Single object pick-and-place
- **Can**: Pick can and place in bin
- **Square**: Peg-in-hole with square nut
- **Transport**: Two-arm coordination task
- **Tool Hang**: Hang tool on rack

### Data Loading Pattern

```python
# From robomimic/utils/dataset.py
class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for fetching sequences of experience.

    Key parameters:
    - hdf5_path: Path to HDF5 file
    - obs_keys: Observation modalities to load
    - action_keys: Action components to load
    - frame_stack: Number of frames to stack (default: 1)
    - seq_length: Sequence length for training (default: 1)
    - hdf5_cache_mode: "all", "low_dim", or None
        - "all": Cache entire dataset in memory (fastest)
        - "low_dim": Cache only non-image data
        - None: Load from disk each time (slowest)
    - hdf5_normalize_obs: Whether to normalize observations
    - filter_by_attribute: Filter key for dataset split
    """

    def __getitem__(self, index):
        """
        Returns a batch with:
        - obs: Dict of observations at time t
        - next_obs: Dict of observations at time t+1
        - actions: Action vector (concatenated if multiple components)
        - rewards: Reward scalar
        - dones: Done flag
        """
```

### Normalization

**Observation Normalization:**
```python
# Compute mean and std across entire dataset
obs_normalization_stats = {
    "obs_key": {
        "offset": mean,  # shape (1, ...)
        "scale": std,    # shape (1, ...)
    }
}
# normalized_obs = (obs - offset) / scale
```

**Action Normalization:**
```python
# Two methods: min_max or gaussian
action_normalization_stats = {
    "action_key": {
        "offset": offset,  # For min_max: input_min - scale * output_min
        "scale": scale,    # For min_max: (input_max - input_min) / (output_max - output_min)
    }
}
```

## 3. MimicGen Specifics

### Differences from Robomimic

**1. Additional Tasks:**
MimicGen adds several new manipulation tasks:
- **Stack**: Stack cube A on cube B
- **StackThree**: Stack three cubes
- **Threading**: Thread needle through tripod
- **Coffee**: Insert coffee pod into machine
- **Kitchen**: Multi-step cooking task
- **HammerCleanup**: Open drawer, grasp hammer, place in drawer
- **MugCleanup**: Open drawer, grasp mug, place in drawer
- **NutAssembly**: Assemble multiple nuts on pegs
- **PickPlace**: Pick and place multiple objects
- **ThreePieceAssembly**: Assemble three-piece object

**2. Dataset Variants:**
- **D0**: Original reset distribution (narrow initialization bounds)
- **D1**: Wider reset distribution (more diverse initial states)
- **Source**: 10-120 human demonstrations per task
- **Core**: 1000 machine-generated demonstrations per task
- **Object**: Variations with different object instances
- **Robot**: Variations with different robot arms

**3. Environment Extensions:**
```python
# From mimicgen/envs/robosuite/single_arm_env_mg.py
class SingleArmEnv_MG(SingleArmEnv):
    """
    Custom base class for MimicGen tasks.

    Key additions:
    - edit_model_xml(): Handle asset path resolution for mimicgen
    - _check_grasp_tolerant(): More robust grasp checking
    - _add_agentview_full_camera(): Full tabletop view camera
    """
```

**4. Task Configuration:**
MimicGen tasks are defined with subtask structure:
```python
# From mimicgen/configs/robosuite.py
class Stack_Config(MG_Config):
    def task_config(self):
        self.task.task_spec.subtask_1 = dict(
            object_ref="cubeA",              # Object for this subtask
            subtask_term_signal="grasp",     # Signal for subtask completion
            subtask_term_offset_range=(10, 20),  # Time offset for segmentation
            selection_strategy="nearest_neighbor_object",  # How to select source demo
            action_noise=0.05,               # Action noise during execution
            num_interpolation_steps=5,       # Steps to bridge subtasks
            num_fixed_steps=0,               # Additional settling steps
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="cubeB",
            subtask_term_signal=None,        # Final subtask has no term signal
            ...
        )
```

### Dataset Format Differences

**MimicGen datasets use the same HDF5 format as robomimic**, with additional metadata:

```python
# Additional attributes in demo groups
demo_0/
├── @datagen_info               # MimicGen-specific metadata
│   ├── grasp                   # Binary indicator for grasp subtask
│   ├── stack                   # Binary indicator for stack subtask
│   └── ...                     # Other subtask indicators
└── ...                         # Standard robomimic structure
```

**Key Differences:**
1. **Subtask Annotations**: Binary indicators for each subtask phase
2. **Object Poses**: Additional object pose information for data generation
3. **Source Demo Info**: Metadata linking generated demos to source demos
4. **Reset Distribution**: Wider initialization bounds for D1 variants

### Task-Specific Observations

**Stack Task:**
```python
# Low-dim observations
obs_keys = [
    "robot0_eef_pos",           # (3,)
    "robot0_eef_quat",          # (4,)
    "robot0_gripper_qpos",      # (2,)
    "cubeA_pos",                # (3,)
    "cubeA_quat",               # (4,)
    "cubeB_pos",                # (3,)
    "cubeB_quat",               # (4,)
    "gripper_to_cubeA",         # (3,)
    "gripper_to_cubeB",         # (3,)
    "cubeA_to_cubeB",           # (3,)
]
```

**Coffee Task:**
```python
# Low-dim observations
obs_keys = [
    "robot0_eef_pos",           # (3,)
    "robot0_eef_quat",          # (4,)
    "robot0_gripper_qpos",      # (2,)
    "coffee_pod_pos",           # (3,)
    "coffee_pod_quat",          # (4,)
    "coffee_machine_pos",       # (3,)
    "gripper_to_coffee_pod",    # (3,)
    "coffee_pod_to_machine",    # (3,)
]
```

## 4. Standard Patterns for JAX Implementation

### Environment Wrapper Pattern

```python
class RobomimicEnv:
    """JAX-compatible wrapper for robomimic environments."""

    def __init__(self, env_name: str, obs_type: str = "lowdim",
                 dataset_path: Optional[str] = None):
        """
        Args:
            env_name: Environment name (e.g., "Lift", "Stack")
            obs_type: "lowdim" or "image"
            dataset_path: Optional path to load env metadata from dataset
        """
        if dataset_path is not None:
            env_meta = get_env_metadata_from_dataset(dataset_path)
        else:
            env_meta = self._get_default_env_meta(env_name)

        self.env = create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=(obs_type == "image"),
            use_image_obs=(obs_type == "image"),
        )
        self.obs_keys = self._get_obs_keys(env_name, obs_type)

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        obs_dict = self.env.reset()
        return self._process_obs(obs_dict)

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Step environment with action."""
        obs_dict, reward, done, info = self.env.step(action)
        obs_dict = self._process_obs(obs_dict)
        return obs_dict, reward, done, info

    def _process_obs(self, obs_dict: Dict) -> Dict[str, np.ndarray]:
        """Filter and process observations."""
        return {k: obs_dict[k] for k in self.obs_keys if k in obs_dict}
```

### Dataset Loader Pattern

```python
class RobomimicDataset:
    """JAX-compatible dataset loader for robomimic HDF5 files."""

    def __init__(self, hdf5_path: str, obs_keys: List[str],
                 horizon: int = 10, obs_steps: int = 2, act_steps: int = 8):
        """
        Args:
            hdf5_path: Path to HDF5 dataset
            obs_keys: List of observation keys to load
            horizon: Action prediction horizon
            obs_steps: Number of observation frames to stack
            act_steps: Number of action steps to execute
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.horizon = horizon
        self.obs_steps = obs_steps
        self.act_steps = act_steps

        # Load dataset into memory
        self.data = self._load_dataset()
        self.normalizer = self._compute_normalizers()

    def _load_dataset(self) -> Dict:
        """Load all demonstrations into memory."""
        with h5py.File(self.hdf5_path, "r") as f:
            demos = list(f["data"].keys())
            data = {}
            for demo in demos:
                data[demo] = {
                    "obs": {k: f[f"data/{demo}/obs/{k}"][()] for k in self.obs_keys},
                    "actions": f[f"data/{demo}/actions"][()],
                    "rewards": f[f"data/{demo}/rewards"][()],
                }
        return data

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of trajectories."""
        # Sample random demos and timesteps
        # Stack observations according to obs_steps
        # Extract action sequences of length horizon
        # Apply normalization
        pass
```

### Evaluation Pattern

```python
def evaluate_policy(policy, env, num_episodes: int = 50,
                   horizon: int = 400) -> Dict[str, float]:
    """
    Standard evaluation protocol.

    Returns:
        metrics: Dict with keys:
            - "success_rate": Fraction of successful episodes
            - "return": Average cumulative reward
            - "horizon": Average episode length
    """
    successes = []
    returns = []
    horizons = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0

        for t in range(horizon):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward

            if done or info.get("is_success", False):
                successes.append(float(info["is_success"]))
                returns.append(episode_return)
                horizons.append(t + 1)
                break

    return {
        "success_rate": np.mean(successes),
        "return": np.mean(returns),
        "horizon": np.mean(horizons),
    }
```

## 5. Key Takeaways for JAX Implementation

1. **Environment Creation**: Use metadata from HDF5 datasets to create environments
2. **Observation Processing**: Handle both low-dim and image observations with proper normalization
3. **Action Normalization**: Support min-max and gaussian normalization methods
4. **Dataset Loading**: Cache data in memory for efficiency, support frame stacking and sequence sampling
5. **Evaluation Protocol**: Use standard metrics (success rate, return, horizon) with 50 episodes
6. **MimicGen Support**: Handle wider reset distributions and subtask annotations
7. **Modular Design**: Separate environment, dataset, and evaluation logic for flexibility

## 6. Recommended Implementation Order

1. **Phase 1**: Basic environment wrapper for low-dim observations
2. **Phase 2**: Dataset loader with normalization and frame stacking
3. **Phase 3**: Image observation support with proper preprocessing
4. **Phase 4**: Evaluation utilities with standard metrics
5. **Phase 5**: MimicGen-specific features (wider resets, subtask handling)
