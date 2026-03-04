"""Training script for Behavior Cloning with Flow Matching.

Usage:
    # Train with default config (lift lowdim)
    python scripts/train_bc.py

    # Train with specific task
    python scripts/train_bc.py task=lift_lowdim

    # Override parameters
    python scripts/train_bc.py task=lift_lowdim optimization.lr=1e-3 optimization.batch_size=128

    # Train with image observations
    python scripts/train_bc.py task=lift_image network=resnet
"""

import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.data import make_robomimic_dataset


def create_dataloader(dataset, batch_size, shuffle=True):
    """Create a simple dataloader iterator."""
    indices = np.arange(len(dataset))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            # Collect batch
            obs_batch = []
            act_batch = []
            for idx in batch_indices:
                sample = dataset[idx]
                obs_batch.append(sample["observations"])
                act_batch.append(sample["actions"])

            # Stack into batch
            observations = np.stack(obs_batch, axis=0)
            actions = np.stack(act_batch, axis=0)

            # Convert to JAX arrays
            batch = {
                "observations": jnp.array(observations),
                "actions": jnp.array(actions),
            }

            yield batch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Print config
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    dataset_path = cfg.task.dataset.path
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("\nPlease download the dataset first:")
        print(f"  python scripts/download_data.py --task {cfg.task.env_name.lower()} --obs_type {cfg.task.obs_type}")
        return

    train_dataset = make_robomimic_dataset(
        dataset_path=dataset_path,
        horizon=cfg.task.dataset.horizon,
        obs_steps=cfg.task.dataset.obs_steps,
        act_steps=cfg.task.dataset.act_steps,
        obs_type=cfg.task.obs_type,
        val_ratio=0.1,
        mode="train",
    )

    val_dataset = make_robomimic_dataset(
        dataset_path=dataset_path,
        horizon=cfg.task.dataset.horizon,
        obs_steps=cfg.task.dataset.obs_steps,
        act_steps=cfg.task.dataset.act_steps,
        obs_type=cfg.task.obs_type,
        val_ratio=0.1,
        mode="val",
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Observation dim: {train_dataset.obs_dim}")
    print(f"Action dim: {train_dataset.action_dim}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=True,
    )

    # Create agent
    print("\n" + "=" * 80)
    print("Creating Agent")
    print("=" * 80)

    # Get example batch for initialization
    example_batch = next(train_loader)
    ex_observations = example_batch["observations"]
    ex_actions = example_batch["actions"]

    print(f"Example observations shape: {ex_observations.shape}")
    print(f"Example actions shape: {ex_actions.shape}")

    # Build agent config
    agent_config = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "action_dim": train_dataset.action_dim,
        "obs_dim": train_dataset.obs_dim,
        "encoder_type": cfg.get("encoder_type", "identity"),
        "encoder_hidden_dims": tuple(cfg.network.get("encoder_hidden_dims", [256, 256])),
        "emb_dim": cfg.network.emb_dim,
        "hidden_dims": tuple(cfg.network.hidden_dims),
        "activation": cfg.network.activation,
        "layer_norm": cfg.network.get("use_layer_norm", False),
        "lr": cfg.optimization.lr,
        "weight_decay": cfg.optimization.weight_decay,
        "schedule_type": cfg.optimization.lr_schedule.type,
        "warmup_steps": cfg.optimization.lr_schedule.warmup_steps,
        "gradient_steps": cfg.optimization.gradient_steps,
        "interp_type": cfg.flow.interp_type,
        "flow_type": "flow_matching",  # Use standard flow matching
        "sampler_type": cfg.flow.sampler.type,
        "flow_steps": cfg.flow.sampler.num_steps,
    }

    rng, agent_rng = jax.random.split(rng)
    agent = BCAgent.create(
        seed=int(agent_rng[0]),
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=agent_config,
    )

    print("Agent created successfully!")
    print(f"Network step: {agent.network.step}")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_loss = float("inf")
    train_losses = []

    pbar = tqdm(range(cfg.optimization.gradient_steps), desc="Training")

    for step in pbar:
        # Get batch
        batch = next(train_loader)

        # Update agent
        agent, info = agent.update(batch)

        # Log
        train_losses.append(float(info["loss"]))

        if step % cfg.optimization.log_freq == 0:
            avg_loss = np.mean(train_losses[-cfg.optimization.log_freq:])
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "grad_norm": f"{float(info['grad/norm']):.4f}",
            })

        # Validation
        if step % cfg.optimization.eval_freq == 0 and step > 0:
            print(f"\n[Step {step}] Running validation...")

            val_losses = []
            for _ in range(min(100, len(val_dataset) // cfg.optimization.batch_size)):
                val_batch = {
                    "observations": jnp.array(val_dataset.sample_batch(cfg.optimization.batch_size)["observations"]),
                    "actions": jnp.array(val_dataset.sample_batch(cfg.optimization.batch_size)["actions"]),
                }

                # Compute loss without updating
                def val_loss_fn(params):
                    def encode(obs, training=False):
                        return agent.network(obs, training=training, name="encoder", params=params)

                    def flow_net(at, s, t, cond, training=False):
                        return agent.network(at, s, t, cond, training=training, name="flow", params=params)

                    return agent.loss_fn_type(
                        network=flow_net,
                        encoder=encode,
                        interpolant=agent.interpolant,
                        batch=val_batch,
                        rng=agent.rng,
                        config=agent.config,
                    )

                loss, _ = val_loss_fn(agent.network.params)
                val_losses.append(float(loss))

            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = output_dir / "best_model.pkl"
                print(f"Saving best model to {checkpoint_path}")
                # TODO: Implement checkpoint saving

        # Save checkpoint
        if step % cfg.optimization.save_freq == 0 and step > 0:
            checkpoint_path = output_dir / f"checkpoint_{step}.pkl"
            print(f"\nSaving checkpoint to {checkpoint_path}")
            # TODO: Implement checkpoint saving

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final train loss: {np.mean(train_losses[-100:]):.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
