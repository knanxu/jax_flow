"""PyTorch → JAX/Flax weight conversion for ImageNet-pretrained ResNet18.

Downloads torchvision ResNet18 weights and maps them to the Flax ResNet18Encoder
(which stops at layer3/256ch, no layer4/512ch).

PyTorch → Flax naming:
    conv1.weight              → Conv_0/kernel
    bn1.*                     → FrozenBatchNorm_0/*
    layer1.0.conv1/bn1        → ResidualBlock_0/Conv_0, FrozenBatchNorm_0
    layer1.0.conv2/bn2        → ResidualBlock_0/Conv_1, FrozenBatchNorm_1
    layer1.1.*                → ResidualBlock_1/*
    layer2.0.conv1/bn1        → ResidualBlock_2/Conv_0, FrozenBatchNorm_0
    layer2.0.conv2/bn2        → ResidualBlock_2/Conv_1, FrozenBatchNorm_1
    layer2.0.downsample.0/1   → ResidualBlock_2/Conv_2, FrozenBatchNorm_2
    layer2.1.*                → ResidualBlock_3/*
    layer3.0.*                → ResidualBlock_4/* (with downsample)
    layer3.1.*                → ResidualBlock_5/*
"""

import logging

import jax
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch ResNet18 pretrained URL (torchvision)
RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def _download_pytorch_weights():
    """Download PyTorch ResNet18 ImageNet weights."""
    import torch

    state_dict = torch.hub.load_state_dict_from_url(RESNET18_URL, map_location="cpu")
    # Convert all tensors to numpy
    return {k: v.detach().numpy() for k, v in state_dict.items()}


def _conv_kernel_pt_to_jax(w):
    """Convert conv kernel from PyTorch (O,I,H,W) to JAX (H,W,I,O)."""
    return np.transpose(w, (2, 3, 1, 0))


# Mapping: (pytorch_layer, block_idx) → flax ResidualBlock index
_LAYER_BLOCK_MAP = {
    (1, 0): 0,
    (1, 1): 1,
    (2, 0): 2,
    (2, 1): 3,
    (3, 0): 4,
    (3, 1): 5,
}


def _load_weights_into_single_resnet(params, batch_stats, pt_weights):
    """Load pretrained weights into a single ResNet18Encoder's params and batch_stats.

    Args:
        params: Flax params dict for one ResNet18Encoder.
        batch_stats: Flax batch_stats dict for one ResNet18Encoder.
        pt_weights: Dict of numpy arrays from PyTorch state_dict.

    Returns:
        (new_params, new_batch_stats) with pretrained values filled in.
    """
    params = jax.tree.map(lambda x: x, params)  # shallow copy
    batch_stats = jax.tree.map(lambda x: x, batch_stats)

    def _set(d, path, value):
        """Set a nested dict value by path list."""
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = np.array(value)

    # --- Stem ---
    _set(
        params, ["Conv_0", "kernel"], _conv_kernel_pt_to_jax(pt_weights["conv1.weight"])
    )
    _set(params, ["FrozenBatchNorm_0", "scale"], pt_weights["bn1.weight"])
    _set(params, ["FrozenBatchNorm_0", "bias"], pt_weights["bn1.bias"])
    _set(batch_stats, ["FrozenBatchNorm_0", "mean"], pt_weights["bn1.running_mean"])
    _set(batch_stats, ["FrozenBatchNorm_0", "var"], pt_weights["bn1.running_var"])

    # --- Residual blocks (layer1-3 only, skip layer4) ---
    for (layer_idx, block_idx), flax_block_idx in _LAYER_BLOCK_MAP.items():
        prefix = f"layer{layer_idx}.{block_idx}"
        block_name = f"ResidualBlock_{flax_block_idx}"

        # conv1 → Conv_0, bn1 → FrozenBatchNorm_0
        _set(
            params,
            [block_name, "Conv_0", "kernel"],
            _conv_kernel_pt_to_jax(pt_weights[f"{prefix}.conv1.weight"]),
        )
        _set(
            params,
            [block_name, "FrozenBatchNorm_0", "scale"],
            pt_weights[f"{prefix}.bn1.weight"],
        )
        _set(
            params,
            [block_name, "FrozenBatchNorm_0", "bias"],
            pt_weights[f"{prefix}.bn1.bias"],
        )
        _set(
            batch_stats,
            [block_name, "FrozenBatchNorm_0", "mean"],
            pt_weights[f"{prefix}.bn1.running_mean"],
        )
        _set(
            batch_stats,
            [block_name, "FrozenBatchNorm_0", "var"],
            pt_weights[f"{prefix}.bn1.running_var"],
        )

        # conv2 → Conv_1, bn2 → FrozenBatchNorm_1
        _set(
            params,
            [block_name, "Conv_1", "kernel"],
            _conv_kernel_pt_to_jax(pt_weights[f"{prefix}.conv2.weight"]),
        )
        _set(
            params,
            [block_name, "FrozenBatchNorm_1", "scale"],
            pt_weights[f"{prefix}.bn2.weight"],
        )
        _set(
            params,
            [block_name, "FrozenBatchNorm_1", "bias"],
            pt_weights[f"{prefix}.bn2.bias"],
        )
        _set(
            batch_stats,
            [block_name, "FrozenBatchNorm_1", "mean"],
            pt_weights[f"{prefix}.bn2.running_mean"],
        )
        _set(
            batch_stats,
            [block_name, "FrozenBatchNorm_1", "var"],
            pt_weights[f"{prefix}.bn2.running_var"],
        )

        # Downsample (only for first block of layer2 and layer3)
        ds_key = f"{prefix}.downsample.0.weight"
        if ds_key in pt_weights:
            _set(
                params,
                [block_name, "Conv_2", "kernel"],
                _conv_kernel_pt_to_jax(pt_weights[ds_key]),
            )
            ds_bn_prefix = f"{prefix}.downsample.1"
            _set(
                params,
                [block_name, "FrozenBatchNorm_2", "scale"],
                pt_weights[f"{ds_bn_prefix}.weight"],
            )
            _set(
                params,
                [block_name, "FrozenBatchNorm_2", "bias"],
                pt_weights[f"{ds_bn_prefix}.bias"],
            )
            _set(
                batch_stats,
                [block_name, "FrozenBatchNorm_2", "mean"],
                pt_weights[f"{ds_bn_prefix}.running_mean"],
            )
            _set(
                batch_stats,
                [block_name, "FrozenBatchNorm_2", "var"],
                pt_weights[f"{ds_bn_prefix}.running_var"],
            )

    return params, batch_stats


def load_pretrained_resnet18(encoder_params, encoder_batch_stats):
    """Load ImageNet-pretrained ResNet18 weights into encoder params.

    Handles both single ResNet18Encoder and MultiImageEncoder (which may
    contain multiple ResNet18 instances under image_encoder_{i} keys).

    SpatialSoftmax and Dense projection layers keep their random init.

    Args:
        encoder_params: Flax params dict for the encoder.
        encoder_batch_stats: Flax batch_stats dict for the encoder.

    Returns:
        (new_encoder_params, new_encoder_batch_stats)
    """
    pt_weights = _download_pytorch_weights()
    logger.info("Downloaded ImageNet-pretrained ResNet18 weights")

    # Make mutable copies
    encoder_params = jax.tree.map(lambda x: np.array(x), encoder_params)
    encoder_batch_stats = jax.tree.map(lambda x: np.array(x), encoder_batch_stats)

    # Detect structure: MultiImageEncoder has image_encoder_{i} keys,
    # or a shared encoder with ResidualBlock_* at top level
    resnet_keys = [k for k in encoder_params if k.startswith("image_encoder_")]
    if resnet_keys:
        # Multiple separate ResNet18 instances
        for key in resnet_keys:
            logger.info("Loading pretrained weights into %s", key)
            new_p, new_bs = _load_weights_into_single_resnet(
                encoder_params[key], encoder_batch_stats[key], pt_weights
            )
            encoder_params[key] = new_p
            encoder_batch_stats[key] = new_bs
    elif "ResidualBlock_0" in encoder_params:
        # Single ResNet18Encoder (shared or standalone)
        logger.info("Loading pretrained weights into single ResNet18Encoder")
        encoder_params, encoder_batch_stats = _load_weights_into_single_resnet(
            encoder_params, encoder_batch_stats, pt_weights
        )
    else:
        # MultiImageEncoder with shared encoder — ResNet is under the auto-named
        # ResNet18Encoder_0 key
        shared_keys = [k for k in encoder_params if k.startswith("ResNet18Encoder")]
        if shared_keys:
            for key in shared_keys:
                logger.info("Loading pretrained weights into %s", key)
                new_p, new_bs = _load_weights_into_single_resnet(
                    encoder_params[key], encoder_batch_stats[key], pt_weights
                )
                encoder_params[key] = new_p
                encoder_batch_stats[key] = new_bs
        else:
            logger.warning(
                "Could not find ResNet18 parameters in encoder. Keys: %s",
                list(encoder_params.keys()),
            )

    return encoder_params, encoder_batch_stats
