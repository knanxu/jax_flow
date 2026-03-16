"""PyTorch → JAX/Flax weight conversion for ImageNet-pretrained ResNet18.

Downloads torchvision ResNet18 weights and maps them to the Flax ResNet18Encoder
(full layer1-4, GroupNorm). Only conv kernels and GroupNorm scale/bias are loaded;
PyTorch BN running_mean/var are discarded since we use GroupNorm.

PyTorch → Flax naming:
    conv1.weight              → Conv_0/kernel
    bn1.weight/bias           → GroupNorm_0/scale, bias
    layer1.0.conv1/bn1        → ResidualBlock_0/Conv_0, GroupNorm_0
    layer1.0.conv2/bn2        → ResidualBlock_0/Conv_1, GroupNorm_1
    layer1.1.*                → ResidualBlock_1/*
    layer2.0.conv1/bn1        → ResidualBlock_2/Conv_0, GroupNorm_0
    layer2.0.conv2/bn2        → ResidualBlock_2/Conv_1, GroupNorm_1
    layer2.0.downsample.0/1   → ResidualBlock_2/Conv_2, GroupNorm_2
    layer2.1.*                → ResidualBlock_3/*
    layer3.0.*                → ResidualBlock_4/* (with downsample)
    layer3.1.*                → ResidualBlock_5/*
    layer4.0.*                → ResidualBlock_6/* (with downsample)
    layer4.1.*                → ResidualBlock_7/*
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
    (4, 0): 6,
    (4, 1): 7,
}


def _load_weights_into_single_resnet(params, pt_weights):
    """Load pretrained weights into a single ResNet18Encoder's params.

    Only loads conv kernels and GroupNorm scale/bias. PyTorch BN running stats
    are discarded since we use GroupNorm instead of BatchNorm.

    Args:
        params: Flax params dict for one ResNet18Encoder.
        pt_weights: Dict of numpy arrays from PyTorch state_dict.

    Returns:
        new_params with pretrained conv/norm values filled in.
    """
    params = jax.tree.map(lambda x: x, params)  # shallow copy

    def _set(d, path, value):
        """Set a nested dict value by path list."""
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = np.array(value)

    # --- Stem ---
    _set(
        params, ["Conv_0", "kernel"], _conv_kernel_pt_to_jax(pt_weights["conv1.weight"])
    )
    _set(params, ["GroupNorm_0", "scale"], pt_weights["bn1.weight"])
    _set(params, ["GroupNorm_0", "bias"], pt_weights["bn1.bias"])

    # --- Residual blocks (layer1-4) ---
    for (layer_idx, block_idx), flax_block_idx in _LAYER_BLOCK_MAP.items():
        prefix = f"layer{layer_idx}.{block_idx}"
        block_name = f"ResidualBlock_{flax_block_idx}"

        # conv1 → Conv_0, bn1 → GroupNorm_0
        _set(
            params,
            [block_name, "Conv_0", "kernel"],
            _conv_kernel_pt_to_jax(pt_weights[f"{prefix}.conv1.weight"]),
        )
        _set(
            params,
            [block_name, "GroupNorm_0", "scale"],
            pt_weights[f"{prefix}.bn1.weight"],
        )
        _set(
            params,
            [block_name, "GroupNorm_0", "bias"],
            pt_weights[f"{prefix}.bn1.bias"],
        )

        # conv2 → Conv_1, bn2 → GroupNorm_1
        _set(
            params,
            [block_name, "Conv_1", "kernel"],
            _conv_kernel_pt_to_jax(pt_weights[f"{prefix}.conv2.weight"]),
        )
        _set(
            params,
            [block_name, "GroupNorm_1", "scale"],
            pt_weights[f"{prefix}.bn2.weight"],
        )
        _set(
            params,
            [block_name, "GroupNorm_1", "bias"],
            pt_weights[f"{prefix}.bn2.bias"],
        )

        # Downsample (first block of layer2, layer3, layer4)
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
                [block_name, "GroupNorm_2", "scale"],
                pt_weights[f"{ds_bn_prefix}.weight"],
            )
            _set(
                params,
                [block_name, "GroupNorm_2", "bias"],
                pt_weights[f"{ds_bn_prefix}.bias"],
            )

    return params


def load_pretrained_resnet18(encoder_params):
    """Load ImageNet-pretrained ResNet18 weights into encoder params.

    Handles both single ResNet18Encoder and MultiImageEncoder (which may
    contain multiple ResNet18 instances under image_encoder_{i} keys).

    SpatialSoftmax and Dense projection layers keep their random init.

    Args:
        encoder_params: Flax params dict for the encoder.

    Returns:
        new_encoder_params with pretrained conv/norm weights.
    """
    pt_weights = _download_pytorch_weights()
    logger.info("Downloaded ImageNet-pretrained ResNet18 weights")

    # Make mutable copies
    encoder_params = jax.tree.map(lambda x: np.array(x), encoder_params)

    # Detect structure: MultiImageEncoder has image_encoder_{i} keys,
    # or a shared encoder with ResidualBlock_* at top level
    resnet_keys = [k for k in encoder_params if k.startswith("image_encoder_")]
    if resnet_keys:
        # Multiple separate ResNet18 instances
        for key in resnet_keys:
            logger.info("Loading pretrained weights into %s", key)
            encoder_params[key] = _load_weights_into_single_resnet(
                encoder_params[key], pt_weights
            )
    elif "ResidualBlock_0" in encoder_params:
        # Single ResNet18Encoder (shared or standalone)
        logger.info("Loading pretrained weights into single ResNet18Encoder")
        encoder_params = _load_weights_into_single_resnet(encoder_params, pt_weights)
    else:
        # MultiImageEncoder with shared encoder — ResNet is under the auto-named
        # ResNet18Encoder_0 key
        shared_keys = [k for k in encoder_params if k.startswith("ResNet18Encoder")]
        if shared_keys:
            for key in shared_keys:
                logger.info("Loading pretrained weights into %s", key)
                encoder_params[key] = _load_weights_into_single_resnet(
                    encoder_params[key], pt_weights
                )
        else:
            logger.warning(
                "Could not find ResNet18 parameters in encoder. Keys: %s",
                list(encoder_params.keys()),
            )

    return encoder_params
