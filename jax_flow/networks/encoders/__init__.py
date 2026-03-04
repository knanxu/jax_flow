"""Observation encoders for robotic manipulation.

This package provides encoders for processing raw observations into
fixed-dimensional embeddings used as conditioning for flow matching networks.
"""

from jax_flow.networks.encoders.base import (
    IdentityEncoder,
    MLPEncoder,
    create_encoder,
)
from jax_flow.networks.encoders.multi_image import MultiImageEncoder
from jax_flow.networks.encoders.resnet import ResNet18Encoder
from jax_flow.networks.encoders.spatial_softmax import SpatialSoftmax

__all__ = [
    "IdentityEncoder",
    "MLPEncoder",
    "create_encoder",
    "ResNet18Encoder",
    "MultiImageEncoder",
    "SpatialSoftmax",
]
