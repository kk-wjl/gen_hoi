"""Point-wise object-local geometry field used to build hand contact features."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass
class GeometryFieldOutput:
    """Predictions for arbitrary leading query-point dimensions."""

    features: torch.Tensor
    tsdf: torch.Tensor
    occupancy_logits: torch.Tensor
    normals_o: torch.Tensor


class GeometryFieldEncoder(nn.Module):
    """Encode a local query point into a compact object-geometry representation.

    This first version is intentionally conditioned only on ``q_o``: all
    training data uses the same smalltable mesh, so an object PointNet or a
    query-point identity embedding would add information that the geometry
    field should not depend on.  For every input coordinate it returns:

    - ``features``: the per-point latent ``f_i`` later concatenated into ``F``;
    - ``tsdf``: a value in ``[-1, 1]`` matching a normalized truncated SDF;
    - ``occupancy_logits``: unnormalized logits for ``sdf < 0``;
    - ``normals_o``: a unit object-local surface-normal prediction.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 8,
        hidden_dim: int = 128,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")

        self.feature_dim = feature_dim
        layers: list[nn.Module] = []
        for layer_index in range(depth):
            layers.append(nn.Linear(3 if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*layers)
        self.feature_head = nn.Linear(hidden_dim, feature_dim)
        self.tsdf_head = nn.Linear(feature_dim, 1)
        self.occupancy_head = nn.Linear(feature_dim, 1)
        self.normal_head = nn.Linear(feature_dim, 3)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, points_o: torch.Tensor) -> GeometryFieldOutput:
        """Predict geometry for a flattened encoder batch with shape ``(B, 3)``."""
        if points_o.ndim != 2 or points_o.shape[-1] != 3:
            raise ValueError(f"points_o must have shape (B, 3), got {tuple(points_o.shape)}")
        features = self.feature_head(self.trunk(points_o))
        tsdf = torch.tanh(self.tsdf_head(features)).squeeze(-1)
        occupancy_logits = self.occupancy_head(features).squeeze(-1)
        normals_o = functional.normalize(self.normal_head(features), dim=-1, eps=1e-8)
        return GeometryFieldOutput(
            features=features,
            tsdf=tsdf,
            occupancy_logits=occupancy_logits,
            normals_o=normals_o,
        )


__all__ = ["GeometryFieldEncoder", "GeometryFieldOutput"]
