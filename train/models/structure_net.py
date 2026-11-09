"""Attribute-only structure encoder.

Maps the 35 standardized catchment attributes to eight gate logits (an OFF/ON
pair for each process) through a 35 -> 128 -> 128 -> 8 network. The encoder never
sees discharge or forcings, so learned gates are functions of static attributes
only.
"""
from __future__ import annotations

import torch

from train.models.parameter_nets import LearnedStructureNet


class LearnedStructureNetPureAttrEncoder(LearnedStructureNet):
    """Pure-attribute nonlinear structure encoder (35 -> 128 -> 128 -> 8)."""

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nmul=nmul,
            device=device,
        )
        self.structure_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 8),
        )
        torch.nn.init.xavier_uniform_(self.structure_encoder[0].weight)
        torch.nn.init.constant_(self.structure_encoder[0].bias, 0.0)
        torch.nn.init.xavier_uniform_(self.structure_encoder[2].weight)
        torch.nn.init.constant_(self.structure_encoder[2].bias, 0.0)
        torch.nn.init.normal_(self.structure_encoder[4].weight, mean=0.0, std=0.001)
        torch.nn.init.constant_(self.structure_encoder[4].bias, 0.0)
        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attrs = x["c_nn_norm"]
        shared = self.backbone(attrs)
        return {
            "params": self.heads["params"](shared),
            "gamma_uh": self.heads["gamma_uh"](shared),
            "weights": self.structure_encoder(attrs),
        }

    def get_structure_logits(self, attrs: torch.Tensor) -> torch.Tensor:
        """Eight structure-gate logits from normalized static attributes."""
        return self.structure_encoder(attrs)

    def structure_parameters(self):
        """Parameters belonging to the structure prediction branch."""
        return self.structure_encoder.parameters()
