from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _BaseParameterNet(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        nmul: int,
        output_sizes: dict[str, int],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.nmul = nmul
        self.output_sizes = output_sizes
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, size) for name, size in output_sizes.items()}
        )
        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        for output_layer in self.heads.values():
            nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)
            if output_layer.bias is not None:
                nn.init.constant_(output_layer.bias, 0.0)

    @classmethod
    def build_by_config(
        cls,
        config: dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> "_BaseParameterNet":
        return cls(
            input_dim=config["nx2"],
            nmul=config["nmul"],
            hidden_dim=config["hidden_size"],
            dropout=config["dr"],
            device=device,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.backbone(x["c_nn_norm"])
        return {name: head(shared) for name, head in self.heads.items()}


class ParamRoutingNet(_BaseParameterNet):
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
            output_sizes={"params": 12 * nmul, "gamma_uh": 2},
            device=device,
        )


class LearnedStructureNet(_BaseParameterNet):
    process_order = ["w_phen", "w_int", "w_snow", "w_sub"]

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
            output_sizes={"params": 12 * nmul, "weights": 8, "gamma_uh": 2},
            device=device,
        )

    def get_structure_logits(self, attrs: torch.Tensor) -> torch.Tensor:
        """Compute 8 structure gate logits from static basin attributes."""
        with torch.no_grad():
            shared_detached = self.backbone(attrs)
        return self.heads["weights"](shared_detached)

    def structure_parameters(self):
        """Iterator over parameters belonging to the structure prediction branch."""
        if "weights" in self.heads:
            return self.heads["weights"].parameters()
        return iter(())
