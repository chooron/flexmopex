from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


MOPEX_PARAM_NAMES = [
    "Sb1",
    "tw",
    "tu",
    "Se",
    "tc",
    "ddf",
    "tcrit",
    "Sb2",
    "alpha",
    "is_time",
    "tmin",
    "tmax",
]

MOPEX_PARAMS_BOUNDS = {
    "Sb1": [0.01, 50.0],
    "tw": [0.01, 5.0],
    "tu": [1.0, 2000.0],
    "Se": [1.0, 1000.0],
    "tc": [0.1, 30.0],
    "ddf": [0.0, 20.0],
    "tcrit": [-3.0, 3.0],
    "Sb2": [1.0, 1500.0],
    "alpha": [0.0, 1.0],
    "is_time": [0.0, 365.0],
    "tmin": [-10.0, 5.0],
    "tmax": [5.0, 30.0],
}

ROUTING_PARAM_NAMES = ["rout_a", "rout_b"]
ROUTING_BOUNDS = {"rout_a": [0.0, 2.9], "rout_b": [0.0, 6.5]}
WEIGHT_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
DEFAULT_FORCINGS = ["prcp", "tmean", "pet"]


class BaseMopex(nn.Module):
    mopex_param_names = MOPEX_PARAM_NAMES
    routing_param_names = ROUTING_PARAM_NAMES
    weight_names = WEIGHT_NAMES
    param_bounds = MOPEX_PARAMS_BOUNDS
    routing_bounds = ROUTING_BOUNDS

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.device = torch.device(device or self.config.get("device", "cpu"))
        self.warm_up = int(self.config.get("warm_up", 0))
        self.warm_up_states = bool(self.config.get("warm_up_states", True))
        self.variables = list(self.config.get("variables", DEFAULT_FORCINGS))
        self.nearzero = float(self.config.get("nearzero", 1e-5))
        self.nmul = int(self.config.get("nmul", 1))
        self.learnable_param_count = (
            len(self.mopex_param_names) * self.nmul + len(self.routing_param_names)
        )
        self.prcp_idx = self.variables.index("prcp")
        self.tmean_idx = self.variables.index("tmean")
        self.pet_idx = self.variables.index("pet")

    def _compile_step(self, step_fn):
        """Compile the physics step, optionally failing closed for formal runs."""
        require_compile = bool(self.config.get("require_torch_compile", False))
        if hasattr(torch, "compile") and not bool(self.config.get("disable_compile", False)):
            try:
                compiled = torch.compile(step_fn)
                self._torch_compile_enabled = True
                return compiled
            except Exception as exc:
                if require_compile:
                    raise RuntimeError("torch.compile was required but could not be enabled") from exc
        elif require_compile:
            raise RuntimeError("torch.compile was required but is unavailable or disabled")
        self._torch_compile_enabled = False
        return step_fn

    def _descale_mopex_params(
        self,
        params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized = torch.sigmoid(params).view(
            params.shape[0], len(self.mopex_param_names), self.nmul
        )
        return {
            name: change_param_range(normalized[:, index, :], self.param_bounds[name])
            for index, name in enumerate(self.mopex_param_names)
        }

    def _descale_routing_params(
        self,
        gamma_uh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized = torch.sigmoid(gamma_uh)
        return {
            name: change_param_range(normalized[:, index], self.routing_bounds[name])
            for index, name in enumerate(self.routing_param_names)
        }

    def _prepare_forcings(
        self,
        x_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        x = x_dict["x_phy"]
        n_steps, n_grid = x.shape[:2]
        P = x[:, :, self.prcp_idx].unsqueeze(-1).expand(-1, -1, self.nmul)
        T = x[:, :, self.tmean_idx].unsqueeze(-1).expand(-1, -1, self.nmul)
        PET = x[:, :, self.pet_idx].unsqueeze(-1).expand(-1, -1, self.nmul)
        doy = x_dict["doy"].expand(-1, -1, self.nmul)
        return P, T, PET, doy, n_steps, n_grid

    def _initial_states(
        self,
        n_grid: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (n_grid, self.nmul)
        S1 = torch.full(shape, self.nearzero, device=self.device)
        S2 = torch.full(shape, self.nearzero, device=self.device)
        Sc1 = torch.full(shape, self.nearzero, device=self.device)
        Sc2 = torch.full(shape, self.nearzero, device=self.device)
        Sn = torch.full(shape, self.nearzero, device=self.device)
        return S1, S2, Sc1, Sc2, Sn

    def _apply_routing(
        self,
        Qsim: torch.Tensor,
        routing_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        n_steps = Qsim.shape[0]
        rout_a = routing_params["rout_a"].unsqueeze(0).expand(n_steps, -1).unsqueeze(-1)
        rout_b = routing_params["rout_b"].unsqueeze(0).expand(n_steps, -1).unsqueeze(-1)
        UH = uh_gamma(rout_a, rout_b, lenF=15).permute([1, 2, 0])
        rf = Qsim.unsqueeze(-1).permute([1, 2, 0])
        return uh_conv(rf, UH).permute([2, 0, 1])

    def _weight_outputs(
        self,
        weights_on: torch.Tensor,
        n_train: int,
    ) -> dict[str, torch.Tensor]:
        return {
            name: weights_on[:, index].view(1, -1, 1).expand(n_train, -1, -1)
            for index, name in enumerate(self.weight_names)
        }

    def _run_weighted_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        params: dict[str, torch.Tensor],
        weights_on: torch.Tensor,
        n_steps: int,
        n_grid: int,
    ) -> torch.Tensor:
        Sb1 = params["Sb1"]
        tw = params["tw"]
        tu = params["tu"]
        Se = params["Se"]
        tc = params["tc"]
        ddf = params["ddf"]
        tcrit = params["tcrit"]
        Sb2 = params["Sb2"]
        alpha = params["alpha"]
        is_time = params["is_time"]
        tmin = params["tmin"]
        tmax = params["tmax"]
        w_phen = weights_on[:, 0].unsqueeze(-1).expand(-1, self.nmul)
        w_int = weights_on[:, 1].unsqueeze(-1).expand(-1, self.nmul)
        w_snow = weights_on[:, 2].unsqueeze(-1).expand(-1, self.nmul)
        w_sub = weights_on[:, 3].unsqueeze(-1).expand(-1, self.nmul)
        S1, S2, Sc1, Sc2, Sn = self._initial_states(n_grid)
        effective_warmup = min(self.warm_up, n_steps)

        with torch.no_grad():
            for t in range(effective_warmup):
                _, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                    P[t],
                    T[t],
                    PET[t],
                    doy[t],
                    w_phen,
                    w_int,
                    w_snow,
                    w_sub,
                    Sb1,
                    tw,
                    tu,
                    Se,
                    tc,
                    ddf,
                    tcrit,
                    Sb2,
                    alpha,
                    is_time,
                    tmin,
                    tmax,
                    S1,
                    S2,
                    Sc1,
                    Sc2,
                    Sn,
                    self.nearzero,
                )

        S1 = S1.detach()
        S2 = S2.detach()
        Sc1 = Sc1.detach()
        Sc2 = Sc2.detach()
        Sn = Sn.detach()
        Q_list = []

        for t in range(effective_warmup, n_steps):
            Q, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                P[t],
                T[t],
                PET[t],
                doy[t],
                w_phen,
                w_int,
                w_snow,
                w_sub,
                Sb1,
                tw,
                tu,
                Se,
                tc,
                ddf,
                tcrit,
                Sb2,
                alpha,
                is_time,
                tmin,
                tmax,
                S1,
                S2,
                Sc1,
                Sc2,
                Sn,
                self.nearzero,
            )
            Q_list.append(Q)

        return torch.stack(Q_list, dim=0)
