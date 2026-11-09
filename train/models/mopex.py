"""Daily MOPEX water-balance simulator used by Flex-MOPEX.

Precipitation is partitioned into liquid and solid before interception, and only
the liquid fraction is intercepted:

    f_r      = sigmoid(T - t_crit)
    P_r      = P * f_r
    P_s      = P - P_r
    E_i      = w_int * min(P_r*season, P_r, PET_eff)
    P_r_net  = P_r - E_i
    PET_soil = max(PET_eff - E_i, 0)
    Q_N      = min(sigmoid(T-t_crit)*softplus(T-t_crit)*ddf, S_n)
    P_eff    = P_r_net + P_s*(1 - w_snow) + Q_N

No canopy snow storage, sublimation or drip is represented: interception is a
direct liquid flux loss, not a storage reservoir.

Invariants covered by ``tests/test_mopex.py``: with ``w_int = 0`` the step
matches the total-precipitation interception formulation algebraically; water
conservation ``dS1+dS2+dSc1+dSc2+dSn + Q + ET == P`` holds per timestep; the
``w_snow`` endpoints are unchanged.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from train.models.base_mopex import BaseMopex


def _phase_rad(doy: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.pi * (doy - torch.clamp(phi, 0.0, 365.0)) / 365.25


def season_linear(doy: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + kappa * torch.cos(_phase_rad(doy, phi)))


def mopex_step(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural weights ---
    w_phen: torch.Tensor,
    w_int: torch.Tensor,
    w_snow: torch.Tensor,
    w_sub: torch.Tensor,
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    kappa: torch.Tensor,
    phi: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    # 0. guards
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # 1. phenology (unchanged)
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)
    PET_effective = w_phen * (PET * flux_gsi) + (1.0 - w_phen) * PET

    # 2. rain / snow partition on raw P, BEFORE interception
    f_r = torch.sigmoid(T - tcrit)
    P_r = P * f_r
    P_s = P - P_r

    # 3. interception on the liquid fraction only (seasonal gate + shared PET cap)
    season = season_linear(doy, kappa, phi)
    flux_potential = P_r * season
    flux_i_pot = torch.minimum(flux_potential, torch.minimum(P_r, PET_effective))
    flux_i = flux_i_pot * w_int
    P_r_net = P_r - flux_i
    pet_for_soil = F.relu(PET_effective - flux_i)          # S0 shared PET budget

    # 4. snow store and melt
    P_to_snow = P_s * w_snow
    P_bypass = P_r_net + P_s * (1.0 - w_snow)
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn = torch.minimum(melt_drive * ddf, Sn)
    Sn_new = Sn + P_to_snow - flux_qn
    P_eff = P_bypass + flux_qn

    # 5. surface soil S1 (unchanged)
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f
    ratio_s1 = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1 = torch.minimum(pet_for_soil * ratio_s1, S1)
    S1 = S1 - flux_et1
    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new = S1 - flux_qw

    # 6. subsurface S2 (unchanged)
    S2 = S2 + flux_qw
    flux_q2f_pot = F.relu(S2 - Sb2)
    flux_q2f = flux_q2f_pot * w_sub
    S2 = S2 - flux_q2f
    flux_q2u = S2 * (1.0 - torch.exp(-1.0 / (tu + nearzero)))
    S2 = S2 - flux_q2u
    remaining_pet = F.relu(pet_for_soil - flux_et1)
    ratio_s2 = torch.clamp(S2 / (Se + nearzero), max=1.0)
    flux_et2 = torch.minimum(remaining_pet * ratio_s2, S2)
    S2_new = S2 - flux_et2

    # 7. routing (unchanged)
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new


class Mopex(BaseMopex):
    """MOPEX with learned structural gates (the Flex-MOPEX model).

    Each catchment gets one ON probability per process from the structure
    encoder. Training samples the gates with a Gumbel-softmax switch; evaluation
    uses the deterministic softmax probability, so reported gates are posterior
    probabilities rather than sampled decisions.
    """

    def __init__(self, config: dict[str, Any] | None = None,
                 device: str | torch.device | None = None) -> None:
        super().__init__(config, device)
        self.name = "Mopex"
        self.structure_tau = float(self.config.get("structure_tau", 1.0))
        self.learnable_param_count += len(self.weight_names) * 2
        semantics = str(self.config.get("interception_semantics", "RF_S0")).upper()
        if semantics not in ("RF_S0", "RAINFIRST"):
            raise ValueError(
                f"Mopex implements the S0 shared-energy-budget semantics; got {semantics!r}."
            )
        self.interception_semantics = "RF_S0"
        # With counterfactual supervision the gate network may learn only from the
        # counterfactual BCE term, never from the fit or complexity term.
        self.counterfactual_supervision = bool(
            self.config.get("counterfactual_supervision", False)
        )
        removed = self.config.get("removed_processes", self.config.get("removed_process", []))
        if isinstance(removed, str):
            removed = [removed]
        self.removed_processes = {str(name) for name in (removed or [])}
        unknown = self.removed_processes - set(self.weight_names)
        if unknown:
            raise ValueError(f"Unknown removed process names: {sorted(unknown)}")
        self.step_fn = self._compile_step(mopex_step)

    def _structure_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        logits = raw_weights.view(raw_weights.shape[0], len(self.weight_names), 2)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.structure_tau, hard=False, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        if self.removed_processes:
            for index, name in enumerate(self.weight_names):
                if name in self.removed_processes:
                    probs = probs.clone()
                    probs[:, index, :] = torch.tensor((1.0, 0.0), device=probs.device, dtype=probs.dtype)
        return probs[..., 1]

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        weights_on = self._structure_weights(parameters["weights"])
        block_gate_grad = self.counterfactual_supervision and self.training
        weights_on_fit = weights_on.detach() if block_gate_grad else weights_on
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, weights_on_fit, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        result = {"streamflow": Qrouted}
        w_out = self._weight_outputs(weights_on, Q_mopex.shape[0])
        if block_gate_grad:
            # Value-identical outputs without a fit/AIC path into the gate logits.
            w_out = {name: value.detach() for name, value in w_out.items()}
        result.update(w_out)
        return result
