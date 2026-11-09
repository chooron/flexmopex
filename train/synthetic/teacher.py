"""Teacher physics used only for synthetic dataset generation.

Two equations appear here and nowhere in the training path:

``first_submission_step``
    Original submitted model: interception of TOTAL precipitation with an
    amplitude ``alpha`` and a 365-day cosine. Used as the E2 teacher (teacher A').

``es0_step``
    Candidate E-S0: interception of total precipitation with a bounded linear
    cosine gate (``kappa``, ``phi``, 365.25-day period) and a shared PET budget.
    This is the teacher the E2 checkpoint was trained with.

The released Full/Flex models use ``train/models/mopex.py::mopex_step`` instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def first_submission_step(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights (New Inputs) ---
    w_phen: torch.Tensor,  # Phenology control [0, 1]
    w_int: torch.Tensor,  # Interception area [0, 1]
    w_snow: torch.Tensor,  # Snow accumulation area [0, 1]
    w_sub: torch.Tensor,  # Subsurface fast-flow connectivity [0, 1]
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # ============================================================
    # 0. Guards
    # ============================================================
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn  = F.relu(Sn)

    # ============================================================
    # 1. Phenology Module (Soft Switch) - GSI with trange
    # tmax 在接口中保留，内部作为 trange 使用（tmax = tmin + trange → trange = tmax - tmin）
    # ============================================================
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)

    # w_phen = 0: Use raw PET; w_phen = 1: Use GSI-modified PET
    PET_bio = PET * flux_gsi
    PET_effective = w_phen * PET_bio + (1.0 - w_phen) * PET

    # ============================================================
    # 2. Interception Module (Flux Gating)
    # 截留受 PET_effective 约束，防止能量不守恒
    # ============================================================
    is_time_safe = torch.clamp(is_time, 0.0, 365.0)
    rad = 2.0 * torch.pi * (doy - is_time_safe) / 365.0
    season_factor = 0.5 * (torch.cos(rad) + 1.0)

    flux_potential = alpha * P * season_factor
    flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))

    # Only w_int fraction of the area intercepts water
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    # 截留消耗后的剩余 PET 交给土壤蒸发
    pet_for_soil = F.relu(PET_effective - flux_i)

    # ============================================================
    # 3. Snow Module (Soft Switch)
    # 用 sigmoid 替代硬阈值，打通 tcrit 的梯度回传
    # ============================================================
    is_rain = torch.sigmoid(T - tcrit)

    # Path A: Bypass (Direct to soil)
    P_bypass  = P_through * is_rain + P_through * (1.0 - is_rain) * (1.0 - w_snow)

    # Path B: Storage (Enters Snowpack)
    P_to_snow = P_through * (1.0 - is_rain) * w_snow

    # 融雪：sigmoid × softplus，消除冰点以下虚假融雪，保证 tcrit/ddf 全程可导
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn    = torch.minimum(melt_drive * ddf, Sn)

    Sn_new = Sn + P_to_snow - flux_qn
    P_eff  = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # 顺序：溢流 → 蒸发 → 下渗
    # ============================================================
    # Step 1: 溢流
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    # Step 2: 蒸发（使用截留后剩余 PET）
    ratio_s1 = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1 = torch.minimum(pet_for_soil * ratio_s1, S1)
    S1 = S1 - flux_et1

    # Step 3: 下渗（指数解析解）
    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new  = S1 - flux_qw

    # ============================================================
    # 5. Subsurface Module (S2) (State Leakage)
    # ============================================================
    S2 = S2 + flux_qw

    # 地下溢流：仅 w_sub 比例快速流出
    flux_q2f_pot = F.relu(S2 - Sb2)
    flux_q2f = flux_q2f_pot * w_sub
    S2 = S2 - flux_q2f

    # 基流（指数解析解）
    flux_q2u = S2 * (1.0 - torch.exp(-1.0 / (tu + nearzero)))
    S2 = S2 - flux_q2u

    # 蒸发（逐层扣减后的剩余 PET）
    remaining_pet = F.relu(pet_for_soil - flux_et1)
    ratio_s2  = torch.clamp(S2 / (Se + nearzero), max=1.0)
    flux_et2  = torch.minimum(remaining_pet * ratio_s2, S2)
    S2_new    = S2 - flux_et2

    # ============================================================
    # 6. Routing（指数解析解）
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total  = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new

def _phase_rad(doy: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    phi_safe = torch.clamp(phi, 0.0, 365.0)
    return 2.0 * torch.pi * (doy - phi_safe) / 365.25

def season_linear(doy: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Candidate E: bounded linear cosine, s in [0, 1] for kappa in [0, 1]."""
    return 0.5 * (1.0 + kappa * torch.cos(_phase_rad(doy, phi)))

def _mopex_step_candidate(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights ---
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
    *,
    season_mode: str = "linear",
    pet_cap: bool = True,
    pet_independent: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body shared by candidates E/F under S0/S1/S2 semantics.

    ``season_mode``: "linear" (candidate E) or "logistic" (candidate F).
    ``pet_cap``: apply the production interception PET cap (S0/S1) or not (S2).
    ``pet_independent``: V1-style independent interception loss (S1/S2) or
    production shared PET budget (S0).
    """
    # ============================================================
    # 0. Guards
    # ============================================================
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # ============================================================
    # 1. Phenology Module (Soft Switch) - GSI with trange
    # ============================================================
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)

    PET_bio = PET * flux_gsi
    PET_effective = w_phen * PET_bio + (1.0 - w_phen) * PET

    # ============================================================
    # 2. Interception Module (bounded seasonal gate)
    # ============================================================
    if season_mode == "linear":
        season = season_linear(doy, kappa, phi)
    elif season_mode == "logistic":
        season = season_logistic(doy, kappa, phi)
    else:
        raise ValueError(f"Unknown season_mode {season_mode!r}")

    flux_potential = P * season
    if pet_cap:
        # Production safety cap (S0/S1): interception never exceeds P or PET.
        flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))
    else:
        # S2: no PET cap; precipitation bound preserved (redundant for s<=1).
        flux_i_pot = torch.minimum(flux_potential, P)

    # w_int is the only multiplicative amplitude control, applied after caps
    # (identical placement to production V0).
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    if pet_independent:
        # S1/S2: interception loss is independent of the soil PET budget.
        pet_for_soil = PET_effective
    else:
        # S0: interception consumes PET before soil ET.
        pet_for_soil = F.relu(PET_effective - flux_i)

    # ============================================================
    # 3. Snow Module (Soft Switch)
    # ============================================================
    is_rain = torch.sigmoid(T - tcrit)

    P_bypass = P_through * is_rain + P_through * (1.0 - is_rain) * (1.0 - w_snow)
    P_to_snow = P_through * (1.0 - is_rain) * w_snow

    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn = torch.minimum(melt_drive * ddf, Sn)

    Sn_new = Sn + P_to_snow - flux_qn
    P_eff = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # ============================================================
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    ratio_s1 = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1 = torch.minimum(pet_for_soil * ratio_s1, S1)
    S1 = S1 - flux_et1

    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new = S1 - flux_qw

    # ============================================================
    # 5. Subsurface Module (S2)
    # ============================================================
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

    # ============================================================
    # 6. Routing
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new

def install_teacher_aprime(inner) -> dict:
    """Swap the loaded teacher step to the first-submission equation (E2 teacher A')."""
    phy = inner.phy_model
    if type(phy).__name__ != "FixedWeightMopex":
        raise RuntimeError(f"teacher A' expects FixedWeightMopex, got {type(phy).__name__}")
    current = getattr(phy, "step_fn", None)
    name = getattr(current, "__name__", type(current).__name__)
    if current is first_submission_step:
        return {"installed": True, "changed": False, "step_fn": "first_submission_step"}
    phy.step_fn = first_submission_step
    return {"installed": True, "changed": True, "previous_step_fn": name,
            "step_fn": "first_submission_step"}


def es0_step(*args, **kwargs):
    """Candidate E-S0 daily step (shared PET budget)."""
    return _mopex_step_candidate(*args, **kwargs, season_mode="linear",
                                 pet_cap=True, pet_independent=False)
