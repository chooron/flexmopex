from __future__ import annotations

from typing import Any

import torch

from train.models.base_mopex import BaseMopex


class FixedWeightMopex(BaseMopex):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "FixedWeightMopex"
        fixed_weights = self.config["fixed_weights"]
        values = [float(fixed_weights[name]) for name in self.weight_names]
        self.register_buffer(
            "fixed_weight_values",
            torch.tensor(values, dtype=torch.float32, device=self.device),
        )
        if self.config.get("teacher_mode"):
            # Dataset-generation teacher: an archived physics step that is not the
            # released RF_S0 simulator. Never reached by training configurations.
            from train.synthetic.teacher import es0_step, first_submission_step

            steps = {"es0": es0_step, "first_submission": first_submission_step}
            self.step_fn = self._compile_step(steps[str(self.config["teacher_step"])])
            self.interception_semantics = str(self.config["teacher_step"])
            return
        semantics = str(self.config.get("interception_semantics", "RF_S0")).upper()
        if semantics not in ("RF_S0", "RAINFIRST"):
            raise ValueError(
                f"FixedWeightMopex implements the S0 shared-energy-budget semantics; got {semantics!r}."
            )
        self.interception_semantics = "RF_S0"
        from train.models.mopex import mopex_step

        self.step_fn = self._compile_step(mopex_step)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        weights_on = self.fixed_weight_values.unsqueeze(0).expand(n_grid, -1)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        result = {"streamflow": Qrouted}
        result.update(self._weight_outputs(weights_on, Q_mopex.shape[0]))
        return result
