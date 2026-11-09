from typing import Any, Optional, Union, Dict

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseDynAicBatchLoss(BaseCriterion):
    """Fit loss with the structural complexity charge.

    L = NSE_fit + alpha * sum_p(cost_p * mean_{b,t} w_{b,p})

    NSE_fit is the per-catchment variance-normalized squared error; the complexity
    term uses the time- and basin-mean gate so a gate that is on every day is
    charged once, not once per timestep. The counterfactual BCE term lives in
    ``train/models/cf_trainer.py``.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device for the loss tensors.
    **kwargs
        - y_obs: Tensor [n_time, n_grid, 1], used for the normalization std.
        - eps: Stability term. Default 0.1.
        - aic_alpha: Complexity weight. Default 0.01.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = "cpu",
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = "Batch NSE Loss with Dynamic Soft-AIC"
        self.config = config
        self.device = device

        # --- 1. NSE Initialization ---
        y_obs = kwargs["y_obs"]
        # Pre-calculate STD for each grid for NSE normalization.
        self.std = np.nanstd(y_obs[:, :, 0].cpu().detach().numpy(), axis=0)

        self.eps = kwargs.get("eps", config.get("eps", 0.1))

        # --- 2. AIC Initialization ---
        # Alpha: Controls the trade-off between fitting (NSE) and sparsity (Complexity)
        # Suggest starting small (e.g., 1e-3) and potentially increasing during training.
        self.aic_alpha = kwargs.get("aic_alpha", config.get("aic_alpha", 0.0))

        # Define the parameter cost for each structural module.
        # This represents how many "free parameters" each module adds.
        self.param_costs = {
            "w_phen": 2.0,  # Adds: tmin, tmax
            "w_int": 2.0,  # Adds: alpha, is_time
            "w_snow": 2.0,  # Adds: ddf, tcrit (or tr)
            "w_sub": 1.0,  # Adds: structural path (complexifies Sb2 usage)
        }

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:

        # y_pred: [n_time, n_grid, 1]
        prediction, target = self._format(y_pred, y_obs)

        sample_ids = kwargs["sample_ids"]
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.cpu().numpy().astype(int)
        else:
            sample_ids = np.asarray(sample_ids).astype(int)

        # ============================================================
        # 1. 计算 Fit Loss (NSE Loss) — 与 NseBatchLoss 一致
        # ============================================================
        if len(target) > 0:
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )

            # Mask: 只取 target 非 NaN 的位置（与 nse_batch_loss 一致）
            mask = ~torch.isnan(target)

            # 用布尔索引提取有效子集，避免 torch.where 的梯度穿透问题
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]

            # 计算归一化残差
            sq_res = (p_sub - t_sub) ** 2
            norm_res = sq_res / (std_sub + self.eps) ** 2

            loss_fit = torch.mean(norm_res)
        else:
            loss_fit = torch.tensor(0.0, device=self.device)

        # ============================================================
        # 2. Dynamic AIC Penalty (针对时变权重)
        # ============================================================
        weights_dict = kwargs.get("weights", None)
        loss_complexity = torch.tensor(0.0, device=self.device)

        if weights_dict is not None:
            for name, cost in self.param_costs.items():
                if name in weights_dict:
                    w = weights_dict[name]
                    # w shape: [n_steps, n_grid, 1]

                    # 关键修改：先计算时间维度的平均值
                    # 这样可以避免AIC值过大，因为每个时间步都会贡献一次
                    w_time_avg = torch.mean(w, dim=0)  # [n_grid, 1]

                    # 然后计算空间维度的平均值
                    w_spatial_avg = torch.mean(w_time_avg)  # scalar

                    # 累加到复杂度损失
                    loss_complexity += w_spatial_avg * cost

        # ============================================================
        # 3. 组合 Loss
        # ============================================================
        final_loss = loss_fit + self.aic_alpha * loss_complexity

        return final_loss
