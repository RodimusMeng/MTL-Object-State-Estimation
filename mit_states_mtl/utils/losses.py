"""
多任务联合损失函数
==================
提供两种权重策略：
  static  — 固定权重 alpha（默认 0.5/0.5），简单可靠，先用这个
  dynamic — Kendall et al. 2018 不确定性加权，用可学习 log-variance
            自动平衡两个任务，适合两任务收敛速度差异大时
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  静态加权损失
# ─────────────────────────────────────────────────────────────────────────────

class StaticWeightedLoss(nn.Module):
    """L = alpha * L_obj + (1 - alpha) * L_state"""

    def __init__(
        self,
        alpha: float = 0.5,
        obj_weight: Optional[torch.Tensor] = None,
        state_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        assert 0 < alpha < 1
        self.alpha = alpha
        self.register_buffer("obj_weight", obj_weight)
        self.register_buffer("state_weight", state_weight)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        obj_logits: torch.Tensor,
        state_logits: torch.Tensor,
        obj_labels: torch.Tensor,
        state_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_obj = F.cross_entropy(
            obj_logits, obj_labels,
            weight=self.obj_weight,
            label_smoothing=self.label_smoothing,
        )
        loss_state = F.cross_entropy(
            state_logits, state_labels,
            weight=self.state_weight,
            label_smoothing=self.label_smoothing,
        )
        total = self.alpha * loss_obj + (1.0 - self.alpha) * loss_state
        return {
            "loss":        total,
            "loss_obj":    loss_obj.detach(),
            "loss_state":  loss_state.detach(),
            "weight_obj":  torch.tensor(self.alpha),
            "weight_state": torch.tensor(1.0 - self.alpha),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  动态不确定性加权损失 (Kendall et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicUncertaintyLoss(nn.Module):
    """
    L = exp(-log_var_obj)  * L_obj  + 0.5 * log_var_obj
      + exp(-log_var_state)* L_state + 0.5 * log_var_state

    log_var_* 是可学习参数，需加入 optimizer 参数组。
    """

    def __init__(
        self,
        obj_weight: Optional[torch.Tensor] = None,
        state_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.log_var_obj   = nn.Parameter(torch.zeros(1))
        self.log_var_state = nn.Parameter(torch.zeros(1))
        self.register_buffer("obj_weight",   obj_weight)
        self.register_buffer("state_weight", state_weight)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        obj_logits: torch.Tensor,
        state_logits: torch.Tensor,
        obj_labels: torch.Tensor,
        state_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_obj = F.cross_entropy(
            obj_logits, obj_labels,
            weight=self.obj_weight,
            label_smoothing=self.label_smoothing,
        )
        loss_state = F.cross_entropy(
            state_logits, state_labels,
            weight=self.state_weight,
            label_smoothing=self.label_smoothing,
        )
        p_obj   = torch.exp(-self.log_var_obj)
        p_state = torch.exp(-self.log_var_state)
        total = (p_obj   * loss_obj   + 0.5 * self.log_var_obj +
                 p_state * loss_state + 0.5 * self.log_var_state)
        return {
            "loss":        total.squeeze(),
            "loss_obj":    loss_obj.detach(),
            "loss_state":  loss_state.detach(),
            "weight_obj":  p_obj.detach().squeeze(),
            "weight_state": p_state.detach().squeeze(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  统一接口
# ─────────────────────────────────────────────────────────────────────────────

class JointMTLoss(nn.Module):
    """
    多任务损失统一接口。

    Args:
        mode            : "static" | "dynamic"
        alpha           : 仅 static 使用，object 分支权重
        obj_weight      : 类别逆频率权重（可选）
        state_weight    : 类别逆频率权重（可选）
        label_smoothing : 标签平滑系数

    用法:
        criterion = JointMTLoss(mode="dynamic")
        # dynamic 模式要把 criterion.parameters() 加进 optimizer！
        out = criterion(obj_logits, state_logits, obj_labels, state_labels)
        out["loss"].backward()
    """

    def __init__(
        self,
        mode: str = "static",
        alpha: float = 0.5,
        obj_weight: Optional[torch.Tensor] = None,
        state_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        assert mode in ("static", "dynamic"), \
            f"mode 只支持 'static' 或 'dynamic'，收到 '{mode}'"
        self.mode = mode
        if mode == "static":
            self.loss_fn = StaticWeightedLoss(
                alpha=alpha,
                obj_weight=obj_weight,
                state_weight=state_weight,
                label_smoothing=label_smoothing,
            )
        else:
            self.loss_fn = DynamicUncertaintyLoss(
                obj_weight=obj_weight,
                state_weight=state_weight,
                label_smoothing=label_smoothing,
            )

    def forward(
        self,
        obj_logits: torch.Tensor,
        state_logits: torch.Tensor,
        obj_labels: torch.Tensor,
        state_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(obj_logits, state_logits, obj_labels, state_labels)
