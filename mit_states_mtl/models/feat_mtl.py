"""
基于冻结骨干特征的多任务（MTL）模型
=====================================
真正的多任务学习结构：
    冻结特征 (DINOv2) → 共享可训练 trunk → ┬→ 物体分类头 (单标签)
                                          └→ 属性预测头 (多标签)

两个任务的梯度在 **共享 trunk** 处交汇 —— 这才是 MTL 的本质
（共享可学习表示），即使骨干冻结也成立。

损失平衡采用 Kendall 不确定性加权（Kendall et al., CVPR 2018），
自动平衡"物体 CE"与"属性 RW-BCE"两个量级悬殊的损失，
对应选题书中"动态加权解决梯度主导"的设想。
"""

import torch
import torch.nn as nn


class FeatMTL(nn.Module):
    def __init__(self, feat_dim: int, n_obj: int, n_attr: int,
                 trunk_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        # ── 共享 trunk（两个任务共用、可训练）─────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # ── 双任务头 ──────────────────────────────────────────────
        self.obj_head  = nn.Linear(trunk_dim, n_obj)    # 物体：单标签
        self.attr_head = nn.Linear(trunk_dim, n_attr)   # 属性：多标签

        # ── Kendall 不确定性加权参数（log σ²，可学习）─────────────
        self.log_var_obj  = nn.Parameter(torch.zeros(()))
        self.log_var_attr = nn.Parameter(torch.zeros(()))

        nn.init.xavier_uniform_(self.trunk[0].weight)
        nn.init.zeros_(self.trunk[0].bias)
        for head in (self.obj_head, self.attr_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        """x: [B, feat_dim] → (obj_logits [B,n_obj], attr_logits [B,n_attr])"""
        h = self.trunk(x)
        return self.obj_head(h), self.attr_head(h)

    def combine_losses(self, obj_loss, attr_loss):
        """Kendall 不确定性加权：L = Σ exp(-s_i)·L_i + 0.5·s_i"""
        lo = torch.exp(-self.log_var_obj)  * obj_loss  + 0.5 * self.log_var_obj
        la = torch.exp(-self.log_var_attr) * attr_loss + 0.5 * self.log_var_attr
        return lo + la

    @property
    def task_weights(self):
        """返回当前两个任务的有效权重 exp(-s)，便于观察平衡情况。"""
        return (float(torch.exp(-self.log_var_obj)),
                float(torch.exp(-self.log_var_attr)))