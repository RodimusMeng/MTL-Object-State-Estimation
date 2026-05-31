"""
VAW 属性预测模型（YOLOv8 backbone + 物体门控 + 属性多标签头）
============================================================
对齐论文 (Pham et al., CVPR 2021) 的核心设计：
  1. 物体名是 **输入**，通过可学习 embedding + 2-层 MLP gating
     调制图像特征 (image_feat ⊙ σ(g(obj_emb)))
  2. 单一属性头输出 N_attr 个 logits，sigmoid + BCE 训练
  3. 骨干替换为 YOLOv8m-cls（与 ResNet50 同量级），方便与 OSDD 实验对齐

架构：
    image  → YOLOv8m-cls 骨干 → 1280-d 特征 ─┐
                                              ⊙ → MLP → N_attr logits
    obj_id → Embedding → MLP gate (sigmoid) ─┘
"""

import torch
import torch.nn as nn


class VAWModel(nn.Module):
    def __init__(
        self,
        n_obj: int,
        n_attr: int,
        base_weights: str = "weights/yolov8m-cls.pt",
        obj_emb_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(base_weights)
        core = yolo.model.model
        classify = core[-1]
        assert classify.__class__.__name__ == "Classify"

        # ── 图像骨干 ──────────────────────────────────────────────
        self.backbone  = core[:-1]
        self.neck_conv = classify.conv
        self.pool      = classify.pool
        feat_dim       = classify.linear.in_features    # 1280
        self.feat_dim  = feat_dim

        # ── 物体 embedding + 门控 MLP（论文 Eq.2）──────────────────
        self.obj_emb = nn.Embedding(n_obj, obj_emb_dim)
        nn.init.normal_(self.obj_emb.weight, std=0.02)
        self.gate = nn.Sequential(
            nn.Linear(obj_emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),
            nn.Sigmoid(),
        )

        # ── 属性头（单一头，多标签）────────────────────────────────
        self.attr_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_attr),
        )
        nn.init.xavier_uniform_(self.attr_head[0].weight)
        nn.init.xavier_uniform_(self.attr_head[-1].weight)

    def forward(self, img: torch.Tensor, obj_id: torch.Tensor):
        """
        img    : [B, 3, H, W]
        obj_id : [B] long tensor
        """
        x = self.backbone(img)
        x = self.neck_conv(x)
        feat = self.pool(x).flatten(1)            # [B, 1280]

        gate = self.gate(self.obj_emb(obj_id))    # [B, 1280]
        feat = feat * gate                         # 论文 Eq.1

        return self.attr_head(feat)                # [B, n_attr] (logits)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        for p in self.neck_conv.parameters():
            p.requires_grad = not freeze

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 向后兼容
YOLOv8_MTL = VAWModel
