"""
ResNet Multi-Task Learning 模型
================================
架构概览:
    Input Image [B, 3, 224, 224]
         │
    ┌────▼────────────────────────────────────┐
    │  Shared Backbone (ResNet-50/18/34/101)  │
    │  → Global Average Pooling               │
    │  → [B, backbone_dim]                    │
    └────┬──────────────┬──────────────────────┘
         │              │
    ┌────▼────┐    ┌────▼──────┐
    │ Head A  │    │  Head B   │
    │ Object  │    │  State    │
    │ FC+BN   │    │  FC+BN    │
    │ → logits│    │  → logits │
    └─────────┘    └───────────┘

支持 ResNet-18 / 34 / 50 / 101 backbone，ImageNet 预训练权重。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
#  Task-Specific Head
# ─────────────────────────────────────────────────────────────────────────────

class TaskHead(nn.Module):
    """
    单任务分类头: Linear → BN → ReLU → Dropout → Linear

    Args:
        in_features : 输入特征维度
        hidden_dim  : 隐藏层维度
        num_classes : 输出类别数
        dropout     : Dropout 概率
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
#  主模型
# ─────────────────────────────────────────────────────────────────────────────

class ResNetMTL(nn.Module):
    """
    基于 ResNet 的多任务学习模型。

    Args:
        num_objects     : 物体类别数（MIT States = 245）
        num_states      : 状态类别数（MIT States = 116，含 'adj' baseline）
        backbone        : "resnet18" | "resnet34" | "resnet50" | "resnet101"
        pretrained      : 是否加载 ImageNet 预训练权重
        head_hidden_dim : 分类头隐藏层维度
        head_dropout    : 分类头 Dropout 概率
        freeze_backbone : 初始是否冻结 backbone（配合 warm-up 使用）
    """

    BACKBONE_OUT_DIM = {
        "resnet18":  512,
        "resnet34":  512,
        "resnet50":  2048,
        "resnet101": 2048,
    }

    def __init__(
        self,
        num_objects: int,
        num_states: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        assert backbone in self.BACKBONE_OUT_DIM, \
            f"不支持的 backbone: {backbone}，请选 {list(self.BACKBONE_OUT_DIM)}"

        self.backbone_name = backbone
        self.num_objects   = num_objects
        self.num_states    = num_states

        # ── Backbone（截断到 GAP，去掉原 FC）─────────────────────
        weights     = "IMAGENET1K_V1" if pretrained else None
        base_model  = getattr(tv_models, backbone)(weights=weights)

        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,   # → [B, C, 1, 1]
        )

        feat_dim = self.BACKBONE_OUT_DIM[backbone]

        # ── 双分支分类头 ──────────────────────────────────────────
        self.obj_head = TaskHead(
            in_features=feat_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_objects,
            dropout=head_dropout,
        )
        self.state_head = TaskHead(
            in_features=feat_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_states,
            dropout=head_dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

    # ──────────────────────────────────────────────────────────────
    #  前向传播
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x               : [B, 3, 224, 224]
            return_features : 若 True，额外返回共享特征向量

        Returns dict:
            "obj_logits"   : [B, num_objects]
            "state_logits" : [B, num_states]
            "features"     : [B, feat_dim]（仅当 return_features=True）
        """
        feat = self.backbone(x)           # [B, C, 1, 1]
        feat = feat.flatten(start_dim=1)  # [B, C]

        out = {
            "obj_logits":   self.obj_head(feat),
            "state_logits": self.state_head(feat),
        }
        if return_features:
            out["features"] = feat
        return out

    # ──────────────────────────────────────────────────────────────
    #  Backbone 冻结 / 解冻
    # ──────────────────────────────────────────────────────────────

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[ResNetMTL] Backbone 已冻结")

    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None):
        if unfreeze_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("[ResNetMTL] Backbone 全部解冻")
        else:
            children = list(self.backbone.children())
            n = min(unfreeze_layers, len(children))
            for module in children[-n:]:
                for param in module.parameters():
                    param.requires_grad = True
            print(f"[ResNetMTL] Backbone 最后 {n} 个模块已解冻")

    def get_trainable_params(self) -> Dict[str, list]:
        return {
            "backbone": [p for p in self.backbone.parameters() if p.requires_grad],
            "heads": (list(self.obj_head.parameters()) +
                      list(self.state_head.parameters())),
        }

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = (filter(lambda p: p.requires_grad, self.parameters())
                  if trainable_only else self.parameters())
        return sum(p.numel() for p in params)

    # ──────────────────────────────────────────────────────────────
    #  Checkpoint
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        payload = {
            "model_state": self.state_dict(),
            "config": {
                "backbone":    self.backbone_name,
                "num_objects": self.num_objects,
                "num_states":  self.num_states,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[ResNetMTL] Checkpoint 已保存 → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu"):
        payload   = torch.load(path, map_location=device)
        cfg       = payload["config"]
        model     = cls(
            num_objects=cfg["num_objects"],
            num_states=cfg["num_states"],
            backbone=cfg["backbone"],
            pretrained=False,
        )
        model.load_state_dict(payload["model_state"])
        model.to(device)
        print(f"[ResNetMTL] Checkpoint 已加载 ← {path}")
        return model, payload


# ─────────────────────────────────────────────────────────────────────────────
#  工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_objects: int, num_states: int, cfg: dict) -> ResNetMTL:
    model = ResNetMTL(
        num_objects=num_objects,
        num_states=num_states,
        backbone=cfg.get("backbone", "resnet50"),
        pretrained=cfg.get("pretrained", True),
        head_hidden_dim=cfg.get("head_hidden_dim", 512),
        head_dropout=cfg.get("head_dropout", 0.5),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )
    total     = model.num_parameters()
    trainable = model.num_parameters(trainable_only=True)
    print(f"[build_model] {cfg.get('backbone','resnet50')} | "
          f"总参数: {total:,} | 可训练: {trainable:,}")
    return model
