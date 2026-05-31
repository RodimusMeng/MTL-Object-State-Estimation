"""
在缓存的 DINOv2 特征上训练 MTL（物体 + 属性）双任务模型
=========================================================
受限硬件下的最优训练：特征预提取后全部驻留显存，无需读图、无 DataLoader，
每个 epoch 仅为头部的矩阵运算 —— 秒级，且彻底规避 Windows 多进程/内存问题。

  - 物体任务：单标签分类，指标 Top-1 Accuracy
  - 属性任务：多标签预测，RW-BCE 损失，指标 federated mAP（每类只在有标注样本上算）
  - 损失平衡：Kendall 不确定性加权（见 models/feat_mtl.py）
  - 两个指标 **分开报告**，不做相乘（多标签 mAP 与单标签 acc 不可乘）

前置：先运行 datasets/extract_dino_features.py 生成 datasets/vaw_features/

用法:
    python train_mtl_feats.py
    python train_mtl_feats.py --epochs 80 --batch 1024 --lr 1e-3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# fmt: off
if sys.platform == "win32":
    import ctypes as _ct, importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _tlib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            _iomp = os.path.join(_tlib, "libiomp5md.dll")
            if os.path.isfile(_iomp): _ct.CDLL(_iomp)
    _cbin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "Library", "bin"))
    if os.path.isdir(_cbin): os.add_dll_directory(_cbin)
# fmt: on

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

FEAT_DIR = "datasets/vaw_features"
OUT_DIR = Path("runs/vaw_mtl_dino")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir",  default=FEAT_DIR)
    p.add_argument("--epochs",    type=int,   default=80)
    p.add_argument("--batch",     type=int,   default=1024)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--trunk-dim", type=int,   default=512)
    p.add_argument("--dropout",   type=float, default=0.4)
    p.add_argument("--alpha",     type=float, default=0.1, help="RW-BCE 平滑因子")
    p.add_argument("--missing-w", type=float, default=0.05, help="缺失标签软负样本权重")
    p.add_argument("--wd",        type=float, default=1e-4)
    p.add_argument("--device",    default="")
    return p.parse_args()


# ── RW-BCE（与 train_vaw_mtl.py 一致，论文 Eq.13）───────────────────
def compute_rw_weights(n_pos, n_neg, alpha=0.1):
    n_pos = np.maximum(n_pos, 1).astype(np.float64)
    n_neg = np.maximum(n_neg, 1).astype(np.float64)
    C = len(n_pos)
    wc = (1.0 / n_pos) ** alpha
    wc = wc / wc.sum() * C
    pc_raw = (1.0 / n_pos) ** alpha
    nc_raw = (1.0 / n_neg) ** alpha
    s = pc_raw + nc_raw
    pc = 2.0 * pc_raw / s
    nc = 2.0 * nc_raw / s
    return (torch.tensor(wc, dtype=torch.float32),
            torch.tensor(pc, dtype=torch.float32),
            torch.tensor(nc, dtype=torch.float32))


def rw_bce_loss(logits, y, wc, pc, nc, missing_w=0.05):
    log_sig   = F.logsigmoid(logits)
    log_1msig = F.logsigmoid(-logits)
    is_pos = (y == 1).float()
    is_neg = (y == 0).float()
    is_mis = (y == -1).float()
    loss_pos = -wc * pc * log_sig    * is_pos
    loss_neg = -wc * nc * log_1msig  * is_neg
    loss_mis = -wc * nc * log_1msig  * is_mis * missing_w
    loss = loss_pos + loss_neg + loss_mis
    denom = is_pos.sum() + is_neg.sum() + is_mis.sum() * missing_w + 1e-6
    return loss.sum() / denom


# ── 评估：物体 Top-1 + 属性 federated mAP ──────────────────────────
@torch.no_grad()
def evaluate(model, feat, obj, attr, device, batch=4096):
    model.eval()
    n = feat.shape[0]
    obj_correct = 0
    all_scores = []
    for i in range(0, n, batch):
        x = feat[i:i+batch]
        obj_logits, attr_logits = model(x)
        obj_correct += (obj_logits.argmax(1) == obj[i:i+batch]).sum().item()
        all_scores.append(torch.sigmoid(attr_logits).float().cpu().numpy())
    obj_acc = obj_correct / n

    s = np.concatenate(all_scores, axis=0)        # [N, C]
    y = attr.cpu().numpy()                         # [N, C] ∈ {1,0,-1}
    aps, valid = [], 0
    for c in range(y.shape[1]):
        labeled = (y[:, c] != -1)
        if labeled.sum() < 2:
            continue
        yc = y[labeled, c]
        if yc.sum() < 1 or yc.sum() == len(yc):
            continue
        aps.append(average_precision_score(yc, s[labeled, c]))
        valid += 1
    attr_map = float(np.mean(aps)) if aps else 0.0
    return obj_acc, attr_map, valid


def load_split(feat_dir, split, device):
    feat = np.load(os.path.join(feat_dir, f"{split}_feat.npy"))   # fp16
    obj  = np.load(os.path.join(feat_dir, f"{split}_obj.npy"))
    attr = np.load(os.path.join(feat_dir, f"{split}_attr.npy"))   # int8
    feat = torch.from_numpy(feat.astype(np.float32)).to(device)
    obj  = torch.from_numpy(obj.astype(np.int64)).to(device)
    attr = torch.from_numpy(attr.astype(np.float32)).to(device)   # {1,0,-1}
    return feat, obj, attr


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fd = args.feat_dir

    with open(os.path.join(fd, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_obj  = len(meta["obj2id"])
    n_attr = len(meta["attr2id"])
    feat_dim = meta["feat_dim"]
    print(f"设备: {device} | 骨干: {meta['model']} | 特征维度: {feat_dim}")
    print(f"物体类别: {n_obj}  属性类别: {n_attr}")

    print("载入缓存特征到显存...")
    tr_f, tr_o, tr_a = load_split(fd, "train", device)
    va_f, va_o, va_a = load_split(fd, "val",   device)
    te_f, te_o, te_a = load_split(fd, "test",  device)
    print(f"  train {tuple(tr_f.shape)} | val {tuple(va_f.shape)} | "
          f"test {tuple(te_f.shape)}")

    stats = np.load(os.path.join(fd, "stats.npz"))
    wc, pc, nc = compute_rw_weights(stats["n_pos"], stats["n_neg"], args.alpha)
    wc, pc, nc = wc.to(device), pc.to(device), nc.to(device)

    from models.feat_mtl import FeatMTL
    model = FeatMTL(feat_dim, n_obj, n_attr,
                    trunk_dim=args.trunk_dim, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    n = tr_f.shape[0]
    log, best_map = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum = ol_sum = al_sum = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            x, yo, ya = tr_f[idx], tr_o[idx], tr_a[idx]
            obj_logits, attr_logits = model(x)
            obj_loss  = F.cross_entropy(obj_logits, yo)
            attr_loss = rw_bce_loss(attr_logits, ya, wc, pc, nc, args.missing_w)
            loss = model.combine_losses(obj_loss, attr_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item(); ol_sum += obj_loss.item()
            al_sum += attr_loss.item(); nb += 1
        scheduler.step()

        obj_acc, attr_map, valid = evaluate(model, va_f, va_o, va_a, device)
        wo, wa = model.task_weights
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"obj_loss {ol_sum/nb:.3f} attr_loss {al_sum/nb:.4f} | "
              f"val obj_acc {obj_acc:.4f}  attr_mAP {attr_map:.4f} "
              f"({valid} cls) | w(o/a)={wo:.2f}/{wa:.2f} | {time.time()-t0:.1f}s",
              flush=True)
        log.append({"epoch": epoch, "obj_acc": obj_acc, "attr_mAP": attr_map,
                    "obj_loss": ol_sum/nb, "attr_loss": al_sum/nb})

        if attr_map > best_map:
            best_map = attr_map
            torch.save(model, OUT_DIR / "best.pt")
        torch.save(model, OUT_DIR / "last.pt")

    # ── 测试集 ─────────────────────────────────────────────────────
    print("\n─── 测试集（best.pt）───")
    best = torch.load(OUT_DIR / "best.pt", map_location=device,
                      weights_only=False)
    t_acc, t_map, t_valid = evaluate(best, te_f, te_o, te_a, device)
    print(f"  test obj_acc {t_acc:.4f} | attr_mAP {t_map:.4f} ({t_valid} cls)")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"args": vars(args), "train_log": log,
                   "test": {"obj_acc": t_acc, "attr_mAP": t_map}},
                  f, indent=2)
    print(f"\n最优 val attr_mAP: {best_map:.4f} | 日志: {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
