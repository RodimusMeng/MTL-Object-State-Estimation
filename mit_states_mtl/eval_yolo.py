"""
YOLOv8 杯子状态检测 — 评估脚本
================================
在 test 集（训练中从未见过）上评估模型，输出每类指标并保存混淆矩阵。

用法:
    python eval_yolo.py --model runs/detect/runs/detect/cup_state-2/weights/best.pt
    python eval_yolo.py --model best.pt --split val   # 跑验证集
    python eval_yolo.py --model best.pt --split test  # 跑测试集（默认）
"""

import argparse
import os
import sys
from pathlib import Path

# fmt: off
if sys.platform == "win32":
    import importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _tlib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            import ctypes as _ct
            _iomp = os.path.join(_tlib, "libiomp5md.dll")
            if os.path.isfile(_iomp):
                _ct.CDLL(_iomp)
    _cbin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "Library", "bin"))
    if os.path.isdir(_cbin):
        os.add_dll_directory(_cbin)
# fmt: on

from ultralytics import YOLO

CLASSES = ["cup_empty", "cup_half", "cup_full", "cup_sideways", "cup_inverted"]
DATA_YAML = "datasets/cup/yolo/cup.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="best.pt 路径")
    p.add_argument("--data",  default=DATA_YAML)
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="评估哪个分割（test=从未参与训练）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf",  type=float, default=0.25, help="置信度阈值")
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"找不到权重文件: {model_path}")
        sys.exit(1)

    model = YOLO(str(model_path))

    # 评估结果保存到权重文件旁边的 eval_{split}/ 目录
    # 用绝对路径，防止 YOLO 自动拼接 runs/detect/ 前缀
    run_dir  = model_path.parent.parent.resolve()  # .../cup_state-1/
    eval_dir = run_dir / f"eval_{args.split}"

    print(f"\n评估: {model_path.name}  split={args.split}\n")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device or None,
        project=str(eval_dir.parent),   # 绝对路径
        name=eval_dir.name,
        plots=True,
    )

    # ── 打印每类指标 ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'类别':<14} {'样本':>5} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    print("-" * 55)

    names = model.names  # {0: 'cup_empty', ...}
    p_per  = metrics.box.p          # per-class precision
    r_per  = metrics.box.r          # per-class recall
    ap50   = metrics.box.ap50       # per-class mAP@0.5
    ap     = metrics.box.ap         # per-class mAP@0.5:0.95
    nc     = metrics.box.nc if hasattr(metrics.box, 'nc') else len(names)

    for i in range(len(names)):
        print(f"  {names[i]:<12} {'?':>5} {p_per[i]:>7.3f} {r_per[i]:>7.3f} "
              f"{ap50[i]:>7.3f} {ap[i]:>9.3f}")

    print("-" * 55)
    print(f"  {'all':<12} {'':>5} {metrics.box.mp:>7.3f} {metrics.box.mr:>7.3f} "
          f"{metrics.box.map50:>7.3f} {metrics.box.map:>9.3f}")
    print("=" * 55)

    save_dir = Path(metrics.save_dir) if hasattr(metrics, 'save_dir') else model_path.parent.parent
    print(f"\n图表保存到: {save_dir}")
    print("  confusion_matrix_normalized.png  ← 重点看这个")
    print("  BoxPR_curve.png")
    print("  BoxF1_curve.png")


if __name__ == "__main__":
    main()