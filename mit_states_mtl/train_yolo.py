"""
YOLOv8 杯子状态检测 — 训练脚本
================================
用法:
    python train_yolo.py                          # 默认配置
    python train_yolo.py --model yolov8s.pt       # 更大模型
    python train_yolo.py --epochs 200 --batch 8   # 调参
    python train_yolo.py --resume                 # 从上次断点续训

模型选择参考（速度 vs 精度）:
    yolov8n  最小最快，资源少时首选
    yolov8s  平衡，推荐
    yolov8m  更准，需要较好 GPU
"""

import argparse
import os
import sys
from pathlib import Path

# fmt: off
# Windows: libiomp5md.dll 在 torch/lib 和 conda Library/bin 各有一份，版本不同。
# 必须在 torch import 之前抢先加载 torch/lib 的版本，避免 fbgemm.dll 拿到错误版本。
if sys.platform == "win32":
    import ctypes as _ct
    import importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _tlib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            # 抢先锁定 torch 自带的 libiomp5md，防止 conda MKL 版本抢占
            _iomp = os.path.join(_tlib, "libiomp5md.dll")
            if os.path.isfile(_iomp):
                _ct.CDLL(_iomp)
    _cbin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "Library", "bin"))
    if os.path.isdir(_cbin):
        os.add_dll_directory(_cbin)
# fmt: on


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s.pt", help="预训练权重文件名（首次运行自动下载）")
    p.add_argument("--data", default="datasets/cup/yolo/cup.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="显存不够时改为 8 或 4")
    p.add_argument("--device", default="", help="留空自动选 GPU；无 GPU 填 cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="cup_state")
    p.add_argument("--resume", action="store_true", help="从 runs/detect/cup_state/weights/last.pt 续训")
    return p.parse_args()


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics，请先运行：")
        print("    pip install ultralytics")
        sys.exit(1)

    args = parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        print(f"找不到数据配置：{data_yaml}")
        print("请先运行：python datasets/convert_to_yolo.py")
        sys.exit(1)

    # ── 续训 ─────────────────────────────────────────────────
    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.is_file():
            print(f"找不到断点文件：{last_pt}")
            sys.exit(1)
        print(f"续训：{last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
        return

    # ── 新训练 ────────────────────────────────────────────────
    print(f"模型：{args.model}  数据：{data_yaml.name}  epochs：{args.epochs}")
    model = YOLO(args.model)

    # 绝对路径，防止 YOLO 自动拼接 runs/detect/ 前缀导致路径重复
    project_abs = str(Path(args.project).resolve())

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        project=project_abs,
        name=args.name,
        exist_ok=False,
        # ── 针对本数据集的增强设置 ────────────────────────────
        # 垂直翻转关闭：倒立杯上下翻后语义变正立
        flipud=0.0,
        # 水平翻转保留
        fliplr=0.5,
        # 旋转限制在 ±15°：更大角度会混淆侧躺/倒立语义
        degrees=15,
        # mosaic 保留：有助于缩小网络图与实拍图的域差异
        mosaic=1.0,
        # mixup 关闭：类别边界模糊，不适合姿态分类
        mixup=0.0,
        # ── 训练策略 ──────────────────────────────────────────
        patience=30,  # 30 epoch val 无提升则 early stop
        save_period=20,  # 每 20 epoch 存一次中间权重
        val=True,
        plots=True,  # 自动生成 PR 曲线、混淆矩阵等
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n训练完成，最优权重：{best}")
    print(f"\n验证命令：")
    print(f"    python eval_yolo.py --model {best}")


if __name__ == "__main__":
    main()
