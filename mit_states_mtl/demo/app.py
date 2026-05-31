"""
Streamlit 演示界面 — 物体状态识别
===================================
用法:
    pip install streamlit
    streamlit run demo/app.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Windows DLL 修复（必须在 torch/ultralytics import 之前）
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

import streamlit as st
import numpy as np
from PIL import Image

# ─── 模型配置 ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

MODELS = {
    "OSDD — YOLOv8s（物体状态检测）": ROOT / "runs" / "osdd_yolov8s" / "weights" / "best.pt",
    "OSDD — YOLOv8n（轻量版）":       ROOT / "runs" / "osdd_yolov8n" / "weights" / "best.pt",
    "OSDD — YOLOv8m（高精度版）":     ROOT / "runs" / "osdd_yolov8m" / "weights" / "best.pt",
    "杯子状态识别（YOLOv8s）":         ROOT / "runs" / "detect" / "cup_state-1" / "weights" / "best.pt",
}

OSDD_CLASSES = [
    "close", "containing_liquid", "containing_solid", "empty",
    "folded", "open", "plugged", "unplugged", "unfolded"
]

CLASS_COLORS = {
    "close": "#E74C3C",
    "containing_liquid": "#3498DB",
    "containing_solid": "#E67E22",
    "empty": "#95A5A6",
    "folded": "#9B59B6",
    "open": "#27AE60",
    "plugged": "#F39C12",
    "unplugged": "#1ABC9C",
    "unfolded": "#2ECC71",
    "cup_empty": "#BDC3C7",
    "cup_half": "#3498DB",
    "cup_full": "#2980B9",
    "cup_sideways": "#E67E22",
    "cup_inverted": "#E74C3C",
}


@st.cache_resource
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def draw_results(image: Image.Image, results) -> Image.Image:
    """在原图上绘制检测框和标签"""
    import cv2
    img_cv = np.array(image.convert("RGB"))
    img_cv = img_cv[:, :, ::-1].copy()  # RGB → BGR

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names[cls_id]

            # 颜色
            hex_c = CLASS_COLORS.get(label, "#4C72B0").lstrip("#")
            bgr = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), bgr, 2)
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_cv, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(img_cv, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return Image.fromarray(img_cv[:, :, ::-1])


# ─── 页面布局 ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="物体状态识别 Demo",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 物体-状态联合识别演示")
st.markdown("""
**项目**：基于多任务学习的物体-状态联合估计 | **模型**：YOLOv8 fine-tuned on OSDD & Cup Dataset
""")

# ─── 侧边栏 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 参数设置")

    available = {k: v for k, v in MODELS.items() if v.is_file()}
    if not available:
        st.error("暂无可用模型权重，请先完成训练。")
        st.stop()

    model_name = st.selectbox("选择模型", list(available.keys()))
    model_path = str(available[model_name])

    conf_thresh = st.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)
    iou_thresh  = st.slider("NMS IoU 阈值", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**模型说明**")
    if "OSDD" in model_name:
        st.markdown("9类物体状态：close / open / empty / containing_liquid / containing_solid / folded / plugged / unplugged / unfolded")
    else:
        st.markdown("5类杯子状态：empty / half / full / sideways / inverted")

# ─── 主区域 ──────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 上传图片")
    uploaded = st.file_uploader(
        "支持 JPG / PNG / JPEG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    use_example = st.checkbox("使用示例图片（来自测试集）")
    if use_example and not uploaded:
        example_dirs = [
            ROOT / "datasets" / "OSDD" / "images" / "test",
            ROOT / "datasets" / "mit_classify" / "test",
            ROOT / "datasets",
        ]
        example_img = None
        for d in example_dirs:
            imgs = list(d.glob("**/*.png")) + list(d.glob("**/*.jpg"))
            if imgs:
                example_img = imgs[0]
                break
        if example_img:
            image = Image.open(example_img)
            st.image(image, caption=f"示例: {example_img.name}", width="stretch")
        else:
            st.info("未找到示例图片")
            image = None
    elif uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="上传的图片", width="stretch")
    else:
        image = None

with col2:
    st.subheader("📊 识别结果")

    if image is not None:
        with st.spinner("推理中..."):
            model = load_model(model_path)

            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.convert("RGB").save(tmp.name)
                tmp_path = tmp.name

            results = model.predict(
                source=tmp_path,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
            )
            os.unlink(tmp_path)

        # 绘制结果图
        result_img = draw_results(image, results)
        st.image(result_img, caption="检测结果", width="stretch")

        # 详细数据表格
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            names = results[0].names
            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                data.append({
                    "类别": label,
                    "置信度": f"{conf:.2%}",
                    "位置 (x1,y1,x2,y2)": f"({x1},{y1},{x2},{y2})",
                })

            st.markdown("**检测详情：**")
            st.dataframe(data, use_container_width=True)

            # 置信度条形图
            labels_list = [d["类别"] for d in data]
            confs_list  = [float(d["置信度"].strip("%")) / 100 for d in data]
            chart_data  = {"类别": labels_list, "置信度": confs_list}
            import pandas as pd
            st.bar_chart(pd.DataFrame(chart_data).set_index("类别"))
        else:
            st.info("未检测到目标（置信度低于阈值）")
    else:
        st.info("请在左侧上传图片或勾选使用示例图片")

# ─── 底部说明 ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<small>
**数据集**：MIT States (63K images, 1890 classes) · OSDD (14.5K images, 9 state classes)
**模型**：YOLOv8s/n/m fine-tuned · **框架**：Ultralytics + PyTorch 2.7 + CUDA 12.8
**项目**：2551661 蒙天成 et al. — 基于多任务学习的物体-状态联合估计
</small>
""", unsafe_allow_html=True)
