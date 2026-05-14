import os
import json
import shutil
import pandas as pd

CROPS_DIR = "crops"
OUT_DIR = "cup_selected"
ANNOTATION_FILE = "c_ccm_annotations.json"

# filling_level → 状态名
FILLING_MAP = {0: "empty", 1: "half", 2: "full"}

# 读取标注
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

annotations = data['annotations']
records = []
os.makedirs(OUT_DIR, exist_ok=True)

for ann in annotations:
    filling = ann.get('filling_level')
    if filling is None:
        continue
    if filling not in FILLING_MAP:
        continue

    img_id = ann['id']
    src_path = os.path.join(CROPS_DIR, f"{img_id:06d}.png")
    if not os.path.exists(src_path):
        # 只复制 crops 里实际存在的图片
        continue

    dst_path = os.path.join(OUT_DIR, f"{img_id:06d}.png")
    shutil.copy2(src_path, dst_path)

    records.append({
        'image_path': dst_path,          # 绝对路径或相对路径均可
        'object_name': 'cup',
        'state_name': FILLING_MAP[filling]
    })

print(f"成功复制 {len(records)} 张图片到 {OUT_DIR}")

# 生成 CSV
df = pd.DataFrame(records)
df.to_csv("cup_dataset.csv", index=False)
print("生成 cup_dataset.csv")

# 生成 JSON 映射（供后续训练使用）
object_map = {"cup": 0}
state_map = {"empty": 0, "half": 1, "full": 2}
with open("classes_object.json", "w") as f:
    json.dump(object_map, f, indent=2)
with open("classes_state.json", "w") as f:
    json.dump(state_map, f, indent=2)
print("生成 classes_object.json 和 classes_state.json")