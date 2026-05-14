import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#  1. 读取数据/修改文件路径
df = pd.read_csv('predictions.csv')   


#  2. 定义评估函数 
def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n=== {name} 指标 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    return acc, prec, rec, f1


# 分别评估物体和状态
evaluate(df['true_obj'], df['pred_obj'], "Object")
evaluate(df['true_state'], df['pred_state'], "State")


#  3. 联合准确率 
joint_correct = ( (df['true_obj'] == df['pred_obj']) & 
                  (df['true_state'] == df['pred_state']) )
joint_acc = joint_correct.mean()
print(f"\n=== 联合准确率 (严格) ===")
print(f"Joint Accuracy: {joint_acc:.4f}  ({joint_correct.sum()} / {len(df)})")


#  4. 混淆矩阵热力图 
def plot_confusion(y_true, y_pred, title, filename):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"保存混淆矩阵图: {filename}")
########################################## ###########
#####################################################
plot_confusion(df['true_obj'], df['pred_obj'], 'Object Confusion Matrix', 'confusion_object.png')
plot_confusion(df['true_state'], df['pred_state'], 'State Confusion Matrix', 'confusion_state.png')


#  5. 置信度阈值曲线 
thresholds = np.arange(0.5, 0.96, 0.05)   # 0.50, 0.55, ..., 0.95
joint_acc_th = []

for th in thresholds:
    # 同时过滤物体和状态的置信度
    mask = (df['conf_obj'] >= th) & (df['conf_state'] >= th)
    if mask.sum() == 0:
        joint_acc_th.append(np.nan)
        continue
    sub = df[mask]
    correct = ( (sub['true_obj'] == sub['pred_obj']) & 
                (sub['true_state'] == sub['pred_state']) )
    joint_acc_th.append(correct.mean())



# 画图
plt.figure(figsize=(8,5))
plt.plot(thresholds, joint_acc_th, 'o-', color='darkred', linewidth=2, markersize=8)
plt.ylim([0, 1.05])
plt.xlabel('Confidence Threshold (both obj & state >= th)')
plt.ylabel('Joint Accuracy')
plt.title('Joint Accuracy vs. Confidence Threshold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('threshold_curve.png', dpi=150)
plt.show()