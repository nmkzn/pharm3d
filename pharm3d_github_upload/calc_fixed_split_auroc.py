import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from pharm.embed.train import MyModel

BASE = os.path.expanduser("~/00011734250639")
TENSOR_PATH = os.path.join(BASE, "tensor.pkl")
STATE_PATH = os.path.join(BASE, "state.csv")
TEST_IDX_PATH = os.path.join(BASE, "test_idx.npy")
MODEL_PATH = os.path.join(BASE, "mymodel_fixed.pth")

# 1) 读取数据
X = np.asarray(pickle.load(open(TENSOR_PATH, "rb")), dtype=np.float32)
df = pd.read_csv(STATE_PATH)
y = df["states"].to_numpy(dtype=np.float32)

test_idx = np.load(TEST_IDX_PATH)

X_test = X[test_idx]
y_test = y[test_idx]

# 2) 加载模型
device = torch.device("cpu")
model = MyModel(X.shape[1]).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 3) 跑 test score
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_score = model(X_test_tensor).cpu().numpy().reshape(-1)

# 4) AUROC
auroc = roc_auc_score(y_test, y_score)

# 5) AUPRC（顺便一起算，论文也用了）
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_score)
auprc = auc(recall_curve, precision_curve)

# 6) ROC curve + Youden index
fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = roc_thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

# 7) 用论文提到的初始阈值 0.5 也算一遍分类指标
y_pred_05 = (y_score >= 0.5).astype(int)

acc_05 = accuracy_score(y_test, y_pred_05)
prec_05 = precision_score(y_test, y_pred_05, zero_division=0)
rec_05 = recall_score(y_test, y_pred_05, zero_division=0)
f1_05 = f1_score(y_test, y_pred_05, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_05).ravel()
spec_05 = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# 8) 用 Youden 最优阈值再算一遍分类指标
y_pred_best = (y_score >= best_threshold).astype(int)

acc_best = accuracy_score(y_test, y_pred_best)
prec_best = precision_score(y_test, y_pred_best, zero_division=0)
rec_best = recall_score(y_test, y_pred_best, zero_division=0)
f1_best = f1_score(y_test, y_pred_best, zero_division=0)

tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_best).ravel()
spec_best = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0

# 9) 保存逐样本结果，方便你以后画 ROC
pred_df = pd.DataFrame({
    "row_index": test_idx,
    "ChemBL": df.iloc[test_idx]["ChemBL"].values if "ChemBL" in df.columns else [""] * len(test_idx),
    "smiles": df.iloc[test_idx]["smiles"].values if "smiles" in df.columns else [""] * len(test_idx),
    "y_true": y_test,
    "y_score": y_score,
    "y_pred_0.5": y_pred_05,
    "y_pred_best": y_pred_best,
})
pred_df.to_csv(os.path.join(BASE, "fixed_split_test_predictions.csv"), index=False)

# 10) 保存 summary
summary = {
    "n_test": int(len(y_test)),
    "positive_test": int(np.sum(y_test == 1)),
    "negative_test": int(np.sum(y_test == 0)),
    "auroc": float(auroc),
    "auprc": float(auprc),
    "youden_best_threshold": float(best_threshold),
    "youden_best_tpr": float(best_tpr),
    "youden_best_fpr": float(best_fpr),
    "metrics_at_threshold_0.5": {
        "accuracy": float(acc_05),
        "precision": float(prec_05),
        "recall": float(rec_05),
        "specificity": float(spec_05),
        "f1": float(f1_05),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    },
    "metrics_at_youden_threshold": {
        "accuracy": float(acc_best),
        "precision": float(prec_best),
        "recall": float(rec_best),
        "specificity": float(spec_best),
        "f1": float(f1_best),
        "tp": int(tp2),
        "fp": int(fp2),
        "tn": int(tn2),
        "fn": int(fn2),
    }
}

with open(os.path.join(BASE, "fixed_split_auroc_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\n已保存:")
print(os.path.join(BASE, "fixed_split_test_predictions.csv"))
print(os.path.join(BASE, "fixed_split_auroc_summary.json"))
