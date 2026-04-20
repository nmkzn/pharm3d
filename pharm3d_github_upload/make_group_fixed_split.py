import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.expanduser("~/00011734250639")
STATE_PATH = os.path.join(BASE, "state.csv")

GROUP_COL = "ChemBL"
LABEL_COL = "states"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8

df = pd.read_csv(STATE_PATH)

if GROUP_COL not in df.columns:
    raise ValueError(f"缺少分组列: {GROUP_COL}")
if LABEL_COL not in df.columns:
    raise ValueError(f"缺少标签列: {LABEL_COL}")

df = df.dropna(subset=[GROUP_COL]).reset_index(drop=True)

group_label_nunique = df.groupby(GROUP_COL)[LABEL_COL].nunique()
bad_groups = group_label_nunique[group_label_nunique > 1]

if len(bad_groups) > 0:
    print("以下 group 内部标签不一致，不能直接做分组划分：")
    print(bad_groups.head(20))
    raise ValueError(f"共有 {len(bad_groups)} 个 group 内部标签不一致")

group_df = (
    df.groupby(GROUP_COL, as_index=False)
      .agg(label=(LABEL_COL, "first"), n_rows=(LABEL_COL, "size"))
)

groups = group_df[GROUP_COL].astype(str).to_numpy()
labels = group_df["label"].to_numpy()

train_groups, temp_groups, train_labels, temp_labels = train_test_split(
    groups,
    labels,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=labels,
)

val_portion_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_groups, test_groups, val_labels, test_labels = train_test_split(
    temp_groups,
    temp_labels,
    test_size=(1.0 - val_portion_in_temp),
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)

train_group_set = set(train_groups.tolist())
val_group_set = set(val_groups.tolist())
test_group_set = set(test_groups.tolist())

train_idx = df.index[df[GROUP_COL].astype(str).isin(train_group_set)].to_numpy()
val_idx   = df.index[df[GROUP_COL].astype(str).isin(val_group_set)].to_numpy()
test_idx  = df.index[df[GROUP_COL].astype(str).isin(test_group_set)].to_numpy()

np.save(os.path.join(BASE, "train_idx.npy"), train_idx)
np.save(os.path.join(BASE, "val_idx.npy"), val_idx)
np.save(os.path.join(BASE, "test_idx.npy"), test_idx)

def count_labels(idxs):
    sub = df.iloc[idxs][LABEL_COL].value_counts().sort_index()
    return {str(k): int(v) for k, v in sub.items()}

info = {
    "group_col": GROUP_COL,
    "label_col": LABEL_COL,
    "random_seed": RANDOM_SEED,
    "n_total_rows": int(len(df)),
    "n_total_groups": int(len(group_df)),
    "train_rows": int(len(train_idx)),
    "val_rows": int(len(val_idx)),
    "test_rows": int(len(test_idx)),
    "train_groups": int(len(train_group_set)),
    "val_groups": int(len(val_group_set)),
    "test_groups": int(len(test_group_set)),
    "train_label_counts": count_labels(train_idx),
    "val_label_counts": count_labels(val_idx),
    "test_label_counts": count_labels(test_idx),
    "train_val_group_overlap": int(len(train_group_set & val_group_set)),
    "train_test_group_overlap": int(len(train_group_set & test_group_set)),
    "val_test_group_overlap": int(len(val_group_set & test_group_set)),
}

with open(os.path.join(BASE, "split_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print("新的 group-level split 已保存到:")
print(os.path.join(BASE, "train_idx.npy"))
print(os.path.join(BASE, "val_idx.npy"))
print(os.path.join(BASE, "test_idx.npy"))
print(os.path.join(BASE, "split_info.json"))

print("\n=== split summary ===")
print(json.dumps(info, indent=2, ensure_ascii=False))
