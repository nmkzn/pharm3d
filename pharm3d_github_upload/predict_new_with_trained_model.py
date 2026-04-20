import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, n_dims):
        super(MyModel, self).__init__()
        self.weight_tensor = nn.Parameter(torch.randn(n_dims))

    def forward(self, featTensor):
        hadamard_prod = torch.mul(featTensor, self.weight_tensor)
        readout = torch.mean(hadamard_prod, dim=-1)
        return readout


def load_pickle_tensor(path):
    with open(path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float32)
    return arr


def find_model_paths(train_dir):
    fixed_model = os.path.join(train_dir, "mymodel_fixed.pth")
    if os.path.exists(fixed_model):
        return [fixed_model], "fixed"

    fold_models = []
    for i in range(5):
        p = os.path.join(train_dir, f"mymodel_{i}fold.pth")
        if os.path.exists(p):
            fold_models.append(p)

    if len(fold_models) > 0:
        return fold_models, "fold_ensemble"

    raise FileNotFoundError(
        f"在 {train_dir} 里既没找到 mymodel_fixed.pth，也没找到 mymodel_0fold.pth~mymodel_4fold.pth"
    )


def score_one_model(model_path, X, device):
    n_dims = X.shape[1]
    model = MyModel(n_dims).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        scores = model(X_tensor).cpu().numpy().reshape(-1)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="训练模型所在目录，比如 ~/00011734250639")
    parser.add_argument("--new-dir", required=True, help="新数据所在目录，里面要有 tensor.pkl 和 state.csv")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认写到 new-dir")
    parser.add_argument("--group-col", default="ChemBL", help="分子分组列，默认 ChemBL")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES 列名，默认 smiles")
    args = parser.parse_args()

    train_dir = os.path.expanduser(args.train_dir)
    new_dir = os.path.expanduser(args.new_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else new_dir

    os.makedirs(out_dir, exist_ok=True)

    tensor_path = os.path.join(new_dir, "tensor.pkl")
    state_path = os.path.join(new_dir, "state.csv")

    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"缺少文件: {tensor_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"缺少文件: {state_path}")

    X = load_pickle_tensor(tensor_path)
    df = pd.read_csv(state_path)

    if len(df) != len(X):
        raise ValueError(
            f"state.csv 行数和 tensor.pkl 样本数不一致: len(df)={len(df)}, len(X)={len(X)}"
        )

    model_paths, model_mode = find_model_paths(train_dir)

    device = torch.device("cpu")

    all_scores = []
    for model_path in model_paths:
        scores = score_one_model(model_path, X, device)
        all_scores.append(scores)

    all_scores = np.vstack(all_scores)  # shape: [n_models, n_samples]
    score_mean = all_scores.mean(axis=0)
    score_std = all_scores.std(axis=0)

    row_df = df.copy()
    row_df.insert(0, "row_id", np.arange(len(row_df)))
    row_df["score_mean"] = score_mean
    row_df["score_std"] = score_std

    for i, model_path in enumerate(model_paths):
        row_df[f"score_model_{i}"] = all_scores[i]

    row_df = row_df.sort_values("score_mean", ascending=False).reset_index(drop=True)

    row_out = os.path.join(out_dir, "pred_rows.csv")
    row_df.to_csv(row_out, index=False, encoding="utf-8")

    group_keys = []
    if args.group_col in row_df.columns:
        group_keys.append(args.group_col)
    if args.smiles_col in row_df.columns and args.smiles_col not in group_keys:
        group_keys.append(args.smiles_col)

    mol_out = None
    if len(group_keys) > 0:
        idx_best = row_df.groupby(group_keys, dropna=False)["score_mean"].idxmax()
        best_df = row_df.loc[idx_best, group_keys + ["row_id", "score_mean"]].rename(
            columns={
                "row_id": "best_row_id",
                "score_mean": "max_score"
            }
        )

        agg_df = (
            row_df.groupby(group_keys, dropna=False)
            .agg(
                n_rows=("row_id", "count"),
                mean_score=("score_mean", "mean"),
                std_score=("score_mean", "std"),
                min_score=("score_mean", "min"),
            )
            .reset_index()
        )

        mol_df = agg_df.merge(best_df, on=group_keys, how="left")

        if "states" in row_df.columns:
            label_df = (
                row_df.groupby(group_keys, dropna=False)["states"]
                .agg(lambda s: s.iloc[0])
                .reset_index()
            )
            mol_df = mol_df.merge(label_df, on=group_keys, how="left")

        mol_df = mol_df.sort_values("max_score", ascending=False).reset_index(drop=True)
        mol_out = os.path.join(out_dir, "pred_molecules.csv")
        mol_df.to_csv(mol_out, index=False, encoding="utf-8")
    else:
        mol_df = None

    summary = {
        "train_dir": train_dir,
        "new_dir": new_dir,
        "out_dir": out_dir,
        "model_mode": model_mode,
        "model_paths": model_paths,
        "n_models": len(model_paths),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "row_prediction_file": row_out,
        "molecule_prediction_file": mol_out,
        "group_keys": group_keys,
    }

    summary_out = os.path.join(out_dir, "pred_summary.json")
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n已生成文件：")
    print(row_out)
    if mol_out:
        print(mol_out)
    print(summary_out)


if __name__ == "__main__":
    main()
