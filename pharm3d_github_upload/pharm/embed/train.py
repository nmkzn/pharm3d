import json
import os
import gc
import pickle
import random
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import write_log, restoreAbsoluteCoord, unravel, drawPoint


class MyModel(nn.Module):
    def __init__(self, n_dims: int):
        super(MyModel, self).__init__()
        self.weight_tensor = nn.Parameter(torch.randn(n_dims))

    def forward(self, featTensor: torch.Tensor) -> torch.Tensor:
        hadamard_prod = torch.mul(featTensor, self.weight_tensor)
        readout = torch.mean(hadamard_prod, dim=-1)
        return readout

    def reset_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data.uniform_(-0.1, 0.1)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_torch_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_divide(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def evaluate_binary(outputs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    preds = (outputs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = labels.numel()
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total),
    }


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, device=torch.device("cpu")):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    outputs_all = []
    labels_all = []

    with torch.set_grad_enabled(is_train):
        for src, label in loader:
            src = src.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.float)

            if is_train:
                optimizer.zero_grad()

            output = model(src)
            loss = criterion(output, label)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * label.size(0)
            outputs_all.append(output.detach().cpu())
            labels_all.append(label.detach().cpu())

    outputs_all = torch.cat(outputs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    mean_loss = running_loss / len(loader.dataset)
    metrics = evaluate_binary(outputs_all, labels_all, threshold=0.5)
    metrics["loss"] = float(mean_loss)
    return metrics


def save_weight_table(dirname, filename, min_x, max_x, min_y, max_y, min_z, max_z, weight_tensor):
    indicesAttPd = pd.read_csv(os.path.join(dirname, "ind_att_pd.csv"))
    weight_np = weight_tensor.detach().cpu().numpy()

    indicesAttWtPd = pd.DataFrame(
        data={
            "gridNo": indicesAttPd["gridNo"].tolist(),
            "featNo": indicesAttPd["featNo"].tolist(),
            "weight": weight_np,
        }
    )

    x_grid_num = int(max_x - min_x)
    y_grid_num = int(max_y - min_y)
    z_grid_num = int(max_z - min_z)
    indicesAttWtPd["grid_x"], indicesAttWtPd["grid_y"], indicesAttWtPd["grid_z"] = zip(
        *indicesAttWtPd["gridNo"].apply(unravel, args=(x_grid_num, y_grid_num, z_grid_num))
    )
    out_path = os.path.join(dirname, filename)
    indicesAttWtPd.to_csv(out_path, index=False)
    return out_path


def train_model(dirname, min_x, max_x, min_y, max_y, min_z, max_z, model, train_loader, val_loader, criterion, optimizer,
                fold=0, patience=500, max_epochs=10000, device=torch.device("cpu"), save_prefix=""):
    best_val_loss = float("inf")
    current_patience = 0
    best_state_dict = None
    history = []

    loss_filename = f"{save_prefix}Fold{fold}_loss.txt" if save_prefix else f"Fold{fold}_loss.txt"
    metrics_filename = f"{save_prefix}Fold{fold}_metrics.csv" if save_prefix else f"Fold{fold}_metrics.csv"

    with open(os.path.join(dirname, loss_filename), "w", encoding="utf-8") as f:
        f.write("")

    for epoch in range(max_epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)

        write_log(
            dirname,
            (
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            ),
        )
        with open(os.path.join(dirname, loss_filename), "a", encoding="utf-8") as f:
            f.write(f"{val_metrics['loss']}\n")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            current_patience = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            current_patience += 1
            if current_patience >= patience:
                write_log(dirname, "Early stopping")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    pd.DataFrame(history).to_csv(os.path.join(dirname, metrics_filename), index=False)
    weight_table_path = save_weight_table(
        dirname=dirname,
        filename=(f"{save_prefix}indicesAttWtPd_{fold}fold.csv" if save_prefix else f"indicesAttWtPd_{fold}fold.csv"),
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
        weight_tensor=model.state_dict()["weight_tensor"],
    )
    clear_torch_cache()
    return model, history, weight_table_path


def make_loader(X, y, indices, batch_size=16, shuffle=False, device=torch.device("cpu")):
    X_tensor = torch.tensor(X[indices], dtype=torch.float, device=device)
    y_tensor = torch.tensor(y[indices], dtype=torch.float, device=device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_fixed_split_files(dirname: str, labels: np.ndarray, seed: int = 42):
    idx = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=0.10,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1 / 9,
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    np.save(os.path.join(dirname, "train_idx.npy"), train_idx)
    np.save(os.path.join(dirname, "val_idx.npy"), val_idx)
    np.save(os.path.join(dirname, "test_idx.npy"), test_idx)
    return train_idx, val_idx, test_idx


def load_or_create_fixed_splits(dirname: str, labels: np.ndarray, seed: int = 42):
    train_path = os.path.join(dirname, "train_idx.npy")
    val_path = os.path.join(dirname, "val_idx.npy")
    test_path = os.path.join(dirname, "test_idx.npy")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_idx = np.load(train_path)
        val_idx = np.load(val_path)
        test_idx = np.load(test_path)
        source = "existing"
    else:
        train_idx, val_idx, test_idx = create_fixed_split_files(dirname, labels, seed=seed)
        source = "generated"

    split_info = {
        "split_source": source,
        "seed": seed,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_label_counts": pd.Series(labels[train_idx]).value_counts().sort_index().to_dict(),
        "val_label_counts": pd.Series(labels[val_idx]).value_counts().sort_index().to_dict(),
        "test_label_counts": pd.Series(labels[test_idx]).value_counts().sort_index().to_dict(),
    }
    save_json(os.path.join(dirname, "split_info.json"), split_info)
    return train_idx, val_idx, test_idx


def k_fold_cv(dirname, min_x, max_x, min_y, max_y, min_z, max_z, model, X, y, criterion, optimizer, k=5,
              batch_size=16, max_epochs=10000, patience=500, device=torch.device("cpu")):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    summary_rows = []
    with open(os.path.join(dirname, "accuracy"), "w", encoding="utf-8") as f:
        f.write("fold,accuracy\n")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        clear_torch_cache()
        write_log(dirname, f"Fold {fold + 1}/{k}")

        train_loader = make_loader(X, y, train_index, batch_size=batch_size, shuffle=True, device=device)
        val_loader = make_loader(X, y, val_index, batch_size=batch_size, shuffle=False, device=device)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model, _, _ = train_model(
            dirname=dirname,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            patience=patience,
            max_epochs=max_epochs,
            fold=fold,
            device=device,
        )

        model_path = os.path.join(dirname, f"mymodel_{fold}fold.pth")
        torch.save(model.state_dict(), model_path)

        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        summary = {"fold": fold, **val_metrics}
        summary_rows.append(summary)

        write_log(dirname, f"Fold {fold} validation accuracy: {val_metrics['accuracy']:.6f}")
        with open(os.path.join(dirname, "accuracy"), "a", encoding="utf-8") as f:
            f.write(f"{fold},{val_metrics['accuracy']}\n")

    pd.DataFrame(summary_rows).to_csv(os.path.join(dirname, "kfold_summary.csv"), index=False)
    return os.path.join(dirname, "accuracy")


def train_fixed_split(dirname, min_x, max_x, min_y, max_y, min_z, max_z, X, y, criterion,
                      batch_size=16, max_epochs=10000, patience=500, device=torch.device("cpu")):
    train_idx, val_idx, test_idx = load_or_create_fixed_splits(dirname, y, seed=42)
    train_loader = make_loader(X, y, train_idx, batch_size=batch_size, shuffle=True, device=device)
    val_loader = make_loader(X, y, val_idx, batch_size=batch_size, shuffle=False, device=device)
    test_loader = make_loader(X, y, test_idx, batch_size=batch_size, shuffle=False, device=device)

    model = MyModel(X.shape[1]).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model, _, _ = train_model(
        dirname=dirname,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        patience=patience,
        max_epochs=max_epochs,
        fold=0,
        device=device,
        save_prefix="fixed_",
    )

    model_path = os.path.join(dirname, "mymodel_fixed.pth")
    torch.save(model.state_dict(), model_path)

    train_metrics = run_epoch(model, train_loader, criterion, optimizer=None, device=device)
    val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=device)

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "model_path": model_path,
    }
    save_json(os.path.join(dirname, "fixed_split_metrics.json"), metrics)

    with open(os.path.join(dirname, "fixed_split_accuracy.csv"), "w", encoding="utf-8") as f:
        f.write("split,accuracy,precision,recall,specificity,f1,loss\n")
        for split_name, split_metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            f.write(
                f"{split_name},{split_metrics['accuracy']},{split_metrics['precision']},{split_metrics['recall']},"
                f"{split_metrics['specificity']},{split_metrics['f1']},{split_metrics['loss']}\n"
            )

    write_log(dirname, f"Fixed split train accuracy: {train_metrics['accuracy']:.6f}")
    write_log(dirname, f"Fixed split val accuracy: {val_metrics['accuracy']:.6f}")
    write_log(dirname, f"Fixed split test accuracy: {test_metrics['accuracy']:.6f}")
    return os.path.join(dirname, "fixed_split_accuracy.csv")


def pharm_train(dirname, min_x, max_x, min_y, max_y, min_z, max_z, tensor_path, state_path):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(dirname, f"Using device: {device}")
    sim = pickle.load(open(tensor_path, "rb"))
    df = pd.read_csv(state_path)
    src_data = np.array(sim)
    label_data = df["states"].to_numpy(dtype=float)

    criterion = nn.MSELoss()

    train_idx_path = os.path.join(dirname, "train_idx.npy")
    val_idx_path = os.path.join(dirname, "val_idx.npy")
    test_idx_path = os.path.join(dirname, "test_idx.npy")
    has_fixed_split = os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and os.path.exists(test_idx_path)

    if has_fixed_split:
        write_log(dirname, "Detected fixed split files. Run strict train/val/test training.")
        return train_fixed_split(
            dirname=dirname,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            X=src_data,
            y=label_data,
            criterion=criterion,
            batch_size=16,
            max_epochs=10000,
            patience=500,
            device=device,
        )

    write_log(dirname, "No fixed split files found. Fall back to 5-fold cross validation.")
    model = MyModel(src_data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return k_fold_cv(
        dirname=dirname,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
        model=model,
        X=src_data,
        y=label_data,
        criterion=criterion,
        optimizer=optimizer,
        k=5,
        batch_size=16,
        patience=500,
        max_epochs=10000,
        device=device,
    )


def drawPoint(chain, row, row_index):
    atom_types = {0: "N", 1: "C", 2: "S", 3: "MG", 4: "O", 5: "AL"}
    atom_colors = {
        0: (0.75, 0.75, 0.75),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 0.0, 1.0),
        3: (1.0, 1.0, 0.0),
        4: (1.0, 1.0, 1.0),
        5: (0.5, 0.5, 0.5),
    }
    residue = Residue.Residue((" ", row_index, " "), "ALA", " ")
    chain.add(residue)
    atom = Atom.Atom(
        atom_types[row["featNo"]],
        [row["abso_x"], row["abso_y"], row["abso_z"]],
        1.0,
        1.0,
        " ",
        atom_types[row["featNo"]],
        row_index,
        " ",
    )
    color = atom_colors[row["featNo"]]
    temp_factor = 100 * (color[0] * 255) + 10 * (color[1] * 255) + (color[2] * 255)
    atom.set_bfactor(temp_factor)
    residue.add(atom)


def restore_mlp(dirname, min_x, max_x, min_y, max_y, min_z, max_z, ind_att_pd_path, model_path, nfeat=20):
    indicesAttPd = pd.read_csv(ind_att_pd_path)
    model = torch.load(os.path.join(dirname, model_path), map_location="cpu")
    trained_weight_tensor = model["weight_tensor"]
    trained_weight_tensor = trained_weight_tensor.cpu().detach().numpy()
    indicesAttWtPd = pd.DataFrame(
        data={
            "gridNo": indicesAttPd["gridNo"].tolist(),
            "featNo": indicesAttPd["featNo"].tolist(),
            "weight": trained_weight_tensor,
        }
    )

    x_grid_num = int(max_x - min_x)
    y_grid_num = int(max_y - min_y)
    z_grid_num = int(max_z - min_z)
    indicesAttWtPd["grid_x"], indicesAttWtPd["grid_y"], indicesAttWtPd["grid_z"] = zip(
        *indicesAttWtPd["gridNo"].apply(unravel, args=(x_grid_num, y_grid_num, z_grid_num))
    )

    step = 1
    min_x, min_y, min_z = int(min_x), int(min_y), int(min_z)
    indicesAttWtPd["abso_x"], indicesAttWtPd["abso_y"], indicesAttWtPd["abso_z"] = zip(
        *indicesAttWtPd.apply(restoreAbsoluteCoord, args=(min_x, min_y, min_z, step), axis=1)
    )

    indicesAttWtPdSort = indicesAttWtPd.sort_values(by=["weight"], ascending=False).reset_index(drop=True)
    indicesAttWtPdSort.to_csv(os.path.join(dirname, "sortedAttWt.csv"), index=False)

    featNo = nfeat
    dfdraw = indicesAttWtPdSort[:featNo].reset_index(drop=False)
    structure = Structure.Structure(os.path.join(dirname, "new_pdb"))
    model_obj = Model.Model(0)
    structure.add(model_obj)
    chain = Chain.Chain("A")
    model_obj.add(chain)

    dfdraw.iloc[:10000].apply(lambda row: drawPoint(chain, row, row_index=row.name), axis=1)
    io = PDBIO()
    io.set_structure(structure)
    io.save(os.path.join(dirname, f"feat{featNo}_{model_path}.pdb"))
    return os.path.join(dirname, f"feat{featNo}_{model_path}.pdb")


def plot_fig(dirname, loss_path, title, loss_fig):
    y_axis_data = np.loadtxt(os.path.join(dirname, loss_path))
    if np.isscalar(y_axis_data):
        y_axis_data = np.array([y_axis_data])
    x_axis_data = np.arange(len(y_axis_data))
    plt.figure()
    plt.plot(x_axis_data, y_axis_data, ",", alpha=0.5, linewidth=0.1)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, loss_fig))
    plt.close()
