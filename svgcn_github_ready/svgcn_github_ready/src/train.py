import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Allow running as: python src/train.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.graph import make_knn_edges_from_scaled_features
from src.model import GCNRegressor
from src.preprocess import TARGET_COL, build_target_and_features, load_school_data, make_tensors


def add_masks(data, random_state: int = 42):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=random_state)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_state)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def evaluate_regression(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = torch.expm1(out)
        pred = torch.clamp(pred, min=0)

        y_true_log = data.y[mask].cpu().numpy().reshape(-1)
        y_pred = pred[mask].cpu().numpy().reshape(-1)
        y_true = np.expm1(y_true_log)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def to_risk_level_array(y):
    y = np.array(y).reshape(-1)
    levels = []
    for v in y:
        if v <= 1:
            levels.append(0)  # low risk
        elif v <= 4:
            levels.append(1)  # middle risk
        else:
            levels.append(2)  # high risk
    return np.array(levels)


def evaluate_risk_classification(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = torch.expm1(out)
        pred = torch.clamp(pred, min=0)

        y_true_log = data.y[mask].cpu().numpy().reshape(-1)
        y_pred = pred[mask].cpu().numpy().reshape(-1)
        y_true = np.expm1(y_true_log)

        true_level = to_risk_level_array(y_true)
        pred_level = to_risk_level_array(y_pred)

        acc = accuracy_score(true_level, pred_level)
        cm = confusion_matrix(true_level, pred_level, labels=[0, 1, 2])
        report = classification_report(
            true_level,
            pred_level,
            labels=[0, 1, 2],
            target_names=["저위험", "중위험", "고위험"],
            zero_division=0,
        )

    return acc, cm, report


def main(args):
    raw_df = load_school_data(args.data_path)
    df = build_target_and_features(raw_df)
    x, y, X_scaled, feature_cols, _ = make_tensors(df, target_col=TARGET_COL)

    edge_index = make_knn_edges_from_scaled_features(X_scaled, k=args.k)
    data = Data(x=x, edge_index=edge_index, y=y)
    data = add_masks(data, random_state=args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCNRegressor(
        input_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.L1Loss()

    print(f"Device: {device}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Features: {len(feature_cols)}")
    print(f"Edges: {data.edge_index.shape[1]}")
    print(f"Train/Val/Test: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            train_mae, _, _ = evaluate_regression(model, data, data.train_mask)
            val_mae, val_rmse, val_r2 = evaluate_regression(model, data, data.val_mask)
            print(
                f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f}"
            )

    test_mae, test_rmse, test_r2 = evaluate_regression(model, data, data.test_mask)
    risk_acc, risk_cm, risk_report = evaluate_risk_classification(model, data, data.test_mask)

    print("\n[Test Result]")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Risk Accuracy: {risk_acc:.4f}")
    print("\nConfusion Matrix:")
    print(risk_cm)
    print("\nClassification Report:")
    print(risk_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the school data Excel file")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    main(parser.parse_args())
