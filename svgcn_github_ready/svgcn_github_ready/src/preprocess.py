import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

TARGET_COL = "학교폭력_빈도수"

DROP_AFTER_AGG = [
    "자체_23(1)", "학폭_23(1)", "자체_23(2)", "학폭_23(2)",
    "자체_24(1)", "학폭_24(1)", "자체_24(2)", "학폭_24(2)",
    "발생여부", "학폭_25(1)", "학폭_25(2)", "자체_25(1)", "자체_25(2)",
]

TEXT_COLS = ["SCHUL_NM", "구"]


def load_school_data(path: str) -> pd.DataFrame:
    """Load school-level data from an Excel file."""
    return pd.read_excel(path)


def build_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate school-violence columns and remove non-numeric columns."""
    df = df.copy()

    df["학교폭력_23_1"] = df["자체_23(1)"] + df["학폭_23(1)"]
    df["학교폭력_23_2"] = df["자체_23(2)"] + df["학폭_23(2)"]
    df["학교폭력_24_1"] = df["자체_24(1)"] + df["학폭_24(1)"]
    df["학교폭력_24_2"] = df["자체_24(2)"] + df["학폭_24(2)"]
    df[TARGET_COL] = (
        df["학폭_25(1)"] + df["학폭_25(2)"] + df["자체_25(1)"] + df["자체_25(2)"]
    )

    drop_cols = [c for c in TEXT_COLS + DROP_AFTER_AGG if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def make_tensors(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Create scaled feature tensor and log-transformed target tensor."""
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x = torch.tensor(X_scaled, dtype=torch.float)

    y_raw = df[target_col].astype(float).values
    y_log = np.log1p(y_raw)
    y = torch.tensor(y_log, dtype=torch.float).view(-1, 1)

    return x, y, X_scaled, feature_cols, scaler
