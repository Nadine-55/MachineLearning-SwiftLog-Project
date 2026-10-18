import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from swiftlog_config import FEATURES
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)

P = Path("../Data/Processed_Datasets")
FILES = {"delay": "delay_features.csv", "demand": "demand_features.csv",
         "customer": "customer_features.csv", "warehouse": "warehouse_features.csv"}

def load_problem(name):
    cfg = FEATURES[name]
    d = pd.read_csv(P / FILES[name])
    cols = cfg["num"] + cfg["cat"]
    sets = {s: (d.loc[d["split"] == s, cols], d.loc[d["split"] == s, cfg["target"]])
            for s in ["train", "val", "test"]}
    return cfg, sets

def make_preprocessor(cfg):
    return ColumnTransformer([
        ("num", StandardScaler(), cfg["num"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["cat"]),
    ])

def evaluate(task, y, pred, proba=None):
    if task == "classification":
        return dict(Accuracy=accuracy_score(y, pred),
                    Precision=precision_score(y, pred, zero_division=0),
                    Recall=recall_score(y, pred), F1=f1_score(y, pred),
                    ROC_AUC=roc_auc_score(y, proba))
    mse = mean_squared_error(y, pred)
    return dict(MAE=mean_absolute_error(y, pred), MSE=mse,
                RMSE=np.sqrt(mse), R2=r2_score(y, pred))

def train_months(name):
    d = pd.read_csv(P / FILES[name])
    return d.loc[d["split"] == "train", "month"].reset_index(drop=True)