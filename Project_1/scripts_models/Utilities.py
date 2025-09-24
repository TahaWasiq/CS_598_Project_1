import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

@dataclass
class DataBundle:
    """Container for split datasets of a single horizon."""
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any
    feature_cols: List[str]
    name: str  # e.g., "H1" or "H7"

@dataclass
class Metrics:
    """Common evaluation metrics for a model on one dataset."""
    mse: float
    rmse: float
    mae: float
    r2: float

    def as_row(self) -> Dict[str, float]:
        return {"MSE": self.mse, "RMSE": self.rmse, "MAE": self.mae, "R2": self.r2}