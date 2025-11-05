# models/isolation_forest.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    x_train,
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 100
) -> IsolationForest:
    """
    Train an Isolation Forest model.

    Parameters
    ----------
    x_train : array-like (n_samples, n_features)
        Training features.
    contamination : float
        Proportion of outliers in the data.
    random_state : int
        Seed for reproducibility.
    n_estimators : int
        Number of base estimators in the ensemble.
    Returns
    -------
    IsolationForest
        Trained model.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state, 
        n_estimators=n_estimators
    )
    model.fit(x_train)
    return model

def predict_isolation_forest(model: IsolationForest, x_test) -> np.ndarray:
    """
    Predict using the trained Isolation Forest model.

    Returns
    -------
    np.ndarray
        Predictions in {-1 (anomaly), 1 (normal)}.
    """
    return model.predict(x_test)
