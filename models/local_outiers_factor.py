# models/isolation_forest.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def train_local_outlier_factor(
    x_train,
    n_neighbors: int = 20,
    contamination: float = 0.01,
    novelty: bool = False
) -> LocalOutlierFactor:
    """
    Train a Local Outlier Factor model.

    Parameters
    ----------
    x_train : array-like (n_samples, n_features)
        Training features.
    n_neighbors : int
        Number of neighbors to use by default for kneighbors queries.
    contamination : float
        Proportion of outliers in the data.
    novelty : bool
        By default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). *
        Set novelty to True if you want to use LocalOutlierFactor for novelty detection. 
        In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; 
        and note that the results obtained this way may differ from the standard LOF results.
    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty)
    clf.fit(x_train)
    return clf

def predict_local_outliers_factor(model: LocalOutlierFactor, x_test) -> np.ndarray:
    """
    Predict using the trained Local Outlier Factor model.

    Returns
    -------
    np.ndarray
        Predictions in {-1 (anomaly), 1 (normal)}.
    """
    return model.fit_predict(x_test)