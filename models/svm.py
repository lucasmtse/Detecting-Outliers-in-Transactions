from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import linear_model
import matplotlib.pyplot as plt

def train_one_class_svm(
    x_train,
    nu: float = 0.01,
    kernel: str = "rbf",
    gamma: str = 'scale',
    tol: float = 0.001
) -> OneClassSVM:
    """
    Train a One-Class SVM model.

    Parameters
    ----------
    x_train : array-like (n_samples, n_features)
        Training features.
    nu : float
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    kernel : str
        Specifies the kernel type to be used in the algorithm.
    gamma : str
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    Returns
    -------
    OneClassSVM
        Trained model.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, tol=tol)
    model.fit(x_train)
    return model

def predict_one_class_svm(model: OneClassSVM, x_test) -> np.ndarray:
    """
    Predict using the trained One-Class SVM model.

    Returns
    -------
    np.ndarray
        Predictions in {1 (normal), -1 (anomaly)}.
    """
    return model.predict(x_test)

#SGD One Class SVM
def train_sgd_one_class_svm(
    x_train,
    nu: float = 0.01,
    max_iter: int = 1000,
    tol: float = 0.001,
    random_state: Optional[int] = None
) -> linear_model.SGDOneClassSVM:
    """
    Train a SGD One-Class SVM model.

    Parameters
    ----------
    x_train : array-like (n_samples, n_features)
        Training features.
    nu : float
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    max_iter : int
        The maximum number of passes over the training data.
    tol : float
        The stopping criterion.
    random_state : Optional[int]
        Seed for reproducibility.

    Returns
    -------
    SGDOneClassSVM
        Trained model.
    """
    model = linear_model.SGDOneClassSVM(nu=nu, max_iter=max_iter, tol=tol, random_state=random_state)
    model.fit(x_train)
    return model

def predict_sgd_one_class_svm(model: linear_model.SGDOneClassSVM, x_test) -> np.ndarray:
    """
    Predict using the trained SGD One-Class SVM model.

    Returns
    -------
    np.ndarray
        Predictions in {1 (normal), -1 (anomaly)}.
    """
    return model.predict(x_test)
