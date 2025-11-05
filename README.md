# Machine Learning Unsupervised Models for Anomaly Detection in Credit Card Transactions  
*(Used in: “A Hybrid Anomaly Detection Framework Combining only Unsupervised Learning”)*

---

## 📚 Table of Contents
1. [Dataset Description for Fraud Detection](#🧾-dataset-description-for-fraud-detection)
2. [Anomaly Detection Methods (Unsupervised)](#🔬-anomaly-detection-methods)
   - [Isolation Forest](#1️⃣-isolation-forest)
   - [One-Class SVM](#2️⃣-one-class-svm)
   - [Local Outlier Factor (LOF)](#3️⃣-local-outlier-factor-lof)
   - [Autoencoder (Neural Network)](#4️⃣-autoencoder-neural-network)
3. [Example Usage](#🧪-example-usage-for-isolation-forest)
4. [Summary Results](#🚀-summary-results)
    - [Metric Definitions](#🧠-metric-definitions)
    - [Results Table](#📊-results-table)
    - [Interpretation](#📈-interpretation)
5. [Dependencies](#⚙️-dependencies)
6. [Author](#👤-author)

---

## 🧾 Dataset Description for Fraud Detection 

### 📘 Title
**Credit Card Transactions Dataset for Fraud Detection**  
*(Used in: A Hybrid Anomaly Detection Framework Combining Supervised and Unsupervised Learning)*

### 🔍 Description
This dataset, commonly known as **`creditcard.csv`**, contains anonymized credit card transactions made by European cardholders in **September 2013**.  
It includes **284,807 transactions**, of which **492 are fraudulent**.

Due to confidentiality, most features have been transformed using **Principal Component Analysis (PCA)** only **`Time`** and **`Amount`** remain in their original form.


>The study proposes an ensemble approach combining **unsupervised methods** (Autoencoder, Isolation Forest, Local Outlier Factor).

### 🧩 Key Features
| Feature | Description |
|----------|-------------|
| **V1–V28** | Numerical features obtained from PCA transformation. |
| **Time** | Seconds elapsed between each transaction and the first one. |
| **Amount** | Transaction amount. |
| **Class** | Target variable (1 = fraud, 0 = normal). |
| **Total Samples** | 284,807 transactions. |
| **Fraud Cases** | 492 (≈0.17%). |
| **Nature** | Highly imbalanced dataset typical for fraud detection. |

---

## 🔬 Anomaly Detection Methods

### 🧠 Overview
The project applies four **unsupervised anomaly detection** techniques to detect fraudulent transactions.  
Each method models the data differently, offering complementary perspectives on what constitutes an anomaly.

| Method | Type | Main Idea | Output |
|--------|------|------------|--------|
| **Isolation Forest** | Ensemble | Randomly isolates points; few splits = anomaly | Anomaly score |
| **One-Class SVM** | Kernel-based | Learns region around normal data | Decision function |
| **Local Outlier Factor (LOF)** | Density-based | Low local density = anomaly | Negative outlier factor |
| **Autoencoder** | Neural Network | High reconstruction error = anomaly | Reconstruction loss |

---
> **Definition of an anomaly:** 
>- These observations are **rare** and
>- Possess values that are **highly distinct** from the majority.

---
### 1️⃣ Isolation Forest

#### 🧩 Theory
**Isolation Forest** is based on a **forest** of Isolation trees that isolates observations by randomly selecting features and split values. The idea behind this method is that anomalies **are easier to isolate** (they require fewer random splits) while normal points need more.

#### 💻 Import
```python
from sklearn.ensemble import IsolationForest
```

#### ⚙️ Key Hyperparameters
| Parameter | Description |
|------------|-------------|
| `n_estimators` | Number of trees in the ensemble (default: 100). |
| `max_samples` | Number of samples drawn to train each tree. |
| `contamination` | Expected proportion of outliers (e.g., 0.01 = 1%). |
| `random_state` | Controls randomness for reproducibility. |

---

### 2️⃣ One-Class SVM

#### 🧩 Theory
The **One-Class SVM** is a boundary-based algorithm that learns the region containing the majority of “normal” data points.
It maps the data into a high-dimensional feature space using a kernel function and tries **to separate normal** data from the origin with **the maximum possible margin**.
Observations that fall outside this learned region are classified as anomalies.
This method is particularly effective when normal data form a compact cluster, but it can be sensitive to feature scaling and kernel parameter choices.

#### 💻 Import
```python
from sklearn.svm import OneClassSVM
```

#### ⚙️ Key Hyperparameters
| Parameter | Description |
|------------|-------------|
| `kernel` | Defines the boundary shape (`'rbf'` is common for non-linear cases). |
| `nu` | Upper bound on outlier fraction (typical 0.01–0.1). |
| `gamma` | Kernel coefficient; higher = more sensitive boundary. |

---

### 3️⃣ Local Outlier Factor (LOF)

#### 🧩 Theory
**Local Outlier Factor** (LOF) is a **density-based approach** that compares the **local density** of each data point to that of its **neighbors**.
If a point’s local density is much lower than that of its neighbors, it is considered an outlier.
LOF does not assume any global distribution of the data. Instead, it evaluates anomalies relative to **their local environment**, making it well-suited for datasets with varying densities.
A lower density ratio indicates that the sample lies in a sparse region, typical of anomalies.

#### 💻 Import
```python
from sklearn.neighbors import LocalOutlierFactor
```

#### ⚙️ Key Hyperparameters
| Parameter | Description |
|------------|-------------|
| `n_neighbors` | Number of neighbors to compute local density (default: 20). |
| `contamination` | Expected proportion of outliers in the dataset. |
| `novelty` | Set to `True` to enable predictions on unseen data (test set). |

---

### 4️⃣ Autoencoder (Neural Network)

#### 🧩 Theory
An Autoencoder is a type of **neural network** designed to learn efficient representations of data **through reconstruction**.
It consists of two main parts:
- **the encoder**, which compresses the input into a smaller latent representation, and
- **the decoder**, which reconstructs the original input from that latent space.

When trained only on normal data, the Autoencoder learns to accurately reconstruct normal patterns.
Anomalies, being unseen or rare, yield larger reconstruction errors, which are used as an anomaly score.
This method captures nonlinear relationships in the data and is powerful for complex feature interactions, though it typically requires more computation and careful hyperparameter tuning.

#### 💻 Import
```python
import torch
from torch import nn
```

#### ⚙️ Key Hyperparameters
| Parameter | Description |
|------------|-------------|
| `encoding_dim` | Latent vector size. Smaller = stronger compression. |
| `learning_rate` | Step size for optimization (1e-4–1e-3 typical). |
| `batch_size` | Number of samples per training batch (e.g., 64). |
| `epochs` | Number of full passes through the data. |

---

## 🧪 Example Usage for Isolation Forest

```python
# Example: Isolation Forest
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)
y_pred = model.predict(X_test)
# Output: -1 = anomaly, 1 = normal
```

## 🚀 Summary Results

#### 🧠 Metric Definitions

- **Precision** → proportion of detected anomalies that are true frauds.  
  High precision = few false positives.  
- **Recall** → proportion of all frauds that were detected.  
  High recall = few missed frauds.  
- **F1-score** → **harmonic mean** of Precision and Recall, balancing both aspects.  

#### 📊 Results Table

| Model               | Precision | Recall | F1-score |
|---------------------|-----------|--------|----------|
| Isolation Forest    | 0.25      | 0.69  | 0.32     |
| One-Class SVM       | 0.36      | **0.83**   | **0.51**     |
| Local Outlier Factor | 0.03      | 0.05   | 0.04   |
| Autoencoder         | **0.38**  | 0.75   | **0.51**     |

#### 📈 Interpretation

- **One-Class SVM** and **Autoencoder** achieved the **best balance** between precision and recall (F1 ≈ 0.51).  
  - *SVM* detects most frauds (high recall 0.83) but also triggers more false positives.  
  - *Autoencoder* is slightly more precise (0.38) with a comparable recall (0.75).  

- **Isolation Forest** performs moderately, while **LOF** struggles on this dataset due to its sensitivity to density variations.

---
>  **Remarks :** These results are **expected** in an **unsupervised learning** setting:  
>- Models are trained **without access to true fraud labels**, relying only on the statistical structure of the data.  
>- Fraudulent transactions are **extremely rare (≈0.17%)**, making them hard to distinguish from normal variations.  
>- Some anomalies may look similar to legitimate outliers, leading to false detections.


---
## ⚙️ Dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib tqdm
```

---

## 👤 Author

**Lucas Miedzyrzecki**  
MSc Statistics & Econometrics – Toulouse School of Economics  
[LinkedIn](https://www.linkedin.com/in/lucas-miedzyrzecki-760993262/)

---
