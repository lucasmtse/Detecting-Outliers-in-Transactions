# Detecting Outliers in Transactions

### About dataset

Title: Credit Card Transactions Dataset for Fraud Detection (Used in: A Hybrid Anomaly Detection Framework Combining Supervised and Unsupervised Learning)

Description:

This dataset, commonly known as creditcard.csv, contains anonymized credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions, with 492 labeled as fraudulent. Due to confidentiality constraints, features have been transformed using PCA, except for 'Time' and 'Amount'.

This dataset was used in the research article titled "A Hybrid Anomaly Detection Framework Combining Supervised and Unsupervised Learning for Credit Card Fraud Detection". The study proposes an ensemble model integrating techniques such as Autoencoders, Isolation Forest, Local Outlier Factor, and supervised classifiers including XGBoost and Random Forest, aiming to improve the detection of rare fraudulent patterns while maintaining efficiency and scalability.

Key Features:

    30 numerical input features (V1–V28, Time, Amount)
    Class label indicating fraud (1) or normal (0)
    Imbalanced class distribution typical in real-world fraud detection