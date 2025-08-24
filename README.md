# NEDA-KAN: Enhancing Academic Regulation Classification with NED-Augmented Kolmogorov–Arnold Networks

This repository contains the implementation for the paper:

**"NEDA-KAN: Enhancing Academic Regulation Classification with NED-Augmented Kolmogorov–Arnold Networks"**  
 _CICT, Can Tho University, Viet Nam_

## Abstract

Academic regulation documents are notoriously difficult to classify due to ambiguous, context-dependent phrasing and domain-specific vocabulary. This project integrates **Named Entity Disambiguation (NED)** into the preprocessing pipeline to enhance semantic clarity before classification. We use **Kolmogorov–Arnold Networks (KANs)** and compare their performance with classical machine learning models and FastText.

## Evaluation

### Results of Classification using different dictionaries

| ID                                      | Models                  | Parameters                  | Acc (%) | Prec (%) | Rec (%) | F1 (%) | Training (s) | Testing (s) |
| --------------------------------------- | ----------------------- | --------------------------- | ------- | -------- | ------- | ------ | ------------ | ----------- |
| **Raw dataset**                         |                         |                             |         |          |         |        |              |             |
| 1                                       | Multinomial Naive Bayes | α = 0.1                     | 89.66   | 89.62    | 89.66   | 89.44  | 0.01         | 0.006       |
| 2                                       | KNN                     | k = 101                     | 86.59   | 87.13    | 86.59   | 85.90  | 0.01         | 0.071       |
| 3                                       | Logistic Regression     | C=10, tol=0.0001            | 94.97   | 95.02    | 94.97   | 94.88  | 0.42         | 0.004       |
| 4                                       | SVM                     | C=1, tol=0.0001, loss=hinge | 93.58   | 93.72    | 93.58   | 93.42  | 0.06         | 0.003       |
| 5                                       | Random Forest           | n_estimators=300            | 96.37   | 96.58    | 96.37   | 96.20  | 0.35         | 0.041       |
| 6                                       | FastText                | epochs=100, Ngrams=3        | 96.08   | 96.14    | 96.08   | 96.02  | 9.1616       | 0.052       |
| **Specialized word dictionary dataset** |                         |                             |         |          |         |        |              |             |
| 1                                       | Multinomial Naive Bayes | α = 0.1                     | 89.94   | 89.94    | 89.94   | 89.76  | 0.01         | 0.004       |
| 2                                       | KNN                     | k = 101                     | 86.31   | 86.90    | 86.31   | 85.60  | 0.01         | 0.067       |
| 3                                       | Logistic Regression     | C=10, tol=0.0001            | 95.25   | 95.29    | 95.25   | 95.15  | 0.42         | 0.006       |
| 4                                       | SVM                     | C=1, tol=0.0001, loss=hinge | 94.13   | 94.25    | 94.13   | 93.98  | 0.05         | 0.003       |
| 5                                       | Random Forest           | n_estimators=300            | 96.09   | 96.30    | 96.09   | 95.92  | 0.46         | 0.050       |
| 6                                       | FastText                | epochs=100, Ngrams=3        | 96.22   | 96.27    | 96.22   | 96.17  | 8.832        | 0.038       |
| **tudientv dictionary dataset**         |                         |                             |         |          |         |        |              |             |
| 1                                       | Multinomial Naive Bayes | α = 0.1                     | 90.78   | 90.94    | 90.78   | 90.65  | 0.01         | 0.003       |
| 2                                       | KNN                     | k = 101                     | 86.31   | 87.90    | 86.31   | 85.38  | 0.01         | 0.065       |
| 3                                       | Logistic Regression     | C=10, tol=0.0001            | 95.25   | 95.29    | 95.25   | 95.19  | 0.45         | 0.005       |
| 4                                       | SVM                     | C=1, tol=0.0001, loss=hinge | 94.41   | 94.49    | 94.41   | 94.37  | 0.36         | 0.014       |
| 5                                       | Random Forest           | n_estimators=300            | 96.09   | 96.36    | 96.09   | 95.94  | 0.36         | 0.054       |
| 6                                       | FastText                | epochs=100, Ngrams=3        | 97.06   | 97.09    | 97.06   | 97.03  | 10.075       | 0.042       |

### Results of KAN models using different dictionaries

| ID                                      | Models             | Architecture | Feature Type | Acc (%) | Prec (%) | Rec (%) | F1 (%) |
| --------------------------------------- | ------------------ | ------------ | ------------ | ------- | -------- | ------- | ------ | --- |
| **Raw dataset**                         |                    |              |              |         |          |         |        |     |
| 1                                       | KAN Deeper TF-IDF  | 265,128,64,6 | TF-IDF       | 97.48   | 97.84    | 97.21   | 97.48  |
| 2                                       | KAN Simple TF-IDF  | 128,64,6     | TF-IDF       | 97.34   | 97.73    | 97.08   | 97.32  |
| 3                                       | KAN Simple PhoBERT | 128,64,6     | PhoBERT      | 93.29   | 93.42    | 93.36   | 93.26  |
| 4                                       | KAN Wide PhoBERT   | 256,6        | PhoBERT      | 93.01   | 93.16    | 93.00   | 92.82  |
| **Specialized word dictionary dataset** |                    |              |              |         |          |         |        |     |
| 1                                       | KAN Deeper TF-IDF  | 265,128,64,6 | TF-IDF       | 97.90   | 98.00    | 97.68   | 97.79  |
| 2                                       | KAN Simple TF-IDF  | 128,64,6     | TF-IDF       | 97.62   | 98.10    | 97.28   | 97.60  |
| 3                                       | KAN Simple PhoBERT | 128,64,6     | PhoBERT      | 94.13   | 94.32    | 94.16   | 94.14  |
| 4                                       | KAN Wide PhoBERT   | 256,6        | PhoBERT      | 93.57   | 93.85    | 93.41   | 93.37  |
| **tudientv dictionary dataset**         |                    |              |              |         |          |         |        |     |
| 1                                       | KAN Deeper TF-IDF  | 265,128,64,6 | TF-IDF       | 95.24   | 95.61    | 95.18   | 95.38  |
| 2                                       | KAN Simple TF-IDF  | 128,64,6     | TF-IDF       | 94.97   | 95.16    | 95.11   | 95.13  |
| 3                                       | KAN Simple PhoBERT | 128,64,6     | PhoBERT      | 89.23   | 89.41    | 89.72   | 89.53  |
| 4                                       | KAN Wide PhoBERT   | 256,6        | PhoBERT      | 91.33   | 91.45    | 91.54   | 91.48  |

## Installation

```bash
pip install -r requirements.txt
```
