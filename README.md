# 🔍 PCA Lab – Dimensionality Reduction with Gaussian Data

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Technique](https://img.shields.io/badge/Technique-PCA-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 🧠 Overview

This project explores **Principal Component Analysis (PCA)** and its applications in reducing the dimensionality of a correlated Gaussian dataset.

Implemented from scratch using NumPy, the notebook helps visualize PCA's effect on data structure, correlation, and variance capture. It's part of the **Machine Learning Summer 2023 Lab at NYU Paris**.

---

## 🎯 Goals

- Understand how PCA works from the ground up  
- Apply PCA on a 2D Gaussian distribution  
- Visualize data before and after PCA transformation  
- Verify decorrelation using the covariance matrix  

---

## 🛠️ Technologies

- Python 3  
- NumPy  
- Matplotlib  

---

## 🔧 Core Features

- Sampling multivariate Gaussian data  
- Manual PCA implementation via eigen-decomposition  
- Covariance analysis  
- Data transformation and visualization  

---

## 📁 Project Structure

```

pca-lab/
├── pca.ipynb                   # Jupyter notebook with PCA implementation and experiments
├── README.md                   # This file
└── output\_13\_0.png             # Optional output image from PCA visualization

````

---

## 🚀 Sample Code

```python
class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        cov_matrix = np.cov(X.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
        sorted_idx = np.flip(np.argsort(eigen_vals))
        self.eigenvectors = -1 * eigen_vecs[:, sorted_idx]
        self.eigenvector_subset = self.eigenvectors[:, :self.n_components]

    def transform(self, Z):
        return np.dot(Z, self.eigenvector_subset)
````

---

## 📊 Result Highlight

**Before PCA:**

Covariance Matrix:

```
[[3.18 2.64]
 [2.64 6.26]]
```

**After PCA:**

Covariance Matrix:

```
[[7.78 0.00]
 [0.00 1.67]]
```

---

## 👨‍🏫 Author

* Guillaume Staerman (Instructor)
* Notebook formatted and structured for GitHub by contributor

---

## 📬 Contact

For educational purposes only. Originally developed for coursework at **NYU Paris**.
