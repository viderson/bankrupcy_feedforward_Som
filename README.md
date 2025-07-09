# Bankruptcy Prediction for Polish Companies using Feedforward Network and SOM 🏦🧠

This project focuses on predicting the **bankruptcy of Polish companies** using two machine learning approaches:

- **Feedforward Neural Networks (FFNN)** – supervised classification
- **Self-Organizing Maps (SOM)** – unsupervised clustering and visualization

The project is based on real financial indicators sourced from EMIS (Emerging Markets Information Service).

---

## 📊 Dataset Overview

- **Source**: EMIS Platform (Emerging Markets Information Service)
- **Goal**: Predict whether a company will go bankrupt in a defined future window based on its financial indicators
- **Features**: Quantitative financial ratios
- **Target**: Binary label – `1` for bankruptcy, `0` for healthy

---

## 🧠 Models Used

### 🔹 Feedforward Neural Network
- Built using TensorFlow/Keras
- Hyperparameter tuning with `Optuna`
- Performance metrics: accuracy, F1-score, AUC
- Imbalance handled using `SMOTE` and `class_weight`

### 🔹 Self-Organizing Map (SOM)
- Used for dimensionality reduction and data clustering
- Visualization of risk zones and anomaly patterns

---

## 📁 Project Files

```
project/
├── polish_bankrupcy_feedforward_Som.ipynb     # Main notebook
└── README.md
```

---

## 🔧 Libraries Used

- `tensorflow`, `keras`, `optuna`, `minisom`
- `scikit-learn`, `imbalanced-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install ucimlrepo optuna ipywidgets tqdm minisom numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow seaborn
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook polish_bankrupcy_feedforward_Som.ipynb
   ```

3. Run cells to:
   - Load and preprocess financial data
   - Train and evaluate neural network
   - Visualize SOM clustering

---

## 🎯 Learning Outcomes

- Apply deep learning and unsupervised learning to real-world financial data
- Handle imbalanced binary classification
- Optimize models using Optuna
- Visualize high-dimensional data using SOM

---

## 📌 Credits

- Author: FW

---

