# 📊 BreastCancer-DataMining-ML

## **Project Overview**
This project explores **Data Mining Techniques** using **Python** and **Jupyter Notebooks** on the **Wisconsin Breast Cancer Diagnostic Dataset**. The study applies **supervised learning, dimensionality reduction, clustering, and feature selection** to derive insights and improve classification accuracy.

---

## **📌 Phase 1: Supervised Learning & Dimensionality Reduction**
### **1️⃣ Implementing Simple Classifiers**
- **Decision Tree Classifier**
  - Computes **entropy** and **information gain** for optimal feature splits.
  - Builds a **recursive decision tree** for classification.
  - Uses **stratified 10-fold cross-validation** for evaluation.
  
- **Naive Bayes Classifier**
  - Assumes **Gaussian distributions** for features.
  - Calculates class probabilities using **Bayes’ Theorem**.
  - Predicts based on the **highest probability class**.

✅ **Performance Comparison:**  
- Models are evaluated using the **F1-score**.
- A **bar chart visualization** compares performance across folds.

---

### **2️⃣ Dimensionality Reduction Using Singular Value Decomposition (SVD)**
- Applied **SVD** to reduce dataset dimensionality.
- Trained **Decision Tree** and **Naive Bayes** classifiers on transformed data.
- Analyzed **F1-scores across different dimensions**.
- **Visualization:** Plotted **F1-score trends** to identify optimal dimensions.

---

## **📌 Phase 2: Unsupervised Learning & Feature Selection**
### **3️⃣ Feature Selection Using Randomization-Based Ranking**
- Ranked features based on importance in classification tasks.
- Applied **Decision Tree** and **Random Forest** for ranking.
- **Comparison:** Evaluated model accuracy with and without feature selection.

---

### **4️⃣ Clustering Algorithms for Data Segmentation**
✅ **Implemented the following clustering techniques:**
- **k-Means Clustering** 🔹 (For partitioning data into k groups)
- **DBSCAN** 🔹 (Density-based spatial clustering)
- **Spectral Clustering** 🔹 (Graph-based clustering)
- **Isolation Forest** 🔹 (For anomaly detection)

📈 **Evaluation Metrics:**
- **Silhouette Score:** Assessed cluster quality.
- **Cluster Visualization:** Plotted clustering results.

---

## **🛠️ Installation**
To run the project, install dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## **🚀 Running the Project**
To execute the analysis:

```bash
jupyter notebook
```

Open **DMT-project-phase1.ipynb** and **DMT-project-phase2.ipynb** to explore the techniques.

---

## **📊 Results & Insights**
- **Supervised Learning:** Decision Trees performed better with **higher F1 scores**.
- **Dimensionality Reduction:** SVD improved model performance with reduced dimensions.
- **Feature Selection:** Improved classification accuracy by removing irrelevant features.
- **Clustering:** k-Means performed best based on **Silhouette Score**.

---

## **📚 References**
- **Dataset:** [Kaggle - Wisconsin Breast Cancer Diagnostic Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

---

## **📢 Conclusion**
This project successfully implements **various data mining techniques**, including:
- **Supervised Learning**
- **Dimensionality Reduction**
- **Feature Selection**
- **Clustering**  
The insights gained can enhance **cancer diagnosis** and **medical data analysis**.

🚀 **Thank you for exploring this project!**
