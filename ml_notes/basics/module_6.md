# **Module 6: Feature Engineering & Preprocessing**

Feature Engineering is the process of using domain knowledge to extract or transform raw data into features that make machine learning algorithms work. Preprocessing ensures that the data is mathematically compatible with the optimization algorithms (like Gradient Descent).

---

## **6.1 Feature Scaling**

Most ML algorithms calculate distances between data points or use gradients for optimization. If features have vastly different scales, the objective function becomes elongated, leading to slow convergence or biased results.

**A. Standardization ($Z$-score Normalization):**
Transforms the data to have a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$.
$$x' = \frac{x - \mu}{\sigma}$$

* **Use Case:** Preferred for algorithms that assume a Gaussian distribution (e.g., Linear Regression, Logistic Regression, SVMs, and PCA).

**B. Min-Max Scaling (Normalization):**
Rescales the data to a fixed range, typically $[0,1]$.
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

* **Use Case:** Necessary for algorithms that do not assume a specific distribution and are distance-based (e.g., k-NN) or for Neural Networks using certain activation functions.

---

## **6.2 Categorical Encoding**

Machine learning models are mathematical equations that require numerical input. Categorical data (strings, labels) must be converted into a numerical format.

**A. One-Hot Encoding:**
Creates a new binary column for each unique category. For a feature "Color" with values {Red, Green, Blue}, it creates three columns.

* **Red:** $[1,0,0]$
* **Green:** $[0,1,0]$
* **Blue:** $[0,0,1]$
* **Note:** This avoids implying a mathematical order between categories.

**B. Label Encoding:**
Assigns each category a unique integer (e.g., Red=0, Green=1, Blue=2).

* **Caution:** Only use this for **ordinal data** where the order matters (e.g., Low=0, Medium=1, High=2). Otherwise, the model may incorrectly assume $Blue>Red$.

---

## **6.3 Handling Missing Values**

Real-world datasets often have missing entries ($NaN$ or $Null$). We have two primary strategies:

1. **Deletion:**
* *Row Deletion:* Remove the entire observation. (Risky if the dataset is small).
* *Column Deletion:* Remove the feature entirely. (Risky if the feature is important).


2. **Imputation:**
* *Univariate:* Fill missing values using the **Mean**, **Median**, or **Mode** of that column.
* *Multivariate (Iterative):* Predict the missing value using other available features (e.g., using a k-NN Imputer).

---

## **6.4 Feature Selection Methods**

To avoid the **Curse of Dimensionality** and reduce overfitting, we select a subset of the most relevant features.

* **Filter Methods:** Use statistical measures to rank features independent of the ML model (e.g., Pearson Correlation, Chi-Square test).
* **Wrapper Methods:** Treat feature selection as a search problem. They train the model on different subsets and pick the best one (e.g., Recursive Feature Elimination).
* **Embedded Methods:** The selection happens *during* the model training (e.g., **Lasso Regression**, where $L_1$ penalty forces coefficients of unimportant features to zero).

---

## **6.5 Handling Outliers**

Outliers can significantly skew the mean and variance, leading to poor model performance.

* **Detection:** Using the **Z-score** (values $|z| > 3\sigma$) or the **Interquartile Range (IQR)**:
* *Lower Bound:* $Q_1 - 1.5 \cdot \mathrm{IQR}$
* *Upper Bound:* $Q_3 + 1.5 \cdot \mathrm{IQR}$


* **Treatment:** Trimming (removal) or Capping (Winsorization).

---
