# **Module 3: Supervised Learning - Classification**

Classification is a supervised learning task where the target variable $y$ is **discrete/categorical**. The objective is to learn a decision boundary that separates different classes in the feature space.

---

## **3.1 Logistic Regression**

Despite its name, Logistic Regression is a **classification** algorithm. It is used to estimate the probability that an instance belongs to a particular class ($y \in \{0,1\}$).

**The Sigmoid Function:**
To map any real-valued number into a probability range $[0,1]$, we use the Logistic (Sigmoid) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The hypothesis is defined as:

$$h_\theta(x) = \sigma(\theta^T x)$$
**Decision Boundary:**
The model predicts class 1 if $h_\theta(x) \ge 0.5$ (which occurs when $\theta^T x \ge 0$) and class 0 otherwise.

**Loss Function (Binary Cross-Entropy):**
We cannot use Mean Squared Error for Logistic Regression because the resulting cost function would be non-convex (difficult to optimize). Instead, we use the **Log Loss**:

$$J(\theta) =-\frac{1}{m}\sum_{i=1}^{m}\left[y^(i) log ( h_\theta(x^{(i)}))+(1 - y^{(i)}) \log ( 1 - _\theta(x^{(i)}) )\right]
$$
---

## **3.2 k-Nearest Neighbors (k-NN)**

A non-parametric, "lazy" learning algorithm. It does not learn a fixed set of parameters $\theta$; instead, it stores the entire training dataset.

* **Logic:** To classify a new point, the algorithm finds the $k$ closest points (neighbors) in the training set based on a distance metric (usually **Euclidean distance**).
$$d(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}$$
* **Prediction:** The new point is assigned the class that is most common among its $k$ neighbors. majority vote: $\hat{y} = \operatorname{mode}\{y_1, y_2, \dots, y_k\}$
* **Hyperparameter $k$:** A small $k$ makes the model sensitive to noise (overfitting), while a large $k$ makes the decision boundary too smooth (underfitting).

---

## **3.3 Probabilistic Models: Naive Bayes**

Naive Bayes is based on **Bayes' Theorem** and assumes that all features are **independent** given the class label (the "naive" assumption : $P(x_1, \dots, x_d \mid y)=
\prod_{i=1}^{d} P(x_i \mid y)$).

**Bayes' Rule:**

$$P(y \mid x_1, \dots, x_d)=
\frac{P(x_1, \dots, x_d \mid y)\, P(y)}
{P(x_1, \dots, x_d)}$$

The model selects the class  that maximizes the posterior probability $P(y \mid X)$. It is highly efficient and works well for high-dimensional data like text classification.

---

## **3.4 Decision Trees**

A model that splits the data into subsets based on feature values, creating a tree-like structure of decisions.

* **Splitting Criteria:** At each node, the model chooses the feature that best separates the classes. This is measured using **Information Gain** (via **Entropy**) or the **Gini Impurity**.
* **Entropy:** A measure of disorder. $H(S) = -\sum_{i} p_i \log_2 p_i$.
* **Pros:** Highly interpretable; handles both numerical and categorical data.
* **Cons:** Prone to **overfitting** if the tree becomes too deep.

---

## **3.5 Support Vector Machines (SVM)**

SVM aims to find the **Maximum Margin Hyperplane** that separates two classes.

* **Hard Margin:** Works if the data is perfectly linearly separable.
* **Soft Margin:** Allows some misclassifications to handle noisy data (controlled by a parameter $C$).
* **The Kernel Trick:** For non-linearly separable data, SVM uses "Kernels" (like **RBF** or **Polynomial**) to project the data into a higher-dimensional space where a linear separation becomes possible.

---
