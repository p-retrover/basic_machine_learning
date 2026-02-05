# **Module 1: Introduction to Machine Learning**

Machine Learning is a branch of computational statistics that utilizes algorithms to find patterns in data. Formally, a computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.

---

## **1.1 Learning Paradigms**

The "experience" $E$ provided to the model determines the learning paradigm.

* **Supervised Learning:** The dataset contains input features $X$ and their corresponding labels $Y$. The goal is to learn a mapping function $f : X \rightarrow Y$.
* *Classification:* $Y$ is a categorical/discrete value.
* *Regression:* $Y$ is a continuous/numerical value.


* **Unsupervised Learning:** The dataset contains only features $X$. The goal is to find hidden structures or patterns within the data.
* *Clustering:* Grouping data points into subsets based on similarity.
* *Dimensionality Reduction:* Projecting high-dimensional data into a lower-dimensional space while preserving variance.


* **Reinforcement Learning:** An agent learns to make a sequence of decisions in an environment to maximize a cumulative reward signal through trial and error.

---

## **1.2 Data Anatomy**

Machine Learning treats data as points in a high-dimensional vector space.

* **Feature Space ($\mathcal{X}$):** A -dimensional space where each data point is represented as a vector $\mathbf{x} \in \mathbb{R}^d$.
* **Label Space ($\mathcal{Y}$):** The set of all possible outcomes.
* **Training Set ($D_{\text{train}}$):** A subset of the data used to estimate the model parameters.
* **Test Set ($D_{\text{test}}$):** A separate subset used exclusively to evaluate the model's ability to **generalize** to unseen data.
* **Generalization:** The central objective of ML. It is the ability of an algorithm to perform accurately on new data drawn from the same underlying distribution as the training data.

---

## **1.3 Mathematical Foundations**

To build and optimize these models, three mathematical pillars are required:

**A. Linear Algebra**
Data is represented as **Tensors** (scalars, vectors, matrices). A single layer of a neural network is essentially a linear transformation followed by a non-linear activation:

$$y = \sigma(W\mathbf{x + b})$$

where $W$ is a weight matrix and $\mathbf{b}$ is a bias vector.

**B. Calculus**
The learning process involves minimizing an error function. We use the **Gradient ($\nabla$)**—a vector of partial derivatives—to find the direction of steepest descent.

$$\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
$$
**C. Probability and Statistics**
Models are often probabilistic. Instead of predicting a single value, we predict the **likelihood** of an outcome.

* **Maximum Likelihood Estimation (MLE):** A method of estimating the parameters of a probability distribution by maximizing a likelihood function.
* **Expectation:** Used to calculate the average performance over a distribution of data.

---

## **1.4 Summary of the ML Workflow**

1. **Data Collection:** Gathering the raw features.
2. **Preprocessing:** Handling missing values and scaling features.
3. **Model Selection:** Choosing an appropriate hypothesis class (e.g., Linear vs. Non-linear).
4. **Training (Optimization):** Minimizing a loss function using algorithms like Gradient Descent.
5. **Evaluation:** Measuring performance on the test set using specific metrics (Accuracy, MSE, etc.).

---
