# **Module 2: Supervised Learning - Regression**

Regression is a supervised learning task where the objective is to predict a **continuous scalar output** $y$ given an input vector $\mathbf{x} \in \mathbb{R}^d$. Mathematically, we aim to find a function $h : \mathcal{X} \rightarrow \mathcal{Y}$ such that the prediction error is minimized.

---

## **2.1 Linear Regression**

The simplest form of regression assumes a linear relationship between the input features and the target variable.

**The Hypothesis Function:**

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_d x_d$$

Using vector notation, we let $x_0 = 1$ (the intercept term):

$$h_\theta(x) = \sum_{i=0}^{d} \theta_i x_i$$

where $\theta$ is the parameter vector (weights).

**The Loss Function (Mean Squared Error):**
To measure how well $\theta$ fits the training data, we define the Cost Function $J(\theta)$ as the sum of squared residuals:

$$
J(\theta) =
\frac{1}{2m}
\sum_{i=1}^{m}
\left(
h_\theta(x^{(i)}) - y^{(i)}
\right)^2
$$

*Note: The factor of $\frac{1}{2}$ is included for mathematical convenience during differentiation.*

---

## **2.2 The Normal Equation**

For linear regression, there exists a closed-form solution to find the $\theta$ that minimizes $J(\theta)$ without using an iterative algorithm. This is derived by setting the gradient $\nabla_\theta J(\theta) = 0$.

The solution, known as the **Normal Equation**, is:

$$\theta = (X^T X)^{-1} X^T y$$

* **Pros:** No need to choose a learning rate; finds the global minimum in one step.
* **Cons:** Computationally expensive if $d$ is large (inverting a $(d\times d)$ matrix is $\mathcal{O}(d^3)$).

---

## **2.3 Gradient Descent for Regression**

When the feature count  is too high for matrix inversion, we use **Gradient Descent**, an iterative optimization algorithm.

**The Update Rule:**

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

For Linear Regression, the partial derivative simplifies to:

$$\theta_j :=
\theta_j -
\alpha
\frac{1}{m}
\sum_{i=1}^{m}
\left(
h_\theta(x^{(i)}) - y^{(i)}
\right)
x_j^{(i)}
$$
**Variations of Gradient Descent:**

1. **Batch Gradient Descent:** Uses all $m$ examples in every step (stable but slow).
2. **Stochastic Gradient Descent (SGD):** Uses one example per step (fast but noisy).
3. **Mini-batch Gradient Descent:** Uses a small subset (e.g., 32 or 64) per step (the industry standard).

---

## **2.4 Polynomial Regression**

If the data does not follow a straight line, we can still use linear regression by transforming the features into a polynomial space. For example, if we have a single feature , we can create a quadratic model:

$$h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2$$

Even though the relationship with  is non-linear, the relationship with the parameters $\theta$ remains **linear**, allowing us to use the same optimization techniques.

---

## **2.5 Regularization: Lasso and Ridge**

To prevent **overfitting** (where the model fits the training noise), we add a penalty term to the cost function to keep weights small.

**A. Ridge Regression ($L_2$ Regularization):**
Adds the squared magnitude of weights to the cost function.

$$J(\theta) = \mathrm{MSE}(\theta) + \lambda \sum_{j=1}^{d} \theta_j^2$$

*Effect:* Shrinks coefficients towards zero, reducing model variance.

**B. Lasso Regression ($L_1$ Regularization):**
Adds the absolute magnitude of weights.

$$J(\theta) = \mathrm{MSE}(\theta) + \lambda \sum_{j=1}^{d} \lvert \theta_j \rvert$$

*Effect:* Can force some coefficients to become **exactly zero**, effectively performing **feature selection**.

---
