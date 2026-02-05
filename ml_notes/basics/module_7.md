# **Module 7: Ensemble Methods**

Ensemble learning is a meta-approach to machine learning that seeks better predictive performance by combining the predictions from multiple models. The underlying mathematical principle is that a "committee" of diverse models will, on average, produce less error than any single constituent model.

---

## **7.1 The Wisdom of the Crowd: Why Ensembles Work**

In ensemble learning, we combine several "weak learners" (models that perform slightly better than random guessing) to create a "strong learner."

* **Bias-Variance Perspective:** Ensembles work by either reducing the **variance** (via Bagging) or the **bias** (via Boosting) of the base models.
* **Diversification:** For an ensemble to be effective, the base models must be diverse—meaning they should make different types of errors.

---

## **7.2 Bagging (Bootstrap Aggregating)**

Bagging aims to reduce **variance** (overfitting) by training multiple versions of the same model on different subsets of the data and averaging their results.

**The Process:**

1. **Bootstrapping:** Create $N$ new training sets by sampling from the original dataset **with replacement**.
2. **Parallel Training:** Train $N$ independent models (usually deep Decision Trees) on these $N$ bootstrap samples.
3. **Aggregation:**

* *Regression:* Average the predictions of all $N$ models.
* *Classification:* Take a majority vote among the models.

**Random Forest:**
The most famous Bagging algorithm. It improves upon standard Bagging by adding **feature randomness**. When splitting a node in a Decision Tree, it only considers a random subset of the total features. This further decorrelates the trees, making the ensemble more robust.

---

## **7.3 Boosting**

Boosting aims to reduce **bias** (underfitting) by training models **sequentially**. Each new model attempts to correct the errors made by the previous models.

**The Logic:**
Unlike Bagging, where models are trained in parallel, Boosting models are dependent on the predecessors. Data points that were misclassified by the first model are given "higher priority" or "higher weight" in the next model.

**Key Algorithms:**

* **AdaBoost (Adaptive Boosting):** Increases the weight of misclassified samples. The final prediction is a weighted sum of all models, where more accurate models have a higher say.
* **Gradient Boosting (GBM):** Instead of adjusting weights, it trains the new model on the **residual errors** (the difference between the actual and predicted values) of the previous model using Gradient Descent.
* **XGBoost (Extreme Gradient Boosting):** An optimized version of GBM that includes $L_1$ and $L_2$ regularization, handling of missing values, and parallel processing. It is the industry standard for tabular data competitions.

---

## **7.4 Stacking (Stacked Generalization)**

Stacking involves training a "Meta-Model" to learn how to best combine the predictions of several different base models.

1. **Base Layer:** Train multiple different models (e.g., an SVM, a k-NN, and a Random Forest).
2. **Meta-Feature Creation:** Pass the training data through these models to get their predictions.
3. **Meta-Layer:** Use these predictions as **input features** for a final model (often a simple Logistic or Linear Regression) that outputs the final prediction.

---

## **7.5 Summary Comparison**

| Feature | Bagging (Random Forest) | Boosting (XGBoost/AdaBoost) |
| --- | --- | --- |
| **Goal** | Reduce Variance (Overfitting) | Reduce Bias (Underfitting) |
| **Model Order** | Parallel | Sequential |
| **Sample Choice** | Random Bootstrap | Focus on hard-to-predict samples |
| **Complexity** | Simple to tune | High sensitivity to hyperparameters |

---
