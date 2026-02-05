# **Module 5: Model Evaluation & Selection**

Now that we have covered how models learn, we must mathematically evaluate how well they generalize to new data.

## **5.1 Validation Strategies**

To avoid optimistic bias, we split data into three sets:

1. **Training Set:** Used to update model weights.
2. **Validation Set:** Used to tune hyperparameters (e.g., $k$ in k-NN, $\lambda$ in Lasso).
3. **Test Set:** Used only once at the very end to provide an unbiased evaluation.

**k-Fold Cross-Validation:**
The data is split into $k$ equal segments. The model is trained $k$ times, each time using a different segment as the validation set and the remaining $k-1$ segments as training data. The final score is the average of all $k$ iterations.

---

## **5.2 The Bias-Variance Tradeoff**

The total error of a model can be decomposed into:

1. **Bias:** Error from erroneous assumptions in the learning algorithm. High bias leads to **Underfitting**.
2. **Variance:** Error from sensitivity to small fluctuations in the training set. High variance leads to **Overfitting**.
3. **Irreducible Error:** Noise inherent in the data itself.
$$\text{Total Error}=\text{Bias}^2+\text{Variance}+\text{Irreducible Error}$$

---

## **5.3 Performance Metrics**

**A. Regression Metrics:**

* **Mean Squared Error :** $\mathrm{MSE}=\frac{1}{m}\sum_{i=1}^{m}\left( y^{(i)} - \hat{y}^{(i)} \right)^2$. Punishes large errors heavily.
* **Mean Absolute Error :** $\mathrm{MAE}=\frac{1}{m}\sum_{i=1}^{m}\left| y^{(i)} - \hat{y}^{(i)} \right|$. More robust to outliers.
* **Coefficient of Determination ($R^2$):** Measures the proportion of variance in $y$ explained by $X$. $R^2=1$ is a perfect fit.

**B. Classification Metrics (The Confusion Matrix):**
A  table for binary classification:

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

* $\text{Accuracy}=\frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}}$. Can be misleading in imbalanced datasets.
* $\text{Precision}=\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}$ . "Of all predicted positives, how many were correct?" (Focus on minimizing False Alarms).
* $\text{Recall}=\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$ . "Of all actual positives, how many did we catch?" (Focus on minimizing Misses).
* $\text{F1-score}=2 \cdot\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ The harmonic mean of Precision and Recall. .

**C. ROC-AUC Curve:**
The **Receiver Operating Characteristic** curve plots the True Positive Rate (Recall) vs. the False Positive Rate at various threshold settings. The **Area Under the Curve (AUC)** measures the overall ability of the classifier to distinguish between classes.

---
