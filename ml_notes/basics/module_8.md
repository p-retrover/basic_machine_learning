# **Module 8: Neural Networks & Deep Learning**

Neural Networks are a class of models inspired by the biological structure of the human brain. They consist of interconnected layers of "neurons" that process information through non-linear transformations.

---

## **8.1 The Perceptron: The Biological Inspiration**

The journey of neural networks began in 1958 with Frank Rosenblatt's **Perceptron**. It was designed as a mathematical model of a single biological neuron.

**The Model:**
A Perceptron takes a vector of inputs $x$, multiplies them by weights $w$, adds a bias $b$, and passes the result through a **Step Function**.
$$f(x) =
\begin{cases}
1, & w^T x + b > 0 \\
0, & \text{otherwise}
\end{cases}$$

**The XOR Problem (The Winter of AI):**
In 1969, Minsky and Papert proved that a single-layer perceptron could only solve **linearly separable** problems. It could model logic gates like `AND` and `OR`, but it fundamentally could not solve `XOR` (Exclusive OR), because `XOR` requires a non-linear decision boundary. This led to a decade of stagnation in the field known as the "AI Winter."

---

## **8.2 Multi-Layer Perceptron (MLP)**

To solve non-linear problems, we stack perceptrons into layers. This creates a **Feedforward Neural Network**.

* **Input Layer:** Receives the raw features.
* **Hidden Layers:** Intermediate layers that perform feature extraction. The "depth" of a network is determined by the number of hidden layers.
* **Output Layer:** Produces the final prediction (e.g., probabilities for classification).

**Activation Functions:**
Without non-linear activation functions, even a 100-layer network would mathematically collapse into a single linear transformation ($W_2 (W_1 x) = W_{\text{new}} x$).

* **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$. Squashes values to $[0,1]$.
* **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$. The industry standard; avoids the "vanishing gradient" problem in deep networks.
* **Tanh:** $\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$. Squashes values to $[-1,1]$.

---

## **8.3 The Forward Pass**

In an MLP, information flows from left to right. For a single hidden layer:

1. **Hidden Layer Activation:** $a^{(1)} = \sigma\!\left(W^{(1)} x + b^{(1)}\right)$
2. **Output Layer:** $\hat{y} = \operatorname{Softmax}\!\left(W^{(2)} a^{(1)} + b^{(2)}\right)$

---

## **8.4 Backpropagation: The Engine of Learning**

Backpropagation is the application of the **Chain Rule** from calculus to calculate the gradient of the loss function with respect to every weight in the network.

**The Mathematical Logic:**
To update a weight $w_{ij}$ in an early layer, we must calculate how a small change in that weight ripples through the network to change the final loss $L$.
$$\frac{\partial L}{\partial w_{ij}}=\frac{\partial L}{\partial \text{output}}\cdot\frac{\partial \text{output}}{\partial \text{hidden}}\cdot\frac{\partial \text{hidden}}{\partial w_{ij}}$$

**The Process:**

1. **Forward Pass:** Calculate the prediction and the total error.
2. **Backward Pass:** Starting from the output, calculate the "error signal" ($\delta$) for each neuron and propagate it backward.
3. **Update:** Use Gradient Descent to update the weights: $W := W - \alpha \nabla_W L$.

---

## **8.5 Deep Learning Challenges**

**A. Vanishing/Exploding Gradients:**
In very deep networks, as the gradient is multiplied by small numbers (like the derivative of Sigmoid) repeatedly during backprop, it can become nearly zero ("vanish"). The early layers stop learning. Conversely, if gradients are large, they can "explode," causing the weights to become `NaN`.

**B. Overfitting in Deep Networks:**
Because deep models have millions of parameters, they can easily memorize the training data.

* **Dropout:** Randomly "turning off" a percentage of neurons during training to force the network to learn redundant representations.
* **Batch Normalization:** Normalizing the activations of each layer to stabilize the learning process and allow for higher learning rates.

---
