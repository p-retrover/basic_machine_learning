# **Roadmap: MLP ( 2 hidden layers)**

To implement a Multi-Layer Perceptron (MLP) from scratch, we will need to build the following components manually:

### **1. Data Flattening**

Unlike the CNN we are currently training, a basic MLP requires 1D input vectors. For CIFAR-10, we will need to flatten the `(3, 32, 32)` (3 is for colors R, G, B) images into a single vector of size `3072`.

### **2. Weight Initialization**

Since we won't have "pretrained" weights, the initialization strategy is critical.

* **He Initialization**: Ideal for layers using the **ReLU** activation.
* **Xavier Initialization**: Best for layers using **Sigmoid** or **Tanh**.

### **3. The Activation Functions**

We will implement the forward and backward (derivative) versions of:

* **ReLU**: $f(x) = max(0,x)$
* **Sigmoid**: $f(x) = \frac{1}{1+e^{-x}}$
* **Softmax**: For the output layer to get class probabilities.

### **4. Backpropagation**

Instead of calling `loss.backward()`, we will manually apply the **Chain Rule** to compute gradients:

1. **Output Error**: Calculate the gradient of the Cross-Entropy loss.
2. **Hidden Layer Error**: Propagate the error backward through the weights and activation derivatives.
3. **Weight Update**: Apply the gradients to update $W$ and $b$ using Stochastic Gradient Descent.

---

## **CNN vs MLP**

This is a fundamental shift in how the network "sees" the world. In the **ResNet-18 (CNN)** we just finished, the model looked at the image as a 3D block. In our from scratch implementation of MLP, we're going to turn that block into a long, flat line.

### **1. What is "Flattening"?**

Imagine you have a Rubik's Cube (the image).

* A **CNN** looks at the cube, recognizing patterns on the faces, edges, and corners while keeping their relative positions.
* **Flattening** for an **MLP** is like taking that cube, smashing it flat, and laying out every single individual color sticker in one long row on the floor.

For CIFAR-10:

* Your image is $32 \times 32$ pixels.
* Each pixel has 3 colors (Red, Green, Blue).
* **The Math**: $32 \times 32 \times 3 = 3072$ total values.
* Our MLP will have **3072 input neurons**, each receiving the intensity of exactly one color at one specific pixel coordinate.

### **2. The Difference: Spatial Awareness**

| Feature | MLP (Multi-Layer Perceptron) | CNN (Convolutional Neural Network) |
| --- | --- | --- |
| **Input Type** | 1D Vector (Flattened) | 3D Tensor (Width, Height, Channels) |
| **Spatial Memory** | **None.** If you shift the pixels by 1, the MLP thinks it's a totally different image. | **High.** It recognizes a "cat ear" whether it's in the top-left or bottom-right corner. |
| **Connectivity** | **Fully Connected.** Every pixel connects to every neuron in the next layer. | **Local Connectivity.** A neuron only "looks" at a small patch of pixels at a time. |
| **Parameters** | Explodes quickly. A deep MLP for high-res images would require billions of weights. | Efficient. Weights (filters) are reused across the entire image. |

### **3. Why CNNs beat MLPs for Images**

In an MLP, the network has no idea that the pixel at `(1,1)` is next to the pixel at `(1,2)`. To the MLP, they are just "Feature #1" and "Feature #2."

A **CNN** uses **Convolutional Filters** (kernels) that slide across the image. These filters are mathematically designed to detect local patterns:

1. **Early Layers**: Detect simple edges and lines.
2. **Middle Layers**: Combine lines into shapes (circles, squares).
3. **Late Layers**: Combine shapes into features (eyes, wheels).

>## **Note**
>
>The idea that layers go from "simple" to "complex" isn't just a guess; it's something researchers have literally "seen" by peering into the brains of these models.
>
>### **1. Visualization Techniques**
>
>These layers we're using are essentially high-dimensional filters. Researchers use a few specific methods to "see" what a filter is looking for:
>
>* **Activation Maximization:** We take a specific neuron in the network and use gradient ascent to generate an image that makes that neuron fire the most.
>* In **Layer 1**, the "perfect" images are always grids of lines or blocks of color.
>* In **Layer 10**, the "perfect" images look like textures or eyes.
>
>* **Deconvolutional Networks (DeconvNet):** This technique "reverses" the flow, taking an internal activation and projecting it back into the pixel space to show exactly which pixels triggered that specific filter.
>
>### **2. Receptive Field"**
>
>The reason layers must start simple and get more complex comes down to the **Receptive Field**.
>
>1. **Early Layers (Small Receptive Field):** A  filter in the first layer only sees a  patch of pixels. Mathematically, it's impossible to "see" a whole cat ear in a  space. The only things that fit in that tiny window are gradients (changes in color) and edges.
>2. **Deeper Layers (Global Receptive Field):** As you go deeper, each neuron in Layer 3 is "looking" at a patch of Layer 2, which was "looking" at Layer 1. Through this stacking, a single neuron in a deep layer might actually represent a  area of the original image—enough space to hold a complex shape like a wheel.
>
>### **3. Bio-Mimicry**
>
>This architecture wasn't invented out of nowhere. It’s based on the **Human Visual Cortex**.
>In the 1960s, researchers Hubel and Wiesel (who won a Nobel Prize for this) discovered that the "primary visual cortex" (V1) in animals contains "simple cells" that only fire when they see lines at specific angles. Modern CNNs are a mathematical approximation of this biological structure.
>
>### **4. Why this matters**
>
>In our from scratch implementation of MLP, we don't have these filters. Every neuron in our first hidden layer is connected to **every** pixel.
>
>* **The Problem:** Because the MLP has no "sliding filter," it doesn't have that biological edge-detection bias. It has to learn *from scratch* that the pixel at `(5,5)` is related to the pixel at `(5,6)`.
>* **The Consequence:** This is why MLPs are so much harder to train for images; they are mathematically "blind" to the spatial structure that CNNs take for granted.
> So,
> "While the CNN benefits from translation invariance and hierarchical feature extraction due to its convolutional kernels, the MLP must treat the image as an unstructured vector, leading to a loss of spatial hierarchy and significantly higher computational cost for lower accuracy."

### **Potential Insight**

An MLP is **"Spatially Invariant."** This is a fancy way of saying it’s blind to geometry. This explains why our ResNet (CNN) hit **76% accuracy** easily, while our from-scratch MLP will likely struggle to pass **40-50%**—it’s fighting an uphill battle because it has to "re-learn" that pixels next to each other are related.
