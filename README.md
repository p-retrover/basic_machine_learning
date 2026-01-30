# ML: From-Scratch Implementations & Deep Learning

A collection of machine learning algorithms built from first principles using NumPy, alongside deep learning implementations and fine-tuning experiments. This repository is designed to be a modular foundation for learning and future model development.

---

## Getting Started

### **Environment Setup**

This project uses **Poetry** for consistent dependency management. To set up your local environment:

1. **Install Dependencies**:
Run this from the root folder to install all necessary packages (NumPy, Pandas, Matplotlib, Scikit-Learn, PyTorch):

```bash
poetry install

```

1. **Launch Workspace**:

```bash
poetry run jupyter lab

```

### **Running Online**

For quick testing without local setup, you can upload the `.ipynb` files directly to **Google Colab**.

---

## Dataset Management

To keep the repository clean and lightweight, raw data files are excluded via `.gitignore`.

**To test the implementations:**

1. Navigate to the `data_note.md` file within the specific project directory, you'll get the instructions there.
2. Download the datasets from the provided links.
3. Create a `data/` folder in that project's directory to keep the downloaded datasets.

---

## Featured Modules

### **1. K-Nearest Neighbors (KNN)**

* **Implementation**: Built from scratch to support Euclidean and Manhattan distance metrics.
* **Visualization**: Incorporates **Principal Component Analysis (PCA)** to visualize 784-dimensional Fashion-MNIST data in a 2D decision boundary map.

### **2. Logistic Regression**

* **Implementation**: Features a numerically stable Sigmoid function to handle high-variance data and Z-score normalization for feature scaling.
* **Analysis**: Focuses on the "First Principles" of gradient descent and weight optimization.

### **3. Deep Learning & Fine-Tuning**

* **Focus**: Leveraging state-of-the-art architectures (like ResNet) through transfer learning to solve complex high-level classification tasks.

---

## Future Roadmap

* [ ] Addition of Support Vector Machines (SVM) from scratch.
* [ ] Implementation of Neural Network backpropagation using only NumPy.
* [ ] Expansion into Natural Language Processing (NLP) with Transformer fine-tuning.

---
