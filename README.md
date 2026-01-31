# ML: From-Scratch Implementations & Deep Learning

A collection of machine learning algorithms built from first principles using NumPy, alongside deep learning implementations and fine-tuning experiments. This repository is designed to be a modular foundation for my **learning** and future model development.

---

## Getting Started

### **Environment Setup**

This project uses **Poetry** for consistent dependency management. To set up your local environment:

1. **Install Dependencies**:

```bash
poetry install

```

2. **Register the Kernel**:
Run this command to link the virtual environment to your Jupyter interface:

```bash
poetry run python -m ipykernel install --user --name basic-ml-env --display-name "Python (Basic ML)"

```

3. **Launch Jupyter**:

```bash
poetry run jupyter lab

```

4. **Select Kernel**:
Once in Jupyter Lab, open any `.ipynb` file and select **"Python (Basic ML)"** from the kernel dropdown menu (top right).

> **Note:** To remove this kernel later, run:
>
> ```bash
> poetry run jupyter kernelspec uninstall basic-ml-env
> ```
>
> Also if you want to have virtual environments inside each of your project folder, you can set
>
> ```bash
> poetry config virtualenvs.in-project true
> ```
>
>If the project already has a Poetry env elsewhere:
>
>```bash
> poetry env remove --all
> poetry install
> ```
>
> This creates the env inside `.venv/`

#### **PyTorch Configuration: CPU vs. GPU**

By default, this repository is configured to use the **CPU-only version of PyTorch**.

**Why CPU?**

* **Compatibility**: Ensures the project runs out-of-the-box on most systems..
* **Disk Space**: The CPU binary is significantly smaller (~200MB) compared to the CUDA-enabled version (~5GB+), preventing unnecessary bloat in your `.cache` or `.venv`.
* **Efficiency**: Since we use **Transfer Learning** (Freezing the backbone), training the final layer is fast enough on modern CPUs.

**How to switch to CUDA (If you have an NVIDIA GPU):**
If you want to leverage hardware acceleration, you need to modify the `pyproject.toml` file:

1. **Change the Source**: Remove the `pytorch-cpu` source and the specific source constraints for `torch` and `torchvision`.
2. **Update Dependencies**:

```toml
# remove this:
[[tool.poetry.source]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"

# and this:
[tool.poetry.dependencies]
torch = { version = "^2.10.0", source = "pytorch-cpu" }
torchvision = { version = "^0.25.0", source = "pytorch-cpu" }

```

3. **Re-install**:

```bash
poetry lock --no-update
poetry install

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

### **AI Disclosure**

This project was developed with the assistance of **Gemini**, and other LLMs.

**How AI was utilized:**

* **Architectural Guidance:** Collaborating on the modular structure of the repository to ensure it remains scalable for future algorithm additions.
* **Debugging & Stability:** AI was instrumental in identifying and resolving numerical instability issues (e.g., handling `NaN` values during Gradient Descent and implementing a numerically stable Sigmoid function).
* **Feature Enhancement:** Assistance in implementing advanced visualization techniques, such as using **Principal Component Analysis (PCA)** to represent high-dimensional decision boundaries in 2D space.
* **Documentation:** Streamlining README and report formatting to ensure technical clarity for external reviewers.

---
