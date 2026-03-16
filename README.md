# ML: From-Scratch Implementations & Deep Learning

A collection of machine learning algorithms built from first principles using NumPy, alongside deep learning implementations and fine-tuning experiments. This repository is designed to be a modular foundation for my **learning** and future model development.
[Books and Resources](./ml_notes/resources.md)

---

## Getting Started

Download the repo

```bash
git clone https://github.com/p-retrover/basic_machine_learning && cd basic_machine_learning
```

### **Environment Setup**

Please make sure that you have Python3.14 installed on your system. If you're a linux/mac user, you can install from your package manager. If you're a windows user, you can either download it from microsoft store, or from a very nice package manager called scoop.

> To install scoop, open your terminal and run
>
>```bash
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
>```
>
> Then to install python3.14
>
>```bash
> scoop bucket add versions
> scoop install python314
>```
>
> Then explicitly tell poetry to use the correct python version.
>
>```bash
> poetry env use python314
>```
>
> Then proceed below.

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

## Dataset and Implementation Management

To keep the repository clean and lightweight, raw data files are excluded via `.gitignore`.

**To test the implementations:**

1. Navigate to the `README.md` file within the specific project directory, you'll get the instructions there.
2. Download the datasets from the provided links.
3. Create a `data/` folder in that project's directory to keep the downloaded datasets.

---

## Projects

| Project Name | Description | Link |
| :--- | :--- | :--- |
| **Fine-Tuning ResNet-18** | Transfer learning on CIFAR-10 with PyTorch. | [View Project](./fine_tuning_pretrained_model/) |
| **NN from Scratch** | 3-layer MLP implemented using pure NumPy. | [View Project](./nn_from_scratch/) |
| **KNN from Scratch** | Distance-based classification on Fashion-MNIST. | [View Project](./KNN_from_scratch/) |
| **Logistic Regression** | Binary classification using Gradient Descent. | [View Project](./logistic_regression/) |

---

## Future Roadmap

### **Classical Machine Learning**

* [ ] **Principal Component Analysis (PCA)** from scratch using Eigen-decomposition and Singular Value Decomposition (SVD).
* [ ] **Support Vector Machines (SVM)** implementation using the Dual Lagrangian formulation.
* [ ] **Gaussian Mixture Models (GMM)** and the Expectation-Maximization (EM) algorithm for soft clustering.
* [ ] **Decision Trees & Random Forests** using Gini Impurity and Information Gain metrics.

### **Deep Learning & Computer Vision**

* [x] Implementation of Neural Network backpropagation using only NumPy.
* [ ] Development of **CNN layers (Convolution, Pooling, Flatten)** in pure NumPy to understand spatial feature extraction.
* [ ] **Generative Adversarial Networks (GANs)**: Training a simple DCGAN to generate handwritten digits or CIFAR-like images.
* [ ] Implementation of **Optimization Algorithms** (Adam, RMSProp, and Adagrad) from scratch and comparing their convergence.

### **Natural Language Processing (NLP)**

* [ ] **Word Embeddings**: Implementing a Word2Vec (Skip-gram) model to visualize semantic vector spaces.
* [ ] **Attention Mechanism**: Building the Scaled Dot-Product Attention block from scratch (the "math" heart of Transformers).
* [ ] **Transformer Fine-tuning**: Leveraging Hugging Face `transformers` to fine-tune a BERT or DistilBERT model for Sentiment Analysis.
* [ ] **Parameter-Efficient Fine-Tuning (PEFT)**: Experimenting with LoRA or QLoRA for local LLM adaptation.

### **MLOps & Deployment**

* [ ] **Containerization**: Writing `Dockerfiles` for each `app.py` to ensure they run seamlessly on any Linux distribution.
* [ ] **CI/CD for ML**: Using GitHub Actions to automate unit tests for NumPy matrix dimensions.
* [ ] **Hugging Face Integration**: Creating a unified "Space" to host a gallery of all project interfaces.

---

## **AI Disclosure**

This project was developed with the assistance of **Gemini**, and other LLMs.

**How AI was utilized:**

* **Architectural Guidance:** Collaborating on the modular structure of the repository to ensure it remains scalable for future algorithm additions.
* **Debugging & Stability:** AI was instrumental in identifying and resolving numerical instability issues (e.g., handling `NaN` values during Gradient Descent and implementing a numerically stable Sigmoid function).
* **Feature Enhancement:** Assistance in implementing advanced visualization techniques, such as using **Principal Component Analysis (PCA)** to represent high-dimensional decision boundaries in 2D space.
* **Documentation:** Streamlining README and report formatting to ensure technical clarity for external reviewers.

---
