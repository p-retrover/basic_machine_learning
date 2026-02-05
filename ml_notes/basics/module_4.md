# **Module 4: Unsupervised Learning**

Unsupervised learning involves analyzing datasets that consist only of input features $x^{(i)}$ without any corresponding output labels $y^{(i)}$. The primary objective is to learn the underlying structure, distribution, or density of the data.

---

## **4.1 Clustering: k-Means**

Clustering is the process of grouping data points such that points in the same group (cluster) are more similar to each other than to those in other groups.

**The Objective Function (Inertia):**
The algorithm seeks to minimize the **Within-Cluster Sum of Squares (WCSS)**:

$$J = \sum_{j=1}^{k} \sum_{i \in C_j} \left\lVert x^{(i)} - \mu_j \right\rVert_2^2$$

where $\mu_j$ is the centroid of cluster $C_j$.

**The Algorithm (Lloyd’s Algorithm):**

1. **Initialization:** Select $k$ initial centroids $\mu_1, \cdots, \mu_k$ (usually via k-means++ for better convergence).
2. **Cluster Assignment:** Assign each point $x^{(i)}$ to the cluster $C_j$ whose centroid is closest:
$$c^{(i)} := \arg\min_{j} \left\lVert x^{(i)} - \mu_j \right\rVert_2$$

3. **Centroid Update:** Recalculate the centroid of each cluster based on its assigned points:
$$\mu_j := \frac{1}{|C_j|} \sum_{i \in C_j} x^{(i)}$$

4. **Convergence:** Repeat steps 2 and 3 until the centroids stabilize.

---

## **4.2 Principal Component Analysis (PCA)**

PCA is a deterministic, linear dimensionality reduction technique that transforms correlated features into a set of linearly uncorrelated variables called **Principal Components**.

**The Mathematical Derivation (The Eigendecomposition Link):**
For a dataset $X \in \mathbb{R}^{m \times d}$ that has been zero-centered ($\mu = 0$):

1. **Compute the Covariance Matrix ($\Sigma$):**

$$\Sigma = \frac{1}{m - 1} X^T X$$

$\Sigma$ is a $d\times d$ symmetric, positive semi-definite matrix.
2. **Solve the Eigenvalue Problem:**
Find the eigenvalues  and eigenvectors  such that:
$$\Sigma v = \lambda v$$
3. **Dimensionality Reduction:**

* The **Eigenvectors** (Principal Components) represent the directions of maximum variance.
* The **Eigenvalues** represent the magnitude of variance explained by each direction.
* To reduce dimensionality to $k$, we construct a projection matrix $P$ using the top $k$ eigenvectors and transform the data: $X_{\text{new}} = X P$.



**Intuition:** PCA performs an orthogonal transformation (rotation) of the coordinate system such that the first axis lies along the direction of greatest variance, the second along the next greatest (orthogonal to the first), and so on.

---

## **4.3 Non-Linear Dimensionality Reduction (Visualization)**

Linear methods like PCA often fail to capture complex, manifold-structured data. For visualization, we use non-linear probabilistic methods:

* **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
* Converts distances between points in high-dimensional space into conditional probabilities.
* Minimizes the **Kullback-Leibler (KL) divergence** between the high-dim distribution and a low-dim (usually 2D) distribution.
* **Best for:** Visualizing clusters and local neighborhoods.


* **UMAP (Uniform Manifold Approximation and Projection):**
* Based on Riemannian geometry and algebraic topology.
* Better at preserving the **global structure** of the data and generally faster than t-SNE.



---

## **4.4 Hierarchical and Density-Based Clustering**

* **Hierarchical Clustering:** Creates a tree of clusters (Dendrogram).
* *Agglomerative:* Bottom-up; starts with each point as its own cluster and merges the closest pairs.
* *Divisive:* Top-down; starts with one cluster and recursively splits.


* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
* Groups points that are packed closely together.
* **Core Concept:** Defines clusters as dense regions separated by sparse regions.
* **Pros:** Can find clusters of arbitrary shapes and identify outliers (noise).

---
