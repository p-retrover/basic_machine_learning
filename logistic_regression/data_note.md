# Breast Cancer Wisconsin (Diagnostic) Dataset

## Overview

The **Breast Cancer Wisconsin (Diagnostic) Dataset** contains features computed from digitized images of **fine needle aspirates (FNA)** of breast masses. The features describe characteristics of the **cell nuclei** present in each image.

The data lies in a three-dimensional space as described in:

**K. P. Bennett and O. L. Mangasarian**
*Robust Linear Programming Discrimination of Two Linearly Inseparable Sets*
Optimization Methods and Software, Volume 1, 1992, pages 23–34.

---

## Data Sources

* **University of Wisconsin CS FTP Server**
  ftp.cs.wisc.edu
  Directory: `math-prog/cpo-dataset/machine-learn/WDBC/`

* **UCI Machine Learning Repository**
  [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

## Attribute Information

### Identifiers

1. **ID Number**
2. **Diagnosis**

   * **M**: Malignant
   * **B**: Benign

### Feature Set (Attributes 3–32)

Ten real-valued features are computed for each cell nucleus:

* **Radius**: Mean of distances from the center to points on the perimeter
* **Texture**: Standard deviation of gray-scale values
* **Perimeter**: Perimeter of the nucleus
* **Area**: Area of the nucleus
* **Smoothness**: Local variation in radius lengths
* **Compactness**: (perimeter² / area) − 1.0
* **Concavity**: Severity of concave portions of the contour
* **Concave Points**: Number of concave portions of the contour
* **Symmetry**: Symmetry of the nucleus
* **Fractal Dimension**: “Coastline approximation” − 1

---

## Feature Construction

For each of the ten features, the following statistics are computed:

* **Mean**
* **Standard Error**
* **Worst** (mean of the three largest values)

This results in a total of **30 features** per sample.

Examples:

* Field 3 → Mean Radius
* Field 13 → Radius Standard Error
* Field 23 → Worst Radius

All feature values are recoded to **four significant digits**.

---

## Dataset Characteristics

* **Missing Attribute Values**: None
* **Class Distribution**:

  * Benign: 357 samples
  * Malignant: 212 samples

---

## Summary

* **Total Samples**: 569
* **Total Features**: 30
* **Classification Type**: Binary (Benign vs Malignant)
* **Common Applications**: Logistic Regression, Support Vector Machines, Neural Networks, Feature Selection, Medical Diagnostics

---
