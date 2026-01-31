# **Dataset Profile: CIFAR-10**

**Note**:
Python can download this dataset automatically using: ```datasets.CIFAR10(root='./data', download=True)```.

## Overview

CIFAR-10 (Canadian Institute For Advanced Research) consists of 60,000 tiny color images. It is small enough to train quickly but complex enough to show the power of Transfer Learning.

* **Image Size**:  pixels (very small, which makes it fast).
* **Channels**: 3 (RGB).
* **Total Images**: 60,000.
* **Training Set**: 50,000 images.
* **Testing Set**: 10,000 images.

* **Classes**: 10 mutually exclusive categories.

| Class ID | Name | Description |
| --- | --- | --- |
| 0 | **Airplane** | Commercial jets, fighter planes, etc. |
| 1 | **Automobile** | Sedans, SUVs (no trucks). |
| 2 | **Bird** | Various species in natural backgrounds. |
| 3 | **Cat** | Domestic cats. |
| 4 | **Deer** | Usually in forest settings. |
| 5 | **Dog** | Various breeds. |
| 6 | **Frog** | Green/Brown frogs. |
| 7 | **Horse** | Standing or running horses. |
| 8 | **Ship** | Cargo ships, boats, etc. |
| 9 | **Truck** | Large trucks (no pickup trucks). |

---

## **Why CIFAR-10 for Transfer Learning?**

1. **The Resolution Gap**: ResNet-18 was originally trained on **ImageNet**, which uses  images. Training it on  CIFAR images requires us to implement **Upsampling (Resizing)** in our preprocessing—this is a key skill in modern Computer Vision.
2. **Semantic Ambiguity**: Unlike Fashion-MNIST, where a "Boot" looks very different from a "T-shirt," CIFAR-10 has tricky pairs (e.g., **Cat vs. Dog** or **Automobile vs. Truck**).
3. **Benchmark Standard**: Using CIFAR-10 makes makes it easy for other ML developers, to compare our ResNet-18 performance against established benchmarks.

---
