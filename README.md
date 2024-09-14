
# Computer-Vision-Project

# Graph-Based Unsupervised Image Segmentation and Tiny NeRF Implementation

## Project Overview

This repository contains the implementation of the following:

1. **Ghaem-Hossein Segmentation (GHS)**
2. **Tiny Neural Radiance Fields (Tiny NeRF)**

Also for the project report please see the `report.pdf`

## 1. Graph-Based Unsupervised Image Segmentation

In this part, we propose a new method based on superpixel algorithms to create an underlying graph of the image and then use DSU (disjoint set union) to iteratively join the vertices in the overall graph.

- **Superpixel Segmentation**: We use the Simple Linear Iterative Clustering (SLIC) algorithm to group pixels into visually coherent clusters, known as superpixels. SLIC efficiently balances segmentation quality with computational efficiency by iteratively assigning pixels to clusters based on spatial and color proximity.
  
- **Graph-Based Clustering**: After obtaining superpixels, we model the image as a graph, where each superpixel is a vertex. Using graph theory algorithms such as Disjoint Set Union (DSU), we iteratively merge clusters to create globally consistent segmentations. This approach is highly resistant to noise due to its iterative refinement process.

- **Robustness**: One of the benefits of this method is that typically, more k-means is known to be robust to perturbations and SLIC superpixel is a variant of k-means thus, we expect this method to be more robust without any need for adversarial training.

### Example:
Please see the ```visualization.py``` for examples of the algorithm.

## Usage

To run the graph-based image segmentation using the **GHS algorithm**, use the following code example:

```python
# Import the GHS algorithm from your algorithm module
from codes.algorithm import GHS

# Load the image
import cv2
img = cv2.imread('your_image.png')

# Apply the GHS algorithm
img_gh_sup, segs_gh_sup = GHS(img, 0.05, 2000)

# Visualize the result
import matplotlib.pyplot as plt

plt.imshow(img_gh_sup)
plt.title("GHS Segmentation Result")
plt.show()
```

## 2. Tiny Neural Radiance Fields (Tiny NeRF)

Tiny NeRF is a simplified implementation of Neural Radiance Fields (NeRF), a state-of-the-art method for synthesizing novel views of complex scenes using sparse input images. Tiny NeRF approximates scene representation using a Multilayer Perceptron (MLP), predicting both the volume density and emitted radiance at any 3D location in space.


### Results:
- Render realistic images of a 3D scene from novel viewpoints.
- Trained using captured RGB images and corresponding camera poses.
  

## Team Members

GHS:
- Hossein Goli
- Mahdi Ghaempanah

Tiny NeRF:
- Ali Ansari
- Bahar Dibaeinia
- Siavash Rahimi

