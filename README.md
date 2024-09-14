# Computer-Vision-Project

# Graph-Based Unsupervised Image Segmentation and Tiny NeRF Implementation

## Project Overview

This repository contains the implementation of two key algorithms:

1. **Graph-Based Unsupervised Image Segmentation**
2. **Tiny Neural Radiance Fields (Tiny NeRF)**

Both projects explore advanced methods in computer vision and 3D rendering, focusing on improving robustness, efficiency, and performance in their respective domains.

## 1. Graph-Based Unsupervised Image Segmentation

This part of the project introduces a novel method for robust image segmentation using a combination of superpixel segmentation techniques and graph theory. Our approach includes the following components:

- **Superpixel Segmentation**: We use the Simple Linear Iterative Clustering (SLIC) algorithm to group pixels into visually coherent clusters, known as superpixels. SLIC efficiently balances segmentation quality with computational efficiency by iteratively assigning pixels to clusters based on spatial and color proximity.
  
- **Graph-Based Clustering**: After obtaining superpixels, we model the image as a graph, where each superpixel is a vertex. Using graph theory algorithms such as Disjoint Set Union (DSU), we iteratively merge clusters to create globally consistent segmentations. This approach is highly resistant to noise due to its iterative refinement process.

### Key Features:
- Efficient image segmentation using SLIC.
- Robust to noise with graph-based merging of superpixels.
- Suitable for various applications, including object tracking and image compression.

### Example:
```python
from skimage.segmentation import slic
import cv2
import matplotlib.pyplot as plt

# Perform SLIC superpixel segmentation
segments = slic(image, n_segments=400, compactness=10, sigma=1)

# Visualize the result
plt.imshow(segments)
plt.show()
```

## 2. Tiny Neural Radiance Fields (Tiny NeRF)

Tiny NeRF is a simplified implementation of Neural Radiance Fields (NeRF), a state-of-the-art method for synthesizing novel views of complex scenes using sparse input images. Tiny NeRF approximates scene representation using a Multilayer Perceptron (MLP), predicting both the volume density and emitted radiance at any 3D location in space.

### Key Components:
- **Scene Representation**: Represent a scene as a 3D continuous volumetric field, where the MLP takes spatial coordinates as input and outputs the color and volume density at that point.
- **Positional Encoding**: Use high-frequency positional encoding to improve network learning of fine details.
- **Volume Rendering**: Render the scene using classical volume rendering techniques to compute the expected color of rays passing through the scene.

### Example:
```python
def volume_rendering(ray):
    # Simplified volume rendering procedure
    for sample in ray_samples:
        T = exp(-sigma * delta)
        C += T * (1 - exp(-sigma * delta)) * color
    return C
```

### Results:
- Render realistic images of a 3D scene from novel viewpoints.
- Trained using captured RGB images and corresponding camera poses.
  
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/project-name.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the graph-based image segmentation, use the following command:
```bash
python algorithm.py --input your_image.png
```

To train and run the Tiny NeRF model:
```bash
python tiny_nerf.py --data_path /path/to/dataset
```

## Authors

GHS:
- Hossein Goli
- Mahdi Ghaempanah
Tiny Nerf:
- Ali Ansari
- Bahar Dibaeinia
- Siavash Rahimi
