# Predicting Eye Fixations using CNN

- Computational **visual saliency models** (using computer vision techniques)
- Saliency modeling by three steps in sequence:
    - early feature extraction
    - feature contrast inference
    - and contrast integration
- **Contrast computation** **over early features** is another key procedure
- The "**center-surround difference**” operator across multiple scales to calculate contrast.
- computing contrast using a variety of mathematical tools, for example:
    - using information theories [12],
    - frequency spectrum [14-17],
    - sparse coding [13, 39] or
    - autoencoder [40]
- The last step for saliency modeling is to integrate various contrast features to yield **saliency maps**
    - Itti et al. [1] linearly fused three contrast maps using fixed weights.
    - Zhao and Koch [18] learned the optimal weights associated with various contrasts using a least square technique upon a set of eye tracking data.
    - Judd et al. [2] learned a linear SVM to fuse bottom-up features.
    - Borji [3] explored the linear regression model and AdaBoost classifier for optimized feature fusion.
- **Multiresolution convolutional neural network (Mr-CNN)** which simultaneously learns early features, bottom-up saliency, top-down factors, and their integration from raw image data.

# References Guide

[1] - for early feature extraction, this reference proposed three low-level features including intensity, color, and orientation.

[2] - this reference considered more early features, for example, subbands of the steerable pyramid, 3D color histograms and color probabilities based features.