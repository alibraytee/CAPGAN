# CAPGAN
**Paper title: Conditional Variational Autoencoder with Balanced Pre-training for Generative Adversarial Networks**

*Published in 2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA) - Research track*

Abstract:
Class imbalance occurs in many real-world applications, including image classification, where the number of images in each class differs significantly. With imbalanced data, the generative adversarial networks (GANs) leans to majority class samples. The two recent methods, Balancing GAN (BAGAN) and improved BAGAN (BAGAN-GP), are proposed as an augmentation tool to handle this problem and restore the balance to the data. The former pre-trains the autoencoder weights in an unsupervised manner. However, it is unstable when the images from different categories have similar features. The latter is improved based on BAGAN by facilitating supervised autoencoder training, but the pre-training is biased towards the majority classes.
In this work, we propose a novel \textit{Conditional Variational Autoencoder with Balanced Pre-training for Generative Adversarial Networks} (CAPGAN) as an augmentation tool to generate realistic synthetic images. In particular, we utilize a conditional convolutional variational autoencoder with supervised and balanced pre-training for the GAN initialization and training with gradient penalty.  Our proposed method presents a superior performance of other state-of-the-art methods on the highly imbalanced version of MNIST, Fashion-MNIST, CIFAR-10, and two medical imaging datasets. Our method can synthesize high-quality minority samples in terms of Fréchet inception distance, structural similarity index measure and perceptual quality.

Corresponding author: Ali Braytee

Cite at:
https://ieeexplore.ieee.org/abstract/document/10032367
Yao, Y., Wang, X., Ma, Y., Fang, H., Wei, J., Chen, L., ... & Braytee, A. (2022, October). Conditional Variational Autoencoder with Balanced Pre-training for Generative Adversarial Networks. In 2022 IEEE 9th International Conference on Data Science and Advanced Analytics (DSAA) (pp. 1-10). IEEE.
