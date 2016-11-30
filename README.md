# Image Generation with Deep Boltzmann Machine and (Variational) Autoencoders

A reseach project trying to scale up deep bolzmann machine for realistic image generation.

This repo currently implements:

1. RBM+Autoencoder model with RBM to model distribution on encoded latent space.
2. RBM+VAE model with RBM to model posterior mean for decoder.
3. Deep convolutional autoencoder, deep residual network and deep residual autoencoder in Keras.
4. Deep convolutional variational autoencoder in Keras.
5. Deep Boltzmann machine in Tensorflow.
6. Deep restricted Bolztmann network, a simple method to stack multiple RBMs for better image generation and feature extraction, in Tensorflow. [Technical report](https://arxiv.org/pdf/1611.07917v1.pdf)
7. RBM, Gaussian-Binary RBM, Convolutional RBM, Gaussian-Binary ConvRBM in Tensorflow.

Sample Images

1. RBM+VAE, (both trained on entire cifar-10)

   ![](figs/rbm_vae.png)

2. VAE trained on entire cifar-10 while RBM trained on a specific class (horse and ship).

   ![Horse](figs/rbm_vae_horse.png)

   ![ship](figs/rbm_vae_ship.png)

   3. RBM + Autoencoder

      ![](figs/rbm_ae.png)

      4. Convolutional deep restricted Bolztmann network

         ![](figs/mnist_conv.png)

         ![](figs/horse_conv.png)

Download the MNIST dataset from: https://drive.google.com/file/d/0B_r9W7MOBsRsYUlKUnRTa1Fxdmc/view?usp=sharing