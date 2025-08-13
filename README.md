Collection of generative models trained from scratch including DDPM, DiT, VAE for learning and research

## DDPM Generations

<div style="display: flex; gap: 1rem;">
    <img src="./diffusion/outputs/mnist.png" alt="mnist_generation" width="48%">
    <img src="./diffusion/outputs/cifar_perceptual.png" alt="CIFAR-10_perceptual_generation" width="48%">
</div>

### Timestep sampling (T=500)
![sampling_timesteps](./diffusion/outputs/sampling.png)

## VAE (Variational Autoncoder)

Implemented a VAE from scratch inspired by [SD-VAE](https://github.com/CompVis/stable-diffusion). It was trained on both MNIST and Minecraft images. The model uses a convolutional autoencoder with upsampling and downsampling blocks along with residual attention layers. 

Training was performed using `KLD loss` and `LPIPS` loss using a pretrained `vgg16` network. The [config.yaml](vae/configs/config.yaml) creates a 64M param VAE model.

### Training 

<div style="display: flex; gap: 1rem;">
    <img src="vae\experiments\minecraft_training.gif" alt="minecraft_training" width="48%">
    <img src="vae\experiments\mnist_training.gif" alt="mnist_training" width="48%">
</div>

### Interpolations

<div style="display: flex; gap: 1rem;">
    <img src="vae\experiments\minecraft_interpolate.gif" alt="minecraft_interpolations" width="48%">
    <img src="vae\experiments\mnist_interpolate.gif" alt="vae_interpolations" width="48%">
</div>
