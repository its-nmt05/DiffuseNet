Collection of generative models trained from scratch including DDPM, DiT, VAE for learning and research

## DDPM Generations

<div style="display: flex; gap: 1rem;">
    <img src="./images/diffusion/mnist.png" alt="mnist_generation" width="48%">
    <img src="./images/diffusion/cifar_perceptual.png" alt="CIFAR-10_perceptual_generation" width="48%">
</div>

### Timestep sampling (T=500)
![sampling_timesteps](./images/diffusion/sampling.png)

## VAE (Variational Autoncoder)

Implemented a VAE from scratch inspired by [SD-VAE](https://github.com/CompVis/stable-diffusion). It was trained on both MNIST and Minecraft images. The model uses a convolutional autoencoder with upsampling and downsampling blocks along with residual attention layers. 

Training was performed using `adversarial loss`, `KLD loss` and `LPIPS` loss using a pretrained `vgg16` network. The [vae_xl.yaml](./vae/configs/vae_xl.yaml) creates a 97.5M param VAE model.

### Training 

<div style="display: flex; gap: 1rem;">
    <img src="./images/vae/minecraft_training.gif" alt="minecraft_training" width="48%">
    <img src="./images/vae/mnist_training.gif" alt="mnist_training" width="48%">
</div>

### Interpolations

<div style="display: flex; gap: 1rem;">
    <img src="./images/vae/minecraft_interpolate.gif" alt="minecraft_interpolations" width="48%">
    <img src="./images/vae/mnist_interpolate.gif" alt="vae_interpolations" width="48%">
</div>

### Reconstructions

The VAE was trained on `256x256` Minecraft images and outputs latents of dim `64x8x8`, with a `48x` compression.

![reconstruction](./images/vae/vae_recon.png)

## DiT
Implementation of Diffusion Transformer inspired by the original [DiT paper](https://arxiv.org/abs/2212.09748). The model uses transformer blocks with timesteps conditioned through `adaLN` Tested both small (76 M) and large (608 M) variants on the Minecraft dataset using our pre-trained VAE. All the models were trained on an NVIDIA A100.

<div style="display: flex; gap: 1rem; margin-bottom:1rem;">
    <img src="./images/DiT/DiT_samples_minecraft.png" alt="dit_minecraft" width="48%">
    <img src="./images/DiT/DiT_samples_pokemon.png" alt="dit_pokemon" width="48%">
</div> 



To train on your own dataset, modify `config.yaml` files and run:

```python
python -m vae.train_vae --config './vae/configs/vae_xl.yaml'
python -m DiT.train --config './DiT/configs/config_xl.yaml'
```

These commands will train the VAE and DiT models using the specified config files.
