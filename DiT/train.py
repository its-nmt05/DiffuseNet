import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import wandb

from torchvision import transforms
from torchvision.datasets import MNIST

from DiT import DiT
from DiT.utils import get_model_params, sample_images
from diffusion.linear_noise_scheduler import LinearNoiseScheduler
from vae.model.vae import VAE
from dataset.VideoDataset import VideoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load model and dataset configs')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'model' / 'config.yaml',
        help='Path to the config file'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(model, vae, scheduler, dataset_config, train_config, log_wandb=True, device='cuda'):
    num_epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    ckpt_save_interval = train_config['ckpt_save_interval']
    ckpt_save_dir = train_config['ckpt_save_dir']
    sample_save_interval = train_config['sample_save_interval']
    sample_save_dir = train_config['sample_save_dir']
    num_samples = train_config['num_samples']
    seed = train_config['seed']

    dataset_name = dataset_config['name']
    dataset_args = dataset_config['args']

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

     # video dataset
    if dataset_name == 'video':
        train_split = VideoDataset(**dataset_args, split='train', transform=transform, seed=seed)
        test_split = VideoDataset(**dataset_args, split='test', transform=transform, seed=seed)
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False)
    
    # mnist
    elif dataset_name == 'mnist':
        train_split = MNIST(dataset_args['root'], train=True, transform=transform, download=True)
        test_split = MNIST(dataset_args['root'], train=False, transform=transform, download=True)
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False)

    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(sample_save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_criterion = torch.nn.MSELoss()

    print("Starting training!!!")
    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        # Training phase
        model.train()
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)

            with torch.no_grad():
                latents, _, _ = vae.encode(images)
                latents = latents / train_config['vae_scale']
                
            t = scheduler.sample_timesteps(latents.shape[0]).to(device)
            x_t, noise = scheduler.add_noise(latents, t)

            predicted_noise = model(x_t, t)
            loss = recon_criterion(predicted_noise, noise) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_losses.append(loss.item())    

            if log_wandb:
                wandb.log({
                    'train_loss': loss,
                    'latent_mu': latents.mean(),
                    'latent_std': latents.std(),
                    'pred_mu': predicted_noise.mean(),
                    'pred_std': predicted_noise.std(),
                })

        # Validation phase
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                latents, _, _ = vae.encode(images)
                latents = latents / train_config['vae_scale']

                t = scheduler.sample_timesteps(latents.shape[0]).to(device)
                x_t, noise = scheduler.add_noise(latents, t)
                predicted_noise = model(x_t, t)
                val_loss = recon_criterion(predicted_noise, noise)
                val_losses.append(val_loss.item())
                
                if log_wandb:
                    wandb.log({
                        'val_loss': val_loss.item()
                    })

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

        # save model checkpoint
        if (epoch + 1) % ckpt_save_interval == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_save_dir, f'model_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")

        # sample and save images
        if (epoch + 1) % sample_save_interval == 0:
            save_path = os.path.join(sample_save_dir, f'epoch_{epoch+1}.png')
            out = sample_images(model, vae, scheduler, vae_scale=train_config['vae_scale'], num_samples=num_samples, device=device)
            out.save(save_path)
            out.close()

            if log_wandb:
                wandb.log({'sample_image': wandb.Image(save_path)})
                print(f"Sampled image at epoch {epoch+1}")

    print("Finished training!!!")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset, vae, DiT, and training configs
    dataset_config = config['dataset']
    vae_config = config['vae']
    scheduler_config = config['scheduler']
    dit_config = config['dit']
    train_config = config['training']
    wandb_config = config['wandb']
    print("Config loaded successfully from:", args.config)

    # setup wandb logging
    log_wandb = not args.no_wandb
    if log_wandb:
        wandb.init(project=wandb_config['project'], name=wandb_config['name'], config=config)
    else:
        print("Diabling WandB logging!!")

    # set a seed for reproducibility
    seed = train_config['seed']
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # load vae
    vae = VAE(config=vae_config).to(device)
    vae_ckpt = vae_config['vae_ckpt']
    
    if vae_ckpt and os.path.exists(vae_ckpt):
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
        print(f"VAE loaded successfully from: {vae_ckpt}")
    else:
        print(f"VAE checkpoint not found at {vae_ckpt}, using untrained VAE")

    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # load noise scheduler
    scheduler = LinearNoiseScheduler(
        timesteps=scheduler_config['timesteps'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end'],
        device=device
    )
    print(f"Noise scheduler initialized with {scheduler.timesteps} timesteps")

    # load dit model
    model = DiT(config=dit_config).to(device)
    model.train()
    print(f"DiT model loaded successfully: {get_model_params(model) // 1e6:.2f} M parameters")

    train(model, vae, scheduler, dataset_config=dataset_config, train_config=train_config, log_wandb=log_wandb, device=device)

    # finish logging
    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
