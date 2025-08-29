import os
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import wandb

from vae.model.vae import VAE
from vae.model.discriminator import Discriminator
from vae.model.lpips import LPIPS
from dataset.VideoDataset import VideoDataset
from vae.utils import get_model_params, get_kl_loss, sample_images


def parse_args():
    parser = argparse.ArgumentParser(description='Load model and dataset configs')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'model' / 'config.yaml',
        help='Path to the config file'
    )
    parser.add_argument(
        '--no_wandb', 
        action='store_true', 
        help='Disable WandB logging')
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset, model, and training configs
    dataset_config  = config['dataset']
    dataset_name = dataset_config['name']
    dataset_args = dataset_config['args']
    model_config = config['model']
    train_config = config['training']
    wandb_config = config['wandb']
    print("Config loaded successfully from:", args.config)

    # setup wandb logging
    use_wandb = not args.no_wandb
    if use_wandb:
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

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

    # video dataset
    if dataset_name == 'video':
        train_split = VideoDataset(**dataset_args, split='train', transform=transform, seed=seed)
        test_split = VideoDataset(**dataset_args, split='test', transform=transform, seed=seed)
        train_loader = DataLoader(train_split, batch_size=train_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_split, batch_size=train_config['batch_size'], shuffle=False)
    
    # mnist
    elif dataset_name == 'mnist':
        train_split = MNIST(dataset_args['root'], train=True, transform=transform, download=True)
        test_split = MNIST(dataset_args['root'], train=False, transform=transform, download=True)
        train_loader = DataLoader(train_split, batch_size=train_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_split, batch_size=train_config['batch_size'], shuffle=False)

    model = VAE(config=model_config).to(device)
    print(f"VAE loaded successfully: {get_model_params(model) / 1e6:.2f} M parameters")

    lpips_model = LPIPS().eval().to(device)
    print("LPIPS model loaded successfully")

    discriminator = Discriminator(im_channels=model_config['in_ch']).to(device)
    print(f"Discriminator model loaded successfully: {get_model_params(discriminator) / 1e6:.2f} M parameters")

    num_epochs = train_config['epochs']
    num_samples = train_config['num_samples']
    percep_weight = train_config['percep_weight']
    kl_weight = train_config['kl_weight']
    disc_weight = train_config['disc_weight']
    disc_start = train_config['disc_start']
    save_interval = train_config['save_interval']
    vae_ckpt_save_dir = train_config['vae_ckpt_save_dir']
    disc_ckpt_save_dir = train_config['disc_ckpt_save_dir']
    recon_img_save_dir = train_config['recon_img_save_dir']

    os.makedirs(vae_ckpt_save_dir, exist_ok=True)
    os.makedirs(disc_ckpt_save_dir, exist_ok=True)
    os.makedirs(recon_img_save_dir, exist_ok=True)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    
    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()
    step_count = 0

    print("Starting training!!!")

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        # Training phase
        model.train()
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            step_count += 1
            data = data.to(device)
            optimizer_g.zero_grad()
        
            recon, mu, logvar = model(data)
            z = mu # For logging 
        
            # compute losses
            recon_loss = recon_criterion(data, recon)
            kl_loss = get_kl_loss(mu, logvar)
            percep_loss = torch.mean(lpips_model(data, recon, normalize=True))
            g_loss = recon_loss + kl_weight * kl_loss + percep_weight * percep_loss

            # start adversrial training for generator
            if step_count > disc_start:
                disc_fake_pred = discriminator(recon) # logits for images from vae
                g_disc_loss = disc_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape, device=device)) 
                g_loss += disc_weight * g_disc_loss
                
            g_loss.backward()
            optimizer_g.step()

            # start optimization for the discriminator
            if step_count > disc_start:
                optimizer_d.zero_grad()

                disc_fake_pred = discriminator(recon.detach()) # only train discriminator
                disc_real_pred = discriminator(data)

                # classify make images as 0 and real images as 1
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=device)) 
                disc_real_loss = disc_criterion(disc_real_pred, torch.ones(disc_real_pred.shape, device=device)) 
                disc_loss = (disc_fake_loss + disc_real_loss) / 2

                disc_loss.backward()
                optimizer_d.step()  

            train_losses.append(g_loss.item())      

            if use_wandb:
                wandb.log({
                    'recon_loss': recon_loss,
                    'kl_loss': kl_loss,
                    'percep_loss': percep_loss,
                    'train_loss': g_loss,
                    'kl_loss_total': kl_weight * kl_loss,
                    'gen_loss': g_disc_loss if step_count > disc_start else 0,
                    'disc_loss': disc_loss if step_count > disc_start else 0,
                    'z_mean': z.mean(),
                    'z_std': z.std()
                })

        # Validation phase
        model.eval()
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                data = data.to(device)
                
                recon, mu, logvar = model(data)
                
                # compute losses
                recon_loss = recon_criterion(data, recon)
                kl_loss = get_kl_loss(mu, logvar)
                percep_loss = torch.mean(lpips_model(data, recon, normalize=True))
                loss = recon_loss + kl_weight * kl_loss + percep_weight * percep_loss

                val_losses.append(loss.item()) 

                if use_wandb:
                    wandb.log({
                        'val_loss': loss.item(),
                    })   

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")        
                
        # save samples
        save_path = os.path.join(recon_img_save_dir, f'epoch_{epoch+1}.png')
        imgs = sample_images(model, test_loader, device, num_samples=num_samples)
        imgs.save(save_path)
        imgs.close()

        if use_wandb:
            wandb.log({'sample_images': wandb.Image(save_path)})

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(vae_ckpt_save_dir, f'vae_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(disc_ckpt_save_dir, f'disc_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")

    print("Finished training!!!")

    # finish logging
    if use_wandb:
        wandb.finish()
            

if __name__ == '__main__':
    main()