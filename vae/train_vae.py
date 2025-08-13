import os
import yaml
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import random
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from model import VAE
from model import LPIPS
from dataset import VideoDataset
from utils import get_model_params, get_kl_loss, sample_images
from torch.utils.data.dataloader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Load model and dataset configs')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'model' / 'config.yaml',
        help='Path to the config file'
    )
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
    model_config    = config['model']
    train_config = config['training']
    print("Config loaded successfully from:", args.config)

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
        train_split = VideoDataset(**dataset_args, split='train', transform=transform)
        test_split = VideoDataset(**dataset_args, split='test', transform=transform)
        train_loader = DataLoader(train_split, batch_size=train_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_split, batch_size=train_config['batch_size'], shuffle=False)
    
    # mnist
    elif dataset_name == 'mnist':
        train_split = MNIST(dataset_args['root'], train=True, transform=transform, download=True)
        test_split = MNIST(dataset_args['root'], train=False, transform=transform, download=True)
        train_loader = DataLoader(train_split, batch_size=train_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_split, batch_size=train_config['batch_size'], shuffle=False)

    model = VAE(config=model_config).to(device)
    print(f"Model loaded successfully: {get_model_params(model) / 1e6:.2f} M parameters")

    lpips_model = LPIPS().eval().to(device)
    print("LPIPS model loaded successfully")

    num_epochs = train_config['epochs']
    num_samples = train_config["num_samples"]
    kl_weight = train_config['kl_weight']
    percep_weight = train_config['percep_weight']
    save_interval = train_config["save_interval"]
    vae_ckpt_save_dir = train_config["vae_ckpt_save_dir"]
    recon_img_save_dir = train_config["recon_img_save_dir"]

    os.makedirs(vae_ckpt_save_dir, exist_ok=True)
    os.makedirs(recon_img_save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    recon_criterion = torch.nn.MSELoss()
    
    print("Starting training!!!")

    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        # Training phase
        model.train()
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            data = data.to(device)
            optimizer.zero_grad()
        
            recon, mu, logvar = model(data)
        
            # compute losses
            recon_loss = recon_criterion(data, recon)
            kl_loss = get_kl_loss(mu, logvar)
            percep_loss = torch.mean(lpips_model(data, recon, normalize=True))
            loss = recon_loss + kl_weight * kl_loss + percep_weight * percep_loss
            
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())            

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
                
        # save samples
        imgs = sample_images(model, test_loader, device, num_samples=num_samples)
        imgs.save(os.path.join(recon_img_save_dir, f'epoch_{epoch+1}.png'))
        imgs.close()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(vae_ckpt_save_dir, f'vae_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")

    print("Finished training!!!")
            

if __name__ == '__main__':
    main()