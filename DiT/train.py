import os
import random
from pathlib import Path
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import wandb

from torchvision import transforms
from torchvision.datasets import MNIST

from DiT.DiT import DiT, TextEmbed
from DiT.utils import get_model_params, sample_images
from diffusion.linear_noise_scheduler import LinearNoiseScheduler
from vae.model.vae import VAE
from vae.utils import cache_latents
from dataset.VideoDataset import VideoDataset


class DitTrainer:

    def __init__(self, config_path, log_wandb=False, use_cond=False):
        self.config = self.load_config(config_path)
        self.log_wandb = log_wandb
        self.use_cond = use_cond
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load dataset, vae, DiT, and training configs
        self.dataset_config = self.config['dataset']
        self.vae_config = self.config['vae']
        self.scheduler_config = self.config['scheduler']
        self.dit_config = self.config['dit']
        self.train_config = self.config['training']
        self.wandb_config = self.config['wandb']
        print("Config loaded successfully from:", config_path)

        # setup conditioning
        self.use_cond = use_cond or ('cond_config' in self.dit_config)

        # setup wandb logging
        if self.log_wandb:
            wandb.init(project=self.wandb_config['project'], name=self.wandb_config['name'], config=self.config)
        else:
            print("Diabling WandB logging!!")

        # set a seed for reproducibility
        seed = self.train_config['seed']
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # using TF32 for FP32 matmul operations
        torch.set_float32_matmul_precision('high')

        # load vae
        self.vae = VAE(config=self.vae_config).to(self.device)
        vae_ckpt = self.vae_config['vae_ckpt']
        
        if vae_ckpt and os.path.exists(vae_ckpt):
            ckpt = torch.load(vae_ckpt, map_location=self.device, weights_only=True)
            if 'vae' in ckpt:
                self.vae.load_state_dict(ckpt['vae'])
                print(f"VAE loaded successfully from: {vae_ckpt}")
            else:
                print(f"Key 'vae' not found in ckpt: {vae_ckpt}, skipping load...")
        else:
            print(f"VAE checkpoint not found at {vae_ckpt}, using untrained VAE")

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # load noise scheduler
        self.scheduler = LinearNoiseScheduler(
            timesteps=self.scheduler_config['timesteps'],
            beta_start=self.scheduler_config['beta_start'],
            beta_end=self.scheduler_config['beta_end'],
            device=self.device
        )
        print(f"Noise scheduler initialized with {self.scheduler.timesteps} timesteps")

        # load dit model
        self.model = DiT(config=self.dit_config).to(self.device)
        self.model.train()
        print(f"DiT model loaded successfully: {get_model_params(self.model) // 1e6:.2f} M parameters")

        # load text encoder
        if self.use_cond:
            cond_config = self.dit_config.get('cond_config')
            self.text_encoder = TextEmbed(text_embed_model=cond_config['text_embed_model'], 
                                     dropout_prob=cond_config['dropout_prob'], device=self.device)
            print(f"Using '{cond_config['text_embed_model']}' for text conditioning")
        else:
            self.text_encoder = None

        dataset_dir = self.dataset_config['args']['dataset_dir']
        latent_save_path = os.path.join(dataset_dir, 'vae_latents.npz')

        # latents are not cached; cache them 
        if not os.path.isfile(latent_save_path):
            print(f"No cached latents found in {dataset_dir}. Caching now...")
            cache_latents(self.vae, dataset_dir, device=self.device)
            
    def load_config(self, config_path: Path) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def train(self):
        # load params from config
        num_epochs = self.train_config['epochs']
        batch_size = self.train_config['batch_size']
        lr = self.train_config['lr']
        weight_decay = self.train_config['weight_decay']
        ckpt_save_interval = self.train_config['ckpt_save_interval']
        ckpt_save_dir = self.train_config['ckpt_save_dir']
        sample_save_interval = self.train_config['sample_save_interval']
        sample_save_dir = self.train_config['sample_save_dir']
        num_samples = self.train_config['num_samples']
        seed = self.train_config['seed']
        cfg_guidance_scale = self.train_config['cfg_guidance_scale']

        dataset_name = self.dataset_config['name']
        dataset_args = self.dataset_config['args']

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        recon_criterion = torch.nn.MSELoss()

        print("Starting training!!!")
        for epoch in range(num_epochs):
            train_losses = []
            val_losses = []

            # Training phase
            self.model.train()
            for im, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                
                # use cached latents
                if train_split.use_latents:
                    latents = im.to(self.device)
                else:
                    with torch.no_grad():
                        latents, _, _ = self.vae.encode(im)
                    
                latents = latents / self.train_config['vae_scale']

                t = self.scheduler.sample_timesteps(latents.shape[0]).to(self.device)
                x_t, noise = self.scheduler.add_noise(latents, t)
            
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    if self.use_cond:
                        # extract text embds with dropout
                        context, mask = self.text_encoder(captions, apply_dropout=True)
                        predicted_noise = self.model(x_t, t, y=context, mask=mask)
                    else:
                        predicted_noise = self.model(x_t, t)

                    loss = recon_criterion(predicted_noise, noise) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_losses.append(loss.item())    

                if self.log_wandb:
                    wandb.log({
                        'train_loss': loss,
                        'latent_mu': latents.mean(),
                        'latent_std': latents.std(),
                        'pred_mu': predicted_noise.mean(),
                        'pred_std': predicted_noise.std(),
                    })

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for im, captions in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    
                    # use cached latents
                    if test_split.use_latents:
                        latents = im.to(self.device)
                    else:
                        with torch.no_grad():
                            latents, _, _ = self.vae.encode(im)
                            
                    latents = latents / self.train_config['vae_scale']

                    t = self.scheduler.sample_timesteps(latents.shape[0]).to(self.device)
                    x_t, noise = self.scheduler.add_noise(latents, t)
                    
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        if self.use_cond:
                            # extract text embds without dropout
                            context, mask = self.text_encoder(captions)
                            predicted_noise = self.model(x_t, t, y=context, mask=mask)
                        else:
                            predicted_noise = self.model(x_t, t)

                        val_loss = recon_criterion(predicted_noise, noise)
                        
                    val_losses.append(val_loss.item())
                    if self.log_wandb:
                        wandb.log({
                            'val_loss': val_loss.item()
                        })

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")

            # save model checkpoint
            if (epoch + 1) % ckpt_save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(ckpt_save_dir, f'model_epoch_{epoch+1}.pth'))
                print(f"Model saved at epoch {epoch+1}")

            # sample and save images
            if (epoch + 1) % sample_save_interval == 0:
                save_path = os.path.join(sample_save_dir, f'epoch_{epoch+1}.png')
                captions = ["Small orange lizard-like creature with flames on its tail, battling against a human trainer in a grassy field, daytime setting, dynamic action shot, energy-filled atmosphere.",
                            "Red-haired character walking through dense forest, overcast day, pixelated art style, serene atmosphere, lush greenery surrounding the path",
                            "A red-roofed healing center in a vibrant green field, daytime, close-up, peaceful atmosphere with gentle sunlight.", 
                            "Pink-haired character standing beside a vibrant blue water body, soft daylight, serene atmosphere."]
                out = sample_images(self, vae_scale=self.train_config['vae_scale'], input_prompt=captions, 
                                    num_samples=num_samples, cfg_guidance_scale=cfg_guidance_scale)
                out.save(save_path)
                out.close()

                if self.log_wandb:
                    wandb.log({'sample_image': wandb.Image(save_path)})
                    print(f"Sampled image at epoch {epoch+1}")

        print("Finished training!!!")


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
    parser.add_argument(
        "--cond",
        action="store_true",
        help="Enable conditioning (override)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # create the trainer
    trainer = DitTrainer(
        config_path=args.config,
        log_wandb=not args.no_wandb,
        use_cond=args.cond
    )

    # start training
    trainer.train()


if __name__ == '__main__':
    main()
