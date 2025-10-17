import os
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
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


class VAETrainer:

    def __init__(self, config_path, log_wandb=False, ddp=False, use_compile=False):
        self.config = load_config(config_path)
        self.use_wandb = log_wandb
        self.ddp = ddp
        if ddp:
            assert torch.cuda.is_available(), "CUDA must be available for DDP"
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f"cuda:{self.ddp_local_rank}"
            self.master_process = self.ddp_rank == 0    # master process will be used for logging, saving, etc
            torch.cuda.set_device(self.device)
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load dataset, model, and training configs
        self.dataset_config  = self.config['dataset']
        self.model_config = self.config['model']
        self.train_config = self.config['training']
        self.wandb_config = self.config['wandb']

        if self.master_process:
            print("Config loaded successfully from:", config_path)

         # setup wandb logging
        if self.use_wandb and self.master_process:
            wandb.init(project=self.wandb_config['project'], name=self.wandb_config['name'], config=self.config)

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
        torch.autograd.set_detect_anomaly(True)

        self.model = VAE(config=self.model_config).to(self.device)
        if use_compile:
            self.model = torch.compile(self.model)
        if ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        if self.master_process:
            print(f"VAE loaded successfully: {get_model_params(self.model) / 1e6:.2f} M parameters")
        
        self.lpips_model = LPIPS().eval().to(self.device)
        if self.master_process:
            print("LPIPS model loaded successfully")

        self.discriminator = Discriminator(im_channels=self.model_config['in_ch']).to(self.device)
        if use_compile:
            self.discriminator = torch.compile(self.discriminator)
        if ddp:
            self.discriminator = DDP(self.discriminator, device_ids=[self.ddp_local_rank])
        if self.master_process:
            print(f"Discriminator model loaded successfully: {get_model_params(self.discriminator) / 1e6:.2f} M parameters")

        self.ckpt_save_dir = self.train_config['ckpt_save_dir']
        self.recon_img_save_dir = self.train_config['recon_img_save_dir']

        if self.master_process:
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            os.makedirs(self.recon_img_save_dir, exist_ok=True)

        self.train_loader, self.test_loader = self.setup_dataloaders()

    def barrier(self):  
        # sync all individual processes
        if self.ddp:
            torch.distributed.barrier()

    def save(self, epoch):
        save_dict = {
            'epoch': epoch + 1,
            'vae': self.model.module.state_dict() if self.ddp else self.model.state_dict(),
            'disc': self.discriminator.module.state_dict() if self.ddp else self.discriminator.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
            'optim_d': self.optimizer_d.state_dict()
        }
        torch.save(save_dict, os.path.join(self.ckpt_save_dir, f"vae_epoch_{epoch+1}.pth"))

        # sync processes after saving model
        self.barrier()
        
    def setup_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_name = self.dataset_config['name']
        dataset_args = self.dataset_config['args']
        seed = self.train_config['seed']
        batch_size = self.train_config['batch_size']

        # video dataset
        if dataset_name == 'video':
            train_split = VideoDataset(**dataset_args, use_latents=False, split='train', transform=transform, seed=seed)
            test_split = VideoDataset(**dataset_args, use_latents=False, split='test', transform=transform, seed=seed)
           
        # mnist
        elif dataset_name == 'mnist':
            train_split = MNIST(dataset_args['root'], train=True, transform=transform, download=True)
            test_split = MNIST(dataset_args['root'], train=False, transform=transform, download=True)

        if self.ddp:
            train_sampler = DistributedSampler(train_split, num_replicas=self.ddp_world_size, rank=self.ddp_rank, shuffle=True)
            test_sampler = DistributedSampler(test_split, num_replicas=self.ddp_world_size, rank=self.ddp_rank, shuffle=False)
            train_loader = DataLoader(train_split, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_split, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader

    def train(self):
        # load params
        num_epochs = self.train_config['epochs']
        num_epochs = self.train_config['epochs']
        num_samples = self.train_config['num_samples']
        percep_weight = self.train_config['percep_weight']
        kl_weight = self.train_config['kl_weight']
        disc_weight = self.train_config['disc_weight']
        disc_start = self.train_config['disc_start']
        save_interval = self.train_config['save_interval']

        self.optimizer_g = torch.optim.Adam(self.model.parameters(), lr=self.train_config['lr'], weight_decay=self.train_config['weight_decay'])
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.train_config['lr'], weight_decay=self.train_config['weight_decay'])
        
        recon_criterion = torch.nn.MSELoss()
        disc_criterion = torch.nn.MSELoss()
        step_count = 0

        if self.master_process:
            print("Starting training!!!")

        for epoch in range(num_epochs):
            if self.ddp:    
                # ensure all gpus are in sync for shuffle
                self.train_loader.sampler.set_epoch(epoch)  

            train_losses = []
            val_losses = []

            # Training phase
            self.model.train()
            for data, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                step_count += 1
                data = data.to(self.device)
                self.optimizer_g.zero_grad()
                
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    recon, mu, logvar = self.model(data)
                    z = mu # For logging 
            
                    # compute losses
                    recon_loss = recon_criterion(data, recon)
                    kl_loss = get_kl_loss(mu, logvar)
                    percep_loss = torch.mean(self.lpips_model(data, recon, normalize=True))
                    g_loss = recon_loss + kl_weight * kl_loss + percep_weight * percep_loss

                    # start adversrial training for generator
                    if step_count > disc_start:
                        disc_fake_pred = self.discriminator(recon) # logits for images from vae
                        g_disc_loss = disc_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape, device=self.device)) 
                        g_loss += disc_weight * g_disc_loss
                        
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config['grad_clip'])
                self.optimizer_g.step()

                # start optimization for the discriminator
                if step_count > disc_start:
                    self.optimizer_d.zero_grad()

                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        disc_fake_pred = self.discriminator(recon.detach()) # only train discriminator
                        disc_real_pred = self.discriminator(data)

                        # classify make images as 0 and real images as 1
                        disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=self.device)) 
                        disc_real_loss = disc_criterion(disc_real_pred, torch.ones(disc_real_pred.shape, device=self.device)) 
                        disc_loss = (disc_fake_loss + disc_real_loss) / 2

                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.train_config['grad_clip'])
                    self.optimizer_d.step()  

                train_losses.append(g_loss.item())      

                if self.use_wandb and self.master_process: 
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
            self.model.eval()
            with torch.no_grad():
                for data, _ in tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    data = data.to(self.device)
                    
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        recon, mu, logvar = self.model(data)
                        
                        # compute losses
                        recon_loss = recon_criterion(data, recon)
                        kl_loss = get_kl_loss(mu, logvar)
                        percep_loss = torch.mean(self.lpips_model(data, recon, normalize=True))
                        loss = recon_loss + kl_weight * kl_loss + percep_weight * percep_loss

                    val_losses.append(loss.item()) 

                    if self.use_wandb and self.master_process:
                        wandb.log({
                            'val_loss': loss.item(),
                        })   

            if self.master_process:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")        

            # save samples
            if self.master_process:
                save_path = os.path.join(self.recon_img_save_dir, f'epoch_{epoch+1}.png')
                with torch.no_grad():
                    imgs = sample_images(self.model, self.test_loader, self.device, num_samples=num_samples)
                imgs.save(save_path)
                imgs.close()

                if self.use_wandb:
                    wandb.log({'sample_images': wandb.Image(save_path)})

            # Save model checkpoint
            if (epoch + 1) % save_interval == 0 and self.master_process:
                self.save(epoch)
                print(f"Model saved at epoch {epoch+1}")

            self.barrier()

        # Training is finished
        if self.master_process:
            print("Finished training!!!")

            # finish logging
            if self.use_wandb:
                wandb.finish()

        # destroy ddp process group
        if self.ddp:
            self.barrier()
            destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    ddp = int(os.environ.get('RANK', -1)) != -1
    trainer = VAETrainer(config_path=args.config, log_wandb=not args.no_wandb, ddp=ddp)
    trainer.train()
