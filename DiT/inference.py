from DiT.DiT import DiT, TextEmbed
from DiT.utils import sample_images
from diffusion.linear_noise_scheduler import LinearNoiseScheduler
from vae.model.vae import VAE
from pathlib import Path
import yaml
import random
import numpy as np
import torch

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('/home/users/ntu/ccdsnirm/projects/DiffuseNet/DiT/configs/config_cfg.yaml')
    dit_config = config['dit']
    vae_config = config['vae']
    scheduler_config = config['scheduler']
    train_config = config['training']
    cond_config = dit_config['cond_config']
    device = 'cuda'

    # set a seed for reproducibility
    seed = train_config['seed']
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    dit = DiT(config=dit_config).to(device)
    dit.load_state_dict(torch.load('/home/users/ntu/ccdsnirm/scratch/DiT/ckpts/cfg-test-pokemon/model_epoch_800.pth'), strict=False)
    vae = VAE(config=vae_config).to(device)
    vae.load_state_dict(torch.load('/home/users/ntu/ccdsnirm/projects/DiffuseNet/vae/ckpts/pokemon/vae_epoch_300.pth'))
    scheduler = LinearNoiseScheduler(
        timesteps=scheduler_config['timesteps'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end'],
        device=device)
    text_encoder = TextEmbed(text_embed_model=cond_config['text_embed_model'], 
                             dropout_prob=cond_config['dropout_prob'], device=device)
    captions = ['Ash and Misty'] 
    out = sample_images(dit, vae, scheduler, train_config['vae_scale'], captions, text_encoder, 
                        150, num_samples=1, device=device)
    out.save('img.png')
    out.close()

if __name__ == '__main__':
    main()