import torch
import torchvision
from torchvision.utils import make_grid


def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_time_embedding(time_steps, temb_dim):
    pass


def get_pos_embedding(pos_emb_dim, grid_size):
    pass


def sample_images(dit, vae, scheduler, num_samples=8, device='cuda'):
    T = scheduler.timesteps
    dit.eval()
    vae.eval()

    x_t = torch.randn(num_samples, dit.latent_ch, dit.latent_size, dit.latent_size, device=device)
    with torch.no_grad():   
        for i in reversed(range(0, T)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            predicted_noise = dit(x_t, t)
            alphas_t = scheduler.alphas[t][:, None, None, None]
            alphas_cumprod_t = scheduler.alphas_cumprod[t][:, None, None, None]
            beta_t = scheduler.betas[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = 1 / torch.sqrt(alphas_t) *  (x_t - ((1.0 - alphas_t) / (torch.sqrt(1.0 - alphas_cumprod_t))) * predicted_noise) + torch.sqrt(beta_t) * noise

        out = vae.decode(x_t)
        grid = make_grid(out, nrow=int(num_samples**0.5))
        img = torchvision.transforms.ToPILImage()(grid)
        return img
