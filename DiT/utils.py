import torch
import torchvision
from torchvision.utils import make_grid


def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


def get_pos_embedding(pos_emb_dim, grid_size):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb


def sample_images(trainer, vae_scale, input_prompt, num_samples=8, cfg_guidance_scale=1.0):
    # grab references from trainer
    scheduler = trainer.scheduler
    dit = trainer.model
    vae = trainer.vae
    text_encoder = trainer.text_encoder
    device = trainer.device
    T = scheduler.timesteps

    trainer.model.eval()
    vae.eval()

    if trainer.text_encoder:
        assert input_prompt is not None, "input_prompt cannot be none when text_encoder is used"
        context, mask = text_encoder(input_prompt) # [N, L, D]
        context_empty, mask_empty = text_encoder([''])    # [1, L, D], [1, L]
        
        context_empty = context_empty.repeat(num_samples, 1, 1) # [1, L, D] -> [N, L, D]
        mask_empty = mask_empty.repeat(num_samples, 1)  # [1, L] -> [N, L]         

    x_t = torch.randn(num_samples, dit.latent_ch, dit.latent_size, dit.latent_size, device=device)
    with torch.no_grad():   
        for i in reversed(range(0, T)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            if text_encoder:
                x_b = torch.cat([x_t, x_t], dim=0)
                t_b = torch.cat([t, t], dim=0)
                y_b = torch.cat([context_empty, context], dim=0)
                mask_b = torch.cat([mask_empty, mask], dim=0)

                predicted_b = dit(x_b, t_b, y=y_b, mask=mask_b)
                pred_uncond, pred_cond = torch.chunk(predicted_b, 2, dim=0)
                predicted_noise = pred_uncond + cfg_guidance_scale * (pred_cond - pred_uncond)
            else:
                predicted_noise = dit(x_t, t)

            alphas_t = scheduler.alphas[t][:, None, None, None]
            alphas_cumprod_t = scheduler.alphas_cumprod[t][:, None, None, None]
            beta_t = scheduler.betas[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = 1 / torch.sqrt(alphas_t) *  (x_t - ((1.0 - alphas_t) / (torch.sqrt(1.0 - alphas_cumprod_t))) * predicted_noise) + torch.sqrt(beta_t) * noise

        out = vae.decode(x_t.mul(vae_scale))
        grid = make_grid(out, nrow=int(num_samples**0.5))
        img = torchvision.transforms.ToPILImage()(grid)
        return img
