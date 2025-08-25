import torch
import torch.nn as nn
from DiT.utils import modulate, get_pos_embedding, get_time_embedding


class PatchEmbed(nn.Module):

    def __init__(self, latent_size, latent_ch, patch_size, embed_dim):
        super().__init__()
        self.latent_size = latent_size
        self.latent_ch = latent_ch
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(latent_ch * patch_size * patch_size, embed_dim)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.num_patches = self.latent_size // self.patch_size

        # save pos embedding
        pos = get_pos_embedding(self.embed_dim, (self.num_patches, self.num_patches))
        self.pos_embed = self.register_buffer('pos_embed', pos)

    def forward(self, x):
        x = self.unfold(x)  # (B, C, H, W) -> (B, C*P*P, N)
        x = x.permute(0, 2, 1)  # (B, C*P*P, N) -> (B, N, C*P*P)
        x = self.patch_embed(x) # (B, N, C*P*P) -> (B, N, embed_dim)
        x = x + self.pos_embed.to(x.device)  
        return x


class TimeEmbed(nn.Module):

    def __init__(self, time_emb_dim, embed_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        time_emb = get_time_embedding(t, self.time_emb_dim)
        return self.time_mlp(time_emb)
    

class DiT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.latent_size = config['latent_size']
        self.latent_ch = config['latent_ch']
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.time_emb_dim = config['time_emb_dim']

        self.patchify_block = PatchEmbed(self.latent_size, self.latent_ch, self.patch_size, self.embed_dim)
        self.time_embd = TimeEmbed(self.time_emb_dim, self.embed_dim)
        
        self.layers = nn.ModuleList([
            DiTBlock(config=config) for _ in range(self.num_layers)
        ])

        # scale and shift parameters for adaptive ln
        self.adaptive_norm_block = nn.Sequential(
            nn.SiLU(),     
            nn.Linear(self.embed_dim, 2 * self.embed_dim)
        )       
        self.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(self.embed_dim, self.latent_ch * self.patch_size * self.patch_size)

        nn.init.xavier_uniform_(self.adaptive_norm_block[-1].weight)
        nn.init.zeros_(self.adaptive_norm_block[-1].bias)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def unpatchify(self, x):
        b, n, _ = x.shape
        grid_size = int(n ** 0.5)
        x = x.reshape(b, grid_size, grid_size, self.patch_size, self.patch_size, self.latent_ch)  # (B, N, C*P*P) -> (B, G, G, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4) # (B, G, G, P, P, C) -> (B, C, G, P, G, P)
        x = x.reshape(b, self.latent_ch, grid_size * self.patch_size, grid_size * self.patch_size)  # (B, C, G, P, G, P) -> (B, C, H, W)
        return x
    
    def forward(self, x, t):
        x = self.patchify_block(x)        
        time_emb = self.time_embd(t)

        for layer in self.layers:
            x = layer(x, time_emb)  

        scale, shift = self.adaptive_norm_block(time_emb).chunk(2, dim=1)
        x = modulate(self.norm(x), scale, shift)
        
        x = self.out_proj(x) # (B, N, embed_dim) -> (B, N, C*P*P)
        x = self.unpatchify(x)  # (B, N, C*P*P) -> (B, C, H, W)
        return x
    

class DiTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.ffwd_hidden_dim = int(self.embed_dim * config.get('mlp_ratio', 1))
        
        # attn norm
        self.norm1 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.attn_block = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # ffwd norm
        self.norm2 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.ffwd_block = nn.Sequential(
            nn.Linear(self.embed_dim, self.ffwd_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.ffwd_hidden_dim, self.embed_dim)
        )

        # scale, shift for ln of attention: (2*embed_dim)
        # scale, shift for ln of ffwd: (2*embed_dim)
        # scale for attention residual connection: (embed_dim)
        # scale for ffwd residual connection: (embed_dim)
        # Total: (6*embed_dim)
        self.adaptive_norm_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim)
        )

        nn.init.xavier_uniform_(self.adaptive_norm_block[-1].weight)
        nn.init.zeros_(self.adaptive_norm_block[-1].bias)
 
    def forward(self, x, cond):
        affine_params = self.adaptive_norm_block(cond).chunk(6, dim=1)
        scale_attn, shift_attn, scale_res_attn, scale_ffwd, shift_ffwd, scale_res_ffwd = affine_params
        x_attn = modulate(self.norm1(x), scale_attn, shift_attn) # LN(x)*(1 + scale(cond)) + shift(cond)
        x_attn, _ = self.attn_block(x_attn, x_attn, x_attn) 
        x = x + x_attn * scale_res_attn.unsqueeze(1)

        x_ffwd = modulate(self.norm2(x), scale_ffwd, shift_ffwd) # LN(x)*(1 + scale(cond)) + shift(cond)
        x_ffwd = self.ffwd_block(x_ffwd)
        x = x + x_ffwd * scale_res_ffwd.unsqueeze(1)    
        return x