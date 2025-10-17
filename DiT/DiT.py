import torch
import torch.nn as nn
from DiT.utils import modulate, get_pos_embedding, get_time_embedding
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPModel
from transformers import AutoTokenizer, AutoModel


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
        self.register_buffer('pos_embed', pos)

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


class TextEmbed():

    def __init__(self, text_embed_model, dropout_prob, max_length=77, device='cuda'):
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        self.device = device

        assert text_embed_model in ('bert', 'clip'), "Text model can only be one of bert, clip"

        if text_embed_model == 'bert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_embed_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16')
            self.text_embed_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').text_model.to(self.device)
      
        self.text_embed_model.eval()

    def encode(self, text):
        tokens = self.tokenizer(
            text, 
            padding='max_length',    
            truncation=True,          
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_emb = self.text_embed_model(input_ids, attention_mask).last_hidden_state

        return text_emb, attention_mask   # [B, L, D]
        
    def __call__(self, texts, apply_dropout=False):
        text_emb, attn_mask = self.encode(texts)
        empty_text_emb, empty_mask = self.encode([''])

        # apply conditional dropout 
        if apply_dropout and self.dropout_prob > 0:
            drop_mask = torch.rand(text_emb.shape[0], device=self.device) < self.dropout_prob
            text_emb[drop_mask, :, :] = empty_text_emb[0]
            attn_mask[drop_mask, :] = empty_mask[0] 

        attn_mask = ~attn_mask.bool()

        return text_emb, attn_mask


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
        self.text_embed_dim = config['cond_config']['text_emb_dim']

        self.patchify_block = PatchEmbed(self.latent_size, self.latent_ch, self.patch_size, self.embed_dim)
        self.time_embd = TimeEmbed(self.time_emb_dim, self.embed_dim)
        self.text_embd = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
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
    
    def forward(self, x, t, y=None, mask=None):
        x = self.patchify_block(x)        
        time_emb = self.time_embd(t)

        if y is not None:
            y = self.text_embd(y)

        for layer in self.layers:
            x = layer(x, time_emb, y=y, mask=mask)  

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
        
        # self-attn 
        self.norm1 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # cross-attn
        self.norm2 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # ffwd norm
        self.norm3 = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

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
 
    def forward(self, x, t, y=None, mask=None):
        affine_params = self.adaptive_norm_block(t).chunk(6, dim=1)
        scale_attn, shift_attn, scale_res_attn, scale_ffwd, shift_ffwd, scale_res_ffwd = affine_params
        x_attn = modulate(self.norm1(x), scale_attn, shift_attn) # LN(x)*(1 + scale(t)) + shift(t)
        x_attn, _ = self.self_attn(x_attn, x_attn, x_attn) 
        x = x + x_attn * scale_res_attn.unsqueeze(1)

        if y is not None: # add cross-attn residual
            x = x + self.cross_attn(self.norm2(x), y, y, key_padding_mask=mask)[0]

        x_ffwd = modulate(self.norm3(x), scale_ffwd, shift_ffwd) # LN(x)*(1 + scale(t)) + shift(t)
        x_ffwd = self.ffwd_block(x_ffwd)
        x = x + x_ffwd * scale_res_ffwd.unsqueeze(1)    
        return x