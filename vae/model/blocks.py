import torch.nn as nn
from vae.utils import get_norm, get_activation


class downSampleBlock(nn.Module):
    """Downsampling block with residual connection"""

    def __init__(self, in_ch, out_ch, n_layers, norm_type='bn', norm_groups=None, activation='elu'):
        super(downSampleBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.activation = get_activation(activation)

        # residual conv block 1
        self.conv_layer1 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, in_ch if i == 0 else out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                )for i in range(n_layers)
            ])

        # residual conv block 2
        self.conv_layer2 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)   
                ) for _ in range(n_layers)
            ])
        
        # skip connection
        self.skip_connection = nn.ModuleList([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=1, stride=1, padding=0) 
                for i in range(n_layers)
            ])

        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        for i in range(self.n_layers):
            residual = x
            x = self.conv_layer1[i](x)
            x = self.conv_layer2[i](x)
            x = x + self.skip_connection[i](residual)

        return self.downsample(x)
    

class upSampleBlock(nn.Module):
    """Upsampling block with residual connection"""

    def __init__(self, in_ch, out_ch, n_layers, norm_type='bn', norm_groups=None, activation='elu'):
        super(upSampleBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.activation = get_activation(activation)

        # residual conv block 1
        self.conv_layer1 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, in_ch if i == 0 else out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                ) for i in range(n_layers)
            ])

        # residual conv block 2
        self.conv_layer2 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)   
                ) for _ in range(n_layers)
            ])
        
        # skip connection
        self.skip_connection = nn.ModuleList([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=1, stride=1, padding=0) 
                for i in range(n_layers)
            ])
        
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        for i in range(self.n_layers):
            residual = x
            x = self.conv_layer1[i](x)
            x = self.conv_layer2[i](x)
            x = x + self.skip_connection[i](residual)

        return x
    

class attentionBlock(nn.Module):
    """Attention block with residual connection"""

    def __init__(self, in_ch, out_ch, n_layers, num_heads, norm_type='bn', norm_groups=None, activation='elu'):
        super(attentionBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.activation = get_activation(activation)

        # residual conv block 1
        self.conv_layer1 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, in_ch if i == 0 else out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, stride=1, padding=1)
                ) for i in range(n_layers + 1)
            ])

        # residual conv block 2
        self.conv_layer2 = nn.ModuleList([
                nn.Sequential(
                    get_norm(norm_type, out_ch, norm_groups),
                    self.activation,
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)   
                ) for _ in range(n_layers + 1)
            ])
        
        # skip connection
        self.skip_connection = nn.ModuleList([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=1, stride=1, padding=0) 
                for i in range(n_layers + 1)
            ])
        
        # Multihead attention layers
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_ch, num_heads=num_heads, batch_first=True) 
            for _ in range(n_layers)
        ])

        # GroupNorm for attention
        self.attn_norms = nn.ModuleList([
            get_norm('gn', out_ch, norm_groups) 
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # Residual block before Attention
        residual = x
        x = self.conv_layer1[0](x)
        x = self.conv_layer2[0](x)
        x = x + self.skip_connection[0](residual)

        for i in range(self.n_layers):
            # Attention block
            b, c, h, w = x.shape
            x_attn = x.view(b, c, h * w)    # (b, c, h, w) -> (b, c, h*w) 
            x_attn = self.attn_norms[i](x_attn)  # (b, c, h*w) -> (b, h*w, c)
            x_attn = x_attn.transpose(1, 2)
            x_attn, _ = self.attns[i](x_attn, x_attn, x_attn)  # (b, h*w, c)
            x_attn = x_attn.transpose(1, 2).view(b, c, h, w)  # (b, h*w, c) -> (b, c, h*w) -> (b, c, h, w)
            x = x + x_attn

            # Residual block after Attention
            res = x
            x = self.conv_layer1[i + 1](x)
            x = self.conv_layer2[i + 1](x)
            x = x + self.skip_connection[i + 1](res)

        return x


