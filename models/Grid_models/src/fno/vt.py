import torch
import torch.nn as nn
from einops import rearrange

# ---------- Activation Dictionary ----------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}

# ---------- Irregular Mesh Attention ----------
class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None])

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_token = torch.matmul(attn, v)  # B H G D

        out_x = torch.einsum("bhgc,bhng->bhnc", out_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

# ---------- MLP ----------
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)])
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.res = res

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)

# ---------- Transformer Block ----------
class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx



class PatchEmbed3D(nn.Module):
    def __init__(self, input_shape, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size  # (p1, p2, p3)
        self.grid_size = (
            input_shape[0] // patch_size[0],
            input_shape[1] // patch_size[1],
            input_shape[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, M, N, L, F) -> (B, F, M, N, L)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # (B, embed_dim, M', N', L')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT3D(nn.Module):
    def __init__(self, input_shape=(32, 32, 32), patch_size=(4, 4, 4), in_channels=1, embed_dim=128,
                 depth=4, num_heads=4, mlp_dim=256, out_channels=None):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.patch_embed = PatchEmbed3D(input_shape, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # self.blocks = nn.Sequential(*[
        #     TransformerEncoder(embed_dim, num_heads, mlp_dim)
        #     for _ in range(depth)
        # ])
        self.blocks = nn.ModuleList([
            Transolver_block(8, embed_dim, 0.0, 'gelu', 4, last_layer=(i == depth - 1), out_dim=embed_dim, slice_num=32)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.patch_unembed = nn.ConvTranspose3d(embed_dim, self.out_channels,
                                                kernel_size=patch_size, stride=patch_size)
        
        # spatial convolution
        self.final_conv1 = nn.Conv3d(
            self.out_channels,  # input channels = output from patch_unembed
            self.out_channels,  # or change to something else if desired
            kernel_size=3,
            padding=1
        )
        self.final_conv2 = nn.Conv3d(
            self.out_channels,  # input channels = output from patch_unembed
            self.out_channels,  # or change to something else if desired
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        '''
        x: (B, M, N, L, F)
        output: (B, M, N, L, F)
        '''
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        x = x + self.pos_embed[:, :x.size(1), :]
        # x = self.dropout(x)
        # x = self.blocks(x)
        # x = self.norm(x)  # (B, num_patches, embed_dim)
        for block in self.blocks:
            x = block(x)

        # Convert back to (B, C, M', N', L')
        grid = (
            self.input_shape[0] // self.patch_size[0],
            self.input_shape[1] // self.patch_size[1],
            self.input_shape[2] // self.patch_size[2],
        )
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *grid)

        # Upsample back to (B, out_channels, M, N, L)
        x = self.patch_unembed(x)  # (B, out_channels, M, N, L)

        # CNN layers
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, M, N, L, F)

        return x  # (B, M, N, L, F)