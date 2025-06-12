import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat, rearrange
from torch.nn import functional as F

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class LinearAttention(nn.Module):
    """
    modified from https://github.com/HaoZhongkai/GNOT/blob/master/models/mmgpt.py
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
        super(LinearAttention, self).__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

    def forward(self, x, y=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1. / T2
        elif self.attn_type == "l2":  # still use l1 normalization
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y

def unified_pos_embedding(shapelist, ref, batchsize=1, device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    if len(shapelist) == 1:
        size_x = shapelist[0]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        grid_ref = gridx.reshape(1, ref, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
        pos = torch.sqrt(torch.sum((grid[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, ref).contiguous()
    if len(shapelist) == 2:
        size_x, size_y = shapelist[0], shapelist[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x * size_y, ref * ref).contiguous()
    if len(shapelist) == 3:
        size_x, size_y, size_z = shapelist[0], shapelist[1], shapelist[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B H W D 3

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1, 1).repeat([batchsize, 1, ref, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1, 1).repeat([batchsize, ref, 1, ref, 1])
        gridz = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, ref, 1).repeat([batchsize, ref, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, size_x * size_y * size_z, ref * ref * ref).contiguous()
    return pos

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,:1])], dim=-1)
    return embedding

class GNOT_block(nn.Module):
    """Transformer encoder block in MOE style."""

    def __init__(self, num_heads: int,
                 hidden_dim: int,
                 dropout: float,
                 act='gelu',
                 mlp_ratio=4,
                 space_dim=2,
                 n_experts=3):
        super(GNOT_block, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)

        self.selfattn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        self.crossattn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        self.resid_drop1 = nn.Dropout(dropout)
        self.resid_drop2 = nn.Dropout(dropout)

        ## MLP in MOE
        self.n_experts = n_experts
        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(space_dim, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim * mlp_ratio),
            self.act(),
            nn.Linear(hidden_dim * mlp_ratio, self.n_experts)
        )

    def forward(self, x, y, pos):
        ## point-wise gate for moe
        gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
        ## cross attention between geo and physics observation
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln2(y)))
        ## moe mlp
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
        x = x + self.ln3(x_moe1)
        ## self attention among geo
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        ## moe mlp
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
        x = x + self.ln5(x_moe2)
        return x


class Model(nn.Module):
    ## GNOT: Transformer in MOE style
    def __init__(self, args, n_experts=3):
        super(Model, self).__init__()
        self.__name__ = 'GNOT'
        self.args = args
        ## embedding
        self.preprocess_x = MLP(args.coor_dim, args.hidden_dim * 2, args.hidden_dim,
                                n_layers=0, res=False, act=args.act)
        self.preprocess_z = MLP(args.fun_dim + args.coor_dim, 
            args.hidden_dim * 2, args.hidden_dim,
            n_layers=0, res=False, act=args.act)

        ## models
        self.blocks = nn.ModuleList([GNOT_block(num_heads=args.num_head,
                                                hidden_dim=args.hidden_dim,
                                                dropout=False,
                                                act=args.act,
                                                mlp_ratio=4,
                                                space_dim=args.coor_dim,
                                                n_experts=n_experts)
                                     for _ in range(args.num_layers)])
        self.placeholder = nn.Parameter((1 / (args.hidden_dim)) * torch.rand(args.hidden_dim, dtype=torch.float))
        # projectors
        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim * 2)
        self.fc2 = nn.Linear(args.hidden_dim * 2, args.out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx=None, T=None, geo=None):
        pos = x
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess_z(fx)
        else:
            fx = self.preprocess_z(x)
        fx = fx + self.placeholder[None, None, :]
        x = self.preprocess_x(x)
        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(x, fx, pos)
        fx = self.fc1(fx)
        fx = F.gelu(fx)
        fx = self.fc2(fx)
        return fx

class GNOT_network(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.last_act = ACTIVATION[args.last_act]()

        self.model = Model(args=args)

    def forward(self, graph):
        return self.last_act(self.model(graph.x.unsqueeze(0))).squeeze(0)