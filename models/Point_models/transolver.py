import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# ---------- Activation Dictionary ----------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
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

# ---------- Full Model Core ----------
class Model(nn.Module):
    def __init__(self, space_dim=3, n_layers=5, n_hidden=256, dropout=0.0, n_head=8, act='gelu', mlp_ratio=4, fun_dim=1, out_dim=1, slice_num=64, unified_pos=False, ref=8):
        super().__init__()
        self.unified_pos = unified_pos
        self.ref = ref
        input_dim = fun_dim + (ref**3 if unified_pos else space_dim)

        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.blocks = nn.ModuleList([
            Transolver_block(n_head, n_hidden, dropout, act, mlp_ratio, last_layer=(i == n_layers - 1), out_dim=out_dim, slice_num=slice_num)
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def forward(self, graph):
        N = graph.x.shape[0]
        x = graph.x.unsqueeze(0)  # shape: (1, N, space_dim)
        if self.unified_pos:
            pos = self.get_grid(graph.pos.unsqueeze(0))  # B N ref^3
            x = torch.cat([x, pos], dim=-1)

        fx = self.preprocess(x) + self.placeholder[None, None, :]  # shape: (1, N, hidden)
        for block in self.blocks:
            fx = block(fx)
        return fx[0]  # return shape: (N, out_dim)
    
    def predict(self, graph):
        x = graph.x.unsqueeze(0)  # shape: (1, N, space_dim)
        if self.unified_pos:
            pos = self.get_grid(graph.pos.unsqueeze(0))  # B N ref^3
            x = torch.cat([x, pos], dim=-1)

        fx = self.preprocess(x) + self.placeholder[None, None, :]  # shape: (1, N, hidden)
        for block in self.blocks:
            fx = block(fx)
        return fx[0]  # return shape: (N, out_dim)

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]
        gridx = torch.linspace(-1.5, 1.5, self.ref).reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.linspace(0, 2, self.ref).reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.linspace(-4, 4, self.ref).reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(my_pos.device).reshape(batchsize, self.ref**3, 3)
        pos = torch.sqrt(torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1))
        return pos.reshape(batchsize, my_pos.shape[1], -1)

# ---------- Wrapper Network ----------
class Transolver_network(nn.Module):
    def __init__(self, args):
        super().__init__()

        space_dim = args.coor_dim
        fun_dim = 0
        n_layers = args.num_layers
        n_hidden = args.hidden_dim
        out_dim = args.out_dim
        dropout = False
        n_head = args.num_head
        dim_head = int(args.hidden_dim / args.num_head)
        mlp_ratio = 4
        slice_num = args.slice_num
        unified_pos = False
        ref = 8
        self.last_act = ACTIVATION[args.last_act]()

        self.model = Model(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=args.act,
            mlp_ratio=mlp_ratio,
            fun_dim=fun_dim,
            out_dim=out_dim,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref
        )

    def forward(self, graph):
        return self.last_act(self.model(graph))
    
    def predict(self, graph):
        return self.last_act(self.model.predict(graph))
