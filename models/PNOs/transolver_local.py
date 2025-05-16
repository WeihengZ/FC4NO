import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# ---------- Activation Dictionary ----------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU
}

# ---------- Irregular Mesh Attention ----------
class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv):  # x_q: B H G1 F, x_kv: B H G2 F
        q = self.to_q(x_q)
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # B H G1 G2
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)  # B H G1 F

        return self.to_out(out)

class Physics_Attention_Irregular_Mesh_Hierarchical(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # token mapping
        self.token_map1 = nn.Linear(dim_head, dim_head)
        self.token_map2 = nn.Linear(dim_head, dim_head)

        # Hierarchical cross-attention modules
        self.cross_attn1 = CrossAttention(dim, heads, dim_head)
        self.cross_attn2 = CrossAttention(dim, heads, dim_head)
        self.cross_attn3 = CrossAttention(dim, heads, dim_head)

        # Final output
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        '''
        x: (B N F)
        '''
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        slice_logits = self.in_project_slice(x_mid)
        slice_weights = self.softmax(slice_logits / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token_layer1 = slice_token / ((slice_norm + 1e-5)[..., None])  # B H G F

        # Hierarchical pooling
        def pool_token(x, factor):
            B, H, G, F = x.shape
            G_new = G // factor
            x = x[:, :, :G_new * factor, :].reshape(B, H, G_new, factor, F).mean(dim=3)
            return x

        slice_token_layer2 = self.token_map1(pool_token(slice_token_layer1, 4))
        slice_token_layer3 = self.token_map2(pool_token(slice_token_layer2, 4))

        # Cross-attention refinement
        slice_token_layer3 = self.cross_attn1(slice_token_layer3, slice_token_layer3)
        slice_token_layer2 = self.cross_attn2(slice_token_layer2, slice_token_layer3)
        slice_token_layer1 = self.cross_attn3(slice_token_layer1, slice_token_layer2)

        # Deslice back to original points
        out_x = torch.einsum("bhgc,bhng->bhnc", slice_token_layer1, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)

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
        # self.Attn = Physics_Attention_Irregular_Mesh_Hierarchical(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num)
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
    def __init__(self, space_dim=3, n_layers=3, n_hidden=256, dropout=0.0, n_head=8, act='gelu', mlp_ratio=4, fun_dim=1, out_dim=1, slice_num=64, unified_pos=False, ref=8):
        super().__init__()
        self.unified_pos = unified_pos
        self.ref = ref
        input_dim = fun_dim + (ref**3 if unified_pos else space_dim)

        self.preprocess = MLP(input_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.blocks = nn.ModuleList([
            Transolver_block(n_head, n_hidden, dropout, act, mlp_ratio, last_layer=False, out_dim=out_dim, slice_num=slice_num)
            for i in range(n_layers)
        ])
        self.local_blocks = nn.ModuleList([
            Transolver_block(n_head, n_hidden, dropout, act, mlp_ratio, last_layer=(i == n_layers - 1), out_dim=out_dim, slice_num=slice_num)
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

        # new features

    
    def forward(self, graph, num_sampled_clusters=50):
        # === Step 1: Sample random clusters ===
        all_x = graph.x  # list of (N_i, 3)
        all_y = graph.y  # list of (N_i,)
        num_clusters = len(all_x)

        # Randomly select `num_sampled_clusters` indices
        sampled_indices = np.random.randint(0, num_clusters, min(num_sampled_clusters, num_clusters))
        
        # Selected clusters
        x_list = [all_x[i] for i in sampled_indices]
        y_list = [all_y[i] for i in sampled_indices]
        cluster_sizes = [x.shape[0] for x in x_list]
        cluster_offsets = [0] + list(torch.cumsum(torch.tensor(cluster_sizes), dim=0)[:-1])

        # === Step 2: Concatenate all sampled points ===
        x_all = torch.cat(x_list, dim=0).unsqueeze(0)  # (1, N, 3)

        # === Step 3: Global coordinate embedding ===
        coor_emb_all = self.preprocess(x_all) + self.placeholder[None, None, :]  # (1, N, F)

        # === Step 4: Global attention ===
        fx = coor_emb_all
        for block in self.blocks:
            fx = block(fx)  # (1, N, F)

        # === Step 5: Per-cluster local attention ===
        all_pred = []
        for i in range(len(sampled_indices)):
            start = cluster_offsets[i]
            end = start + cluster_sizes[i]

            local_feat = fx[:, start:end, :]  # (1, N_i, F)
            # global_feat = torch.mean(local_feat, dim=1, keepdim=True)  # (1, 1, F)

            local_feature = local_feat # torch.cat([local_feat, global_feat.repeat(1, cluster_sizes[i], 1)], dim=-1)  # (1, N_i, 2F)

            for block in self.local_blocks:
                local_feature = block(local_feature)

            all_pred.append(local_feature.squeeze(0))  # (N_i, F_out)

        # === Step 6: Final concatenation ===
        all_pred = torch.cat(all_pred, dim=0)  # (N_total, F_out)
        ygt = torch.cat(y_list, dim=0).unsqueeze(-1)  # (N_total, 1)

        return all_pred, ygt
    
    def predict(self, graph):
        # === Step 1: Flatten the clusters ===
        x_list = graph.x  # List of (N_i, 3)
        y_list = graph.y  # List of (N_i,)
        cluster_sizes = [x.shape[0] for x in x_list]
        cluster_offsets = [0] + list(torch.cumsum(torch.tensor(cluster_sizes), dim=0)[:-1])
        num_clusters = len(x_list)

        # [N_total, 3]
        x_all = torch.cat(x_list, dim=0).unsqueeze(0)  # Add batch dim: (1, N, 3)

        # === Step 2: Preprocess (e.g., coord embedding) and Global Attention ===
        # (1, N, F)
        coor_emb_all = self.preprocess(x_all) + self.placeholder[None, None, :]  # broadcasting placeholder

        # Pass through global attention blocks
        global_features = coor_emb_all
        for block in self.blocks:
            global_features = block(global_features)  # (1, N, F)

        # === Step 3: Per-cluster processing ===
        all_pred = []
        for i in range(num_clusters):
            start = cluster_offsets[i]
            end = start + cluster_sizes[i]

            # Extract local feature from global
            local_feat = global_features[:, start:end, :]  # (1, N_i, F)
            # global_feat_cluster = torch.mean(local_feat, dim=1, keepdim=True)  # (1, 1, F)

            # Combine local + global
            local_feature = local_feat # torch.cat([local_feat, global_feat_cluster.repeat(1, cluster_sizes[i], 1)], dim=-1)

            for block in self.local_blocks:
                local_feature = block(local_feature)  # (1, N_i, F')

            all_pred.append(local_feature.squeeze(0))  # remove batch dim

        # === Step 4: Concatenate predictions and ground truth ===
        ygt = torch.cat(y_list, dim=0).unsqueeze(-1)  # (N, 1)
        all_pred = torch.cat(all_pred, dim=0)         # (N, F')

        return all_pred, ygt

    def forward2(self, graph):

        # get the information
        num_cluster = len(graph.x)

        # compute the coordinate embedding
        coor_emb = []
        for i in range(num_cluster):
            coor_emb.append(self.preprocess(graph.x[i]) + self.placeholder[None, None, :])    # list of (B N F)
        
        # obtain the global feature
        global_features = torch.cat(tuple(coor_emb), 1)    # B C F
        fx = global_features
        for block in self.blocks:
            fx = block(fx)
        gt = torch.cat(graph.y, 0).unsqueeze(-1)

        return fx[0], gt  # return shape: (N, out_dim)
    
    def predict2(self, graph):

        # get the information
        num_cluster = len(graph.x)

        # compute the coordinate embedding
        coor_emb = []
        for i in range(num_cluster):
            coor_emb.append(self.preprocess(graph.x[i]) + self.placeholder[None, None, :])    # list of (B N F)
        
        # obtain the global feature
        global_features = torch.cat(tuple(coor_emb), 1)    # B C F
        fx = global_features
        for block in self.blocks:
            fx = block(fx)
        gt = torch.cat(graph.y, 0).unsqueeze(-1)

        return fx[0], gt  # return shape: (N, out_dim)

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]
        gridx = torch.linspace(-1.5, 1.5, self.ref).reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.linspace(0, 2, self.ref).reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.linspace(-4, 4, self.ref).reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(my_pos.device).reshape(batchsize, self.ref**3, 3)
        pos = torch.sqrt(torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1))
        return pos.reshape(batchsize, my_pos.shape[1], -1)

# ---------- Wrapper Network ----------
class Transolver_local_network(nn.Module):
    def __init__(self, space_dim=3, fun_dim=1, n_layers=5, n_hidden=256, out_dim=1, dropout=0.0, n_head=8, dim_head=64, mlp_ratio=4, slice_num=64, unified_pos=False, ref=8):
        super().__init__()
        self.model = Model(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act='gelu',
            mlp_ratio=mlp_ratio,
            fun_dim=fun_dim,
            out_dim=out_dim,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref
        )

    def forward(self, graph):
        return self.model(graph)
    
    def predict(self, graph):
        return self.model.predict(graph)
