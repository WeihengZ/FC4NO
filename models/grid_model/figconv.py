
import sys
import os
import torch.nn as nn
import torch

from .src import networks
from modulus.models.figconvnet.geometries import GridFeaturesMemoryFormat
from einops import rearrange

class figconv_based_model(nn.Module):
    def __init__(self, base_model, scale, scale2, aabb_max, aabb_min, point_feature_dim, patch_shape=None, out_channels=1, hidden_dim=16):
        super(figconv_based_model, self).__init__()

        if base_model == 'figconv':
            
            gridshape1 = int((aabb_max[0] - aabb_min[0]) * scale)
            gridshape2 = int((aabb_max[1] - aabb_min[1]) * scale)
            gridshape3 = int((aabb_max[2] - aabb_min[2]) * scale)
            p_shape1 = int((aabb_max[0] - aabb_min[0]) * scale2)
            p_shape2 = int((aabb_max[1] - aabb_min[1]) * scale2)
            p_shape3 = int((aabb_max[2] - aabb_min[2]) * scale2)


            if point_feature_dim == None:
                in_channels = hidden_dim
            else:
                in_channels = point_feature_dim

            self.model = networks.FIGConvUNetDrivAerNet(
              aabb_max=aabb_max,
              aabb_min=aabb_min,
              hidden_channels=[hidden_dim, hidden_dim, hidden_dim],
              in_channels=in_channels,
              kernel_size=5,
              mlp_channels=[2048, 2048],
              neighbor_search_type="radius",
              num_down_blocks=1,
              num_levels=2,
              out_channels=out_channels,
              pooling_layers=[2],
              pooling_type="max",
              reductions=["mean"],
              resolution_memory_format_pairs=[
                (GridFeaturesMemoryFormat.b_xc_y_z, [ p_shape1, gridshape2, gridshape3]),
                (GridFeaturesMemoryFormat.b_yc_x_z, [gridshape1,   p_shape2, gridshape3]),
                (GridFeaturesMemoryFormat.b_zc_x_y, [gridshape1, gridshape2,   p_shape3]),
              ],
              use_rel_pos_encode=False,    # True
            )
        elif base_model == 'gifno':
            gridshape1 = int((aabb_max[0] - aabb_min[0]) * scale)
            gridshape2 = int((aabb_max[1] - aabb_min[1]) * scale)
            gridshape3 = int((aabb_max[2] - aabb_min[2]) * scale)

            if point_feature_dim == None:
                in_channels = hidden_dim
            else:
                in_channels = point_feature_dim

            self.model = networks.FIGConvUNetDrivAerNet_fno(
              aabb_max=aabb_max,
              aabb_min=aabb_min,
              hidden_channels=[hidden_dim, hidden_dim, hidden_dim],
              in_channels=in_channels,
              kernel_size=5,
              mlp_channels=[2048, 2048],
              neighbor_search_type="radius",
              num_down_blocks=1,
              num_levels=2,
              out_channels=1,
              pooling_layers=[0],
              pooling_type="max",
              reductions=["mean"],
              resolution_memory_format_pairs=[
                (GridFeaturesMemoryFormat.b_xc_y_z, [gridshape1, gridshape2, gridshape3]),
                # (GridFeaturesMemoryFormat.b_yc_x_z, [250,   3, 100]),
                # (GridFeaturesMemoryFormat.b_zc_x_y, [250, 150,   2]),
              ],
              use_rel_pos_encode=True,
            )
        elif base_model == 'vt':

            patch_x, patch_y, patch_z = patch_shape

            gridshape1 = (int((aabb_max[0] - aabb_min[0]) * scale / patch_x) + 1) * patch_x
            gridshape2 = (int((aabb_max[1] - aabb_min[1]) * scale / patch_y) + 1) * patch_y
            gridshape3 = (int((aabb_max[2] - aabb_min[2]) * scale / patch_z) + 1) * patch_z

            # VT setting
            VT_settings = dict()
            VT_settings['input_shape'] = [gridshape1, gridshape2, gridshape3]
            VT_settings['patch_shape'] = (patch_x, patch_y, patch_z)

            if point_feature_dim == None:
                in_channels = hidden_dim
            else:
                in_channels = point_feature_dim

            self.model = networks.FIGConvUNetDrivAerNet_vt(
              aabb_max=aabb_max,
              aabb_min=aabb_min,
              hidden_channels=[hidden_dim, hidden_dim, hidden_dim],
              in_channels=in_channels,
              kernel_size=5,
              mlp_channels=[2048, 2048],
              neighbor_search_type="radius",
              num_down_blocks=1,
              num_levels=2,
              out_channels=out_channels,
              pooling_layers=[2],
              pooling_type="max",
              reductions=["mean"],
              resolution_memory_format_pairs=[
                (GridFeaturesMemoryFormat.b_xc_y_z, [gridshape1, gridshape2, gridshape3]),
                # (GridFeaturesMemoryFormat.b_yc_x_z, [250,   3, 100]),
                # (GridFeaturesMemoryFormat.b_zc_x_y, [250, 150,   2]),
              ],
              use_rel_pos_encode=True,
              VT_settings = VT_settings,
            )

    def forward(self, graph):

        if hasattr(graph, 'param'):
            graph_features = graph.param.unsqueeze(0)
            graph_features = graph_features.repeat(graph.x.shape[0],1)
            graph_features = torch.cat((graph.x, graph_features), -1)
            graph_features = graph_features.unsqueeze(0)
        
            point_clouds = graph.x.unsqueeze(0)
            out = self.model(point_clouds, graph_features)
            sol_pred = out[0].squeeze(0)
        
        else:

            point_clouds = graph.x.unsqueeze(0)
            out = self.model(point_clouds)
            sol_pred = out[0].squeeze(0)

        return sol_pred

def get_figconv_model(scale, scale2, aabb_max, aabb_min, point_feature_dim, out_channels=1, hidden_dim=16):

    model = figconv_based_model('figconv', scale, scale2, aabb_max, aabb_min, 
        point_feature_dim, out_channels=out_channels, hidden_dim = hidden_dim)

    return model

def get_figconv_fno_model(scale, aabb_max, aabb_min, point_feature_dim, out_channels=1, hidden_dim=16):

    model = figconv_based_model('gifno', scale, 0, aabb_max, aabb_min, 
        point_feature_dim, out_channels=out_channels, hidden_dim = hidden_dim)

    return model

def get_figconv_vt_model(scale, aabb_max, aabb_min, point_feature_dim, patch_shape, out_channels=1, hidden_dim=16):

    model = figconv_based_model('vt', scale, 0, aabb_max, aabb_min, 
        point_feature_dim, 
        patch_shape=patch_shape, out_channels=out_channels, hidden_dim=hidden_dim)

    return model

''' ----------------------- New model --------------------------- '''
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

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # B H N C

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)    # B H G C
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


class multi_grid_model(nn.Module):
    def __init__(self, scale, in_channels, out_channels, hidden_dim, aabb_max, aabb_min, resolution_memory_format_pairs):
        super().__init__()
        '''
        Using to_point_sample_method = "interp"
        '''

        final_hidden = hidden_dim
        self.in_channels = in_channels

        # encoder
        act = 'gelu'
        encoder_out_dim = hidden_dim*8
        self.encoder = MLP(in_channels, encoder_out_dim * 2, encoder_out_dim, n_layers=0, res=False, act=act)

        # attn blocks
        self.placeholder = nn.Parameter((1 / encoder_out_dim) * torch.rand(encoder_out_dim, dtype=torch.float))
        n_layers = 3
        self.attn_blocks = nn.ModuleList([
                Transolver_block(num_heads=4, hidden_dim=encoder_out_dim,
                    dropout=0.0, act=act, mlp_ratio=4, last_layer=(i == n_layers - 1), 
                    out_dim=final_hidden, slice_num=64)
                for i in range(n_layers)
            ])

        self.local_model = networks.FIGConvUNetDrivAerNet_multi_grid(
          hidden_channels=[16, 16, final_hidden],
          in_channels=hidden_dim,
          kernel_size=5,
          mlp_channels=[2048, 2048],
          neighbor_search_type="radius",
          aabb_max=aabb_max,
          aabb_min=aabb_min,
          num_down_blocks=1,
          num_levels=2,
          out_channels=final_hidden,
          pooling_layers=[2],
          pooling_type="max",
          reductions=["mean"],
          resolution_memory_format_pairs=resolution_memory_format_pairs,
          use_rel_pos_encode=True,
          scale=scale,
          to_point_sample_method = "attn"
        )

        self.outmap = nn.Sequential(   
            nn.Linear(final_hidden, 4*final_hidden),
            nn.GELU(),
            nn.Linear(4*final_hidden, 4*final_hidden), 
            nn.GELU(),
            nn.Linear(4*final_hidden, 4*final_hidden), 
            nn.GELU(),
            nn.Linear(4*final_hidden, out_channels), 
        )

    def forward(self, graph):
        '''
        x: (B, N, 3) - coordinates of points
        features: (B, N, input_dim) - features of points
        '''
        print(f"[CUDA] Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"[CUDA] Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        
        x = graph.x.unsqueeze(0)

        # # global mapping
        # xg = self.encoder(x) + self.placeholder[None, None, :]
        # for block in self.attn_blocks:
        #     xg = block(xg)
        
        # local mapping
        xl, _ = self.local_model(x)

        # feature combine
        x_total = xl # + xg

        # output
        x = self.outmap(x_total).squeeze(0)

        return x

def get_figconv_multi_model(data, scale, scale2, aabb_max, aabb_min, point_feature_dim, patch_shape, out_channels=1, hidden_dim=16):

    if point_feature_dim == None:
        in_channels = hidden_dim
    else:
        in_channels = point_feature_dim
    patch_x, patch_y, patch_z = patch_shape

    if data == 'driver' or data == 'driver_plus':

        model = multi_grid_model(
            scale = scale,
            in_channels = in_channels,
            out_channels = out_channels,
            hidden_dim = hidden_dim,
            aabb_max = aabb_max,
            aabb_min = aabb_min,
            resolution_memory_format_pairs=[
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*54, scale*22, scale*18], 
                #     [scale*2, scale*2, scale*2], aabb_max, aabb_min),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*54, scale*5, scale*18], 
                #     [scale*2, scale*1, scale*1], aabb_max, aabb_min),
                # (GridFeaturesMemoryFormat.b_x_y_z_c, [scale*54, scale*22, scale*5], 
                #     [scale*2, scale*2, scale*2], aabb_max, aabb_min),
                (GridFeaturesMemoryFormat.b_x_y_z_c, [8, 4, 4], 
                    [scale*1, scale*1, scale*1], [-0.25, aabb_max[1], 1.1], [aabb_min[0], aabb_min[1], aabb_min[2]]),

                # (GridFeaturesMemoryFormat.b_x_y_z_c, [scale*27, scale*11, scale*9], 
                #     [scale*1, scale*2, scale*1], [1.3, aabb_max[1], aabb_max[2]], [0.3, aabb_min[1], 1.0]),

                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*1, scale*2, scale*1], 
                #     [scale*1, scale*2, scale*1], [1.3, aabb_max[1], aabb_max[2]], [0.3, aabb_min[1], 1.0]),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*15, scale*22, scale*11], 
                #     [scale*1, scale*2, scale*1], [-0.25, aabb_max[1], 1.1], [aabb_min[0], aabb_min[1], aabb_min[2]]),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*10, scale*6, scale*7], 
                #     [scale*1, scale*1, scale*1], [0.5, -0.5, 0.7], [-0.5, aabb_min[1], aabb_min[2]]),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*10, scale*6, scale*7], 
                #     [scale*1, scale*1, scale*1], [3.2, -0.5, 0.7], [2.2, aabb_min[1], aabb_min[2]]),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*10, scale*6, scale*7], 
                #     [scale*1, scale*1, scale*1], [0.5, aabb_max[1], 0.7], [-0.5, 0.5, aabb_min[2]]),
                # (GridFeaturesMemoryFormat.b_xc_y_z, [scale*10, scale*6, scale*7], 
                #     [scale*1, scale*1, scale*1], [3.2, aabb_max[1], 0.7], [2.2, 0.5, aabb_min[2]]),
            ],
        )

    return model

















