
import sys
import os
import torch.nn as nn
import torch

from .src import networks
from modulus.models.figconvnet.geometries import GridFeaturesMemoryFormat
from einops import rearrange

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class figconv_based_model(nn.Module):
    def __init__(self, args):
        super(figconv_based_model, self).__init__()

        # extract the parameters
        base_model = args.model
        aabb_max = args.aabb_max
        aabb_min = args.aabb_min
        point_feature_dim = args.coor_dim
        patch_shape = args.patch_shape if args.patch_shape is not None else None
        out_channels = args.out_dim
        hidden_dim = args.hidden_dim
        scale = args.grid_scale
        scale2 = args.grid_scale_low
        depth = args.num_layers
        fno_mode = args.fno_mode
        self.last_act = ACTIVATION[args.last_act]()

        if base_model == 'figconv':
            
            gridshape1 = int((aabb_max[0] - aabb_min[0]) * scale)
            gridshape2 = int((aabb_max[1] - aabb_min[1]) * scale)
            gridshape3 = int((aabb_max[2] - aabb_min[2]) * scale)
            p_shape1 = int((aabb_max[0] - aabb_min[0]) * scale2)
            p_shape2 = int((aabb_max[1] - aabb_min[1]) * scale2)
            p_shape3 = int((aabb_max[2] - aabb_min[2]) * scale2)

            # set the input channels
            if point_feature_dim == None:
                in_channels = hidden_dim
            else:
                in_channels = point_feature_dim

            self.model = networks.FIGConvUNetDrivAerNet(
              aabb_max=aabb_max,
              aabb_min=aabb_min,
              hidden_channels=[hidden_dim for _ in range(depth)],
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
              act = ACTIVATION[args.act]
            )

        elif base_model == 'gifno':
            gridshape1 = int((aabb_max[0] - aabb_min[0]) * scale)
            gridshape2 = int((aabb_max[1] - aabb_min[1]) * scale)
            gridshape3 = int((aabb_max[2] - aabb_min[2]) * scale)

            if point_feature_dim == None:
                in_channels = hidden_dim
            else:
                in_channels = point_feature_dim
            
            # adjust the mode if the resolution is not enough
            fno_mode = min(gridshape1//2, gridshape2//2, gridshape3//2, fno_mode) - 1

            self.model = networks.FIGConvUNetDrivAerNet_fno(
              aabb_max=aabb_max,
              aabb_min=aabb_min,
              hidden_channels=[hidden_dim for _ in range(depth)],
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
                (GridFeaturesMemoryFormat.b_x_y_z_c, [gridshape1, gridshape2, gridshape3]),
              ],
              use_rel_pos_encode=True,
              act = ACTIVATION[args.act],
              fno_mode=fno_mode
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
              hidden_channels=[hidden_dim for _ in range(depth)],
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
              ],
              use_rel_pos_encode=True,
              VT_settings = VT_settings,
              act = ACTIVATION[args.act]
            )

    def forward(self, graph):

        point_clouds = graph.x.unsqueeze(0)
        out = self.model(point_clouds[:,:,:3], point_clouds)
        sol_pred = out[0].squeeze(0)

        return self.last_act(sol_pred)

def get_figconv_model(args):

    model = figconv_based_model(args)

    return model

def get_figconv_fno_model(args):

    model = figconv_based_model(args)

    return model

def get_figconv_vt_model(args):

    model = figconv_based_model(args)

    return model

