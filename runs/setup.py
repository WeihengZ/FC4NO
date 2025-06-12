import torch
import torch.optim as optim
import numpy as np
import os
from torch_geometric.data import Data, DataLoader
import argparse
import time
import torch.nn as nn
import yaml

import sys
sys.path.append(r'./')
from utils.utils_data import create_data_loaders
from utils.utils_train import train_model, test_model, train_grid_model
from utils.utils_plot import plot_model, plot_model_jet, plot_model_driver

from models.Branch_trunk_models.deepOnet import DON
from models.Branch_trunk_models.SDON import S_DeepONet
from models.Branch_trunk_models.dcon import DCON
from models.Branch_trunk_models.gano import GANO
from models.Branch_trunk_models.geom_don import Geom_DeepONet

from models.Graph_models.gno import GNO
from models.Graph_models.eagno import EAGNO
from models.Graph_models.GraphUNet import G_Unet

try:
    from models.Grid_models.figconv import get_figconv_model, get_figconv_fno_model, get_figconv_vt_model
    No_nvidia = False
except:
    No_nvidia = True
    print('No Nvidia modulus is detected.')

from models.Point_models.pointnet import PointNet
from models.Point_models.transolver import Transolver_network
from models.Point_models.gnot import GNOT_network

# define data path
data_loc_dict = {
    'heatsink': '/taiga/illinois/eng/cee/meidani/Vincent/heatsink/Processed/',
    'driver': '/taiga/illinois/eng/cee/meidani/Vincent/driver/Processed/',
    'driver_plus': '/taiga/illinois/eng/cee/meidani/Vincent/driver_plus/Processed/',
    'bracket': '/taiga/illinois/eng/cee/meidani/Vincent/bracket_static/Processed/',
    'bracket_time': '/taiga/illinois/eng/cee/meidani/Vincent/bracket_time/Processed/',
    'jet': '/taiga/illinois/eng/cee/meidani/Vincent/JEB/Processed/'
}

# define model path
model_dict = {
    'DeepONet': DON,
    'SDON': S_DeepONet,
    'geomDON': Geom_DeepONet,
    'DCON': DCON,
    'GANO': GANO,
    'GNO': GNO,
    'EAGNO': EAGNO,
    'GUNet': G_Unet,
    'figconv': None if No_nvidia else get_figconv_model,
    'gifno': None if No_nvidia else get_figconv_fno_model,
    'vt': None if No_nvidia else get_figconv_vt_model,
    'pointnet': PointNet,
    'transolver': Transolver_network,
    'gnot': GNOT_network,
}

# define non-geometric models
non_geometric_model_dict = {
    'DeepONet': DON,
    'SDON': S_DeepONet,
    'DCON': DCON,
    'GeomDON': Geom_DeepONet,
}

Attn_model_dict = {
    'transolver': Transolver_network,
    'gnot': GNOT_network,
}

def set_arguments(args):

    # extract the configuration
    with open("./configs/data/{}.yaml".format(args.data), "r") as f:
        data_config = yaml.safe_load(f)
    with open("./configs/model/{}.yaml".format(args.model), "r") as f:
        model_config = yaml.safe_load(f)
    
    # set the arguments
    args.act = model_config['act']
    args.last_act = data_config['last_act']
    args.num_layers = model_config['num_layers']
    args.hidden_dim = model_config['hidden_dim']
    args.coor_dim = data_config['coor_dim']
    args.geo_param_dim = data_config['geo_param_dim']
    args.load_param_dim = data_config['load_param_dim']
    args.out_dim = data_config['out_dim']
    args.aabb_min = data_config['aabb_min']
    args.aabb_max = data_config['aabb_max']
    args.grid_scale = data_config['grid_scale']
    args.grid_scale_low = data_config['grid_scale_low']
    args.patch_shape = model_config['patch_shape'] if 'patch_shape' in model_config else None
    args.grid_base_model = args.data
    args.slice_num = model_config['slice_num'] if 'slice_num' in model_config else None
    args.num_head = model_config['num_head'] if 'num_head' in model_config else None
    args.fno_mode = model_config['fno_mode'] if 'fno_mode' in model_config else None

    return args

# vanilla model
class Vanilla_Model(nn.Module):
    def __init__(self, args):
        super(Vanilla_Model, self).__init__()

        if args.model in non_geometric_model_dict:
            self.non_geometric_model = True
        else:
            self.non_geometric_model = False
            args.coor_dim = args.coor_dim + args.load_param_dim
        
        if args.data == 'driver_plus' or args.data == 'driver':
            if args.model in Attn_model_dict:
                args.hidden_dim = 16

        self.backend = model_dict[args.model](args)

        # set the time-dependent parameters
        if args.data == 'bracket_time':
            self.time_dependent = True
        else:
            self.time_dependent = False
        
        if args.model == 'SDON':
            self.time_dependent_model = True
        else:
            self.time_dependent_model = False

        

    def extract_params(self, graph):

        # extract the 
        params = []
        if hasattr(graph, 'geo_params'):
            params.append(graph.geo_params)
        if hasattr(graph, 'load_params'):
            params.append(graph.load_params) 
        if len(params) > 0:
            params = torch.cat(tuple(params), dim=-1)   # (branch_dim)
            graph.params = params
        
        # set time-dependent parameters

        return graph
    
    def extract_time_dependent_params(self, graph):
        graph.time_dependent_params = graph.load_params

        return graph

    def forward(self, graph):
        
        graph = self.extract_params(graph)
        if self.time_dependent:
            graph = self.extract_time_dependent_params(graph)

        # concate the loading information
        if hasattr(graph, 'load_params') and not self.non_geometric_model:
            if not self.time_dependent_model:
                graph.x = torch.cat([graph.x, 
                    graph.load_params.unsqueeze(0).repeat(graph.x.shape[0], 1)], dim=-1)
        
        out = self.backend(graph)

        return out

# Geometric model with parameter representation concatenated
class Param_concat_geometric_Model(nn.Module):
    def __init__(self, args):
        super(Vanilla_Model, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph):
        if args.model in non_geometric_model_dict:
            assert False, 'Param_concat_geometric_Model is not supported for non-geometric models'
        
        params = graph.geo_param
        x = graph.x
        graph.x = torch.cat([x, params], dim=-1)

        out = self.backend(graph)
        return out


# Geometric model with branch-trunk enhancement
class Param_branch_trunk_geometric_Model(nn.Module):
    def __init__(self, args):
        super(Param_branch_trunk_geometric_Model, self).__init__()
        act = activation_dict[args.act]
        self.branch = nn.Sequential(
            nn.Linear(args.coor_dim, args.hidden_dim),
            act,
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.branch = nn.Sequential(
            nn.Linear(args.geo_param_dim, args.hidden_dim),
            act,
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.backend = Model_dict[args.model]()
    
    def forward(self, graph):
        
        if hasattr(graph, 'geo_param'):
            if hasattr(graph, 'load_param'):
                params = torch.cat([graph.geo_param, graph.load_param], dim=1)
            else:
                params = graph.geo_param
        
        x = graph.x