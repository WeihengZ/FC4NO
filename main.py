import torch
import torch.optim as optim
import numpy as np
import os
from torch_geometric.data import Data, DataLoader
import argparse
import time

import sys
sys.path.append(r'./')
from utils_data import create_data_loaders
from utils import train_model, test_model, train_grid_model
from utils_plot import plot_model, plot_model_jet, plot_model_driver

from models.DeepONets.deepOnet import DON
from models.DeepONets.SDON import S_DeepONet
from models.DeepONets.dcon import DCON
from models.DeepONets.gano import GANO
from models.DeepONets.geom_don import Geom_DeepONet

from models.GNOs.gno import GNO
from models.GNOs.eagno import EAGNO
from models.GNOs.GraphUNet import G_Unet

try:
    from models.grid_model.figconv import get_figconv_model, get_figconv_fno_model, get_figconv_vt_model
except:
    print('No Nvidia modulus is detected.')

from models.PNOs.transformer import Transformer
from models.PNOs.pointnet import PointNet
from models.PNOs.transolver import Transolver_network
from models.PNOs.transolver_local import Transolver_local_network

# arguments
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'plot'])
parser.add_argument('--model', type=str, default='transolver', choices=['DON', 'GANO', 'SDON', 'DCON', 'geomDON', 'GNO', 'EAGNO', 'GUnet',  'figconv', 'gifno', 'vt', 'transformer', 'pointnet', 'transolver'])
parser.add_argument('--data', type=str, default='plastic', choices=['heatsink', 'plastic', 'elastic', 'jet', 'driver', 'driver_plus'])
args = parser.parse_args()

# load the data
if 1==1:
    if args.data == 'heatsink':
        data_path_name = 'heatsink'
        scale_factor = [296.45, 343.75]
        branch_dim=4; control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim + branch_dim
        point_feature_dim = trunk_dim + branch_dim
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5
    if args.data == 'heatsink2':
        data_path_name = 'heatsink'
        scale_factor = [296.45, 343.75]
        branch_dim=4; control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim 
        point_feature_dim = 16
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5

    if args.data == 'plastic':
        scale_factor = [0.0, 327.38]
        branch_dim=101; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim + branch_dim
        point_feature_dim = gnn_trunk_dim
        aabb_max, aabb_min = [1.1, 0.4, 0.2], [-0.2, -0.1, 0.0]
        scale=100; scale2=5
    if args.data == 'elastic':
        data_path_name = 'elastic'
        scale_factor = [0.0, 758.23]
        branch_dim=4; control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim + branch_dim
        point_feature_dim = gnn_trunk_dim
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5
    if args.data == 'elastic2':
        data_path_name = 'elastic'
        scale_factor = [0.0, 758.23]
        branch_dim=4; control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim 
        point_feature_dim = gnn_trunk_dim
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5
    if args.data == 'jet':
        scale_factor = [-1551.0989, 2104.0876]
        # scale_factor = [22.895, 85.987]
        control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = control_pc_branch_dim
        gnn_trunk_dim = trunk_dim 
        point_feature_dim = None
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5
        vt_patch_shape = (10, 10, 10)
    if args.data == 'jet2':
        scale_factor = [22.895, 85.987]
        control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = control_pc_branch_dim
        gnn_trunk_dim = trunk_dim 
        point_feature_dim = None
        aabb_max, aabb_min = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        scale=100; scale2=5
        vt_patch_shape = (10, 10, 10)
    if args.data == 'driver':
        scale_factor = [-20000, 10000]
        branch_dim=31; control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = branch_dim
        gnn_trunk_dim = trunk_dim + branch_dim
        point_feature_dim = None
        aabb_max, aabb_min = [4.1, 1.1, 1.8], [-1.2, -1.1, 0.0]
        scale=25; scale2=1
        vt_patch_shape = (10, 10, 10)
    if args.data == 'driver_plus':
        data_path_name = 'driver_plus'
        scale_factor = [-20000, 10000]
        control_pc_branch_dim=3; trunk_dim = 3; out_dim=1
        gano_branch_dim = control_pc_branch_dim
        gnn_trunk_dim = trunk_dim 
        point_feature_dim = None
        aabb_max, aabb_min = [4.1, 1.1, 1.8], [-1.2, -1.1, 0.0]
        scale=25; scale2=1
        vt_patch_shape = (10, 10, 10)
    
    try: 
        data_loc = '/work/hdd/bdsy/wzhong/processed_data/{}/'.format(data_path_name)
        print('Creating data loaders...')
        train_loader, val_loader, test_loader = create_data_loaders(data_loc)
    except: 
        print('Not finding in path 1')
    try: 
        data_loc = '/work/nvme/bdsy/wzhong/processed_data/{}/'.format(data_path_name)
        print('Creating data loaders...')
        train_loader, val_loader, test_loader = create_data_loaders(data_loc)
    except: 
        print('Not finding in path 2')

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and train the model
HIDDEN_DIM = 128
if args.model == 'DON':
    model = DON(branch_dim=branch_dim, trunk_dim=trunk_dim, hidden_dim=HIDDEN_DIM, out_dim=out_dim).float().to(device)
if args.model == 'SDON':
    model = S_DeepONet(branch_dim=branch_dim, trunk_dim=trunk_dim, hidden_dim=HIDDEN_DIM, out_dim=out_dim).float().to(device)
if args.model == 'DCON':
    model = DCON(branch_dim=branch_dim, trunk_dim=trunk_dim, hidden_dim=HIDDEN_DIM, out_dim=out_dim).float().to(device)
if args.model == 'GANO':
    model = GANO(branch_dim=gano_branch_dim, trunk_dim=trunk_dim, hidden_dim=HIDDEN_DIM, out_dim=out_dim).float().to(device)
if args.model == 'geomDON':
    model = Geom_DeepONet(branch_dim=gano_branch_dim, trunk_dim=trunk_dim, hidden_dim=HIDDEN_DIM, out_dim=out_dim).float().to(device)

if args.model == 'GNO':
    if args.data == 'driver' or args.data == 'driver_plus':
        model = GNO(width_node=16, width_kernel=32, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)
    else:
        model = GNO(width_node=32, width_kernel=64, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)
if args.model == 'EAGNO':
    if args.data == 'driver' or args.data == 'driver_plus':
        model = EAGNO(width_node=16, width_kernel=32, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)
    else:
        model = EAGNO(width_node=32, width_kernel=64, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)
if args.model == 'GUnet':
    if args.data == 'driver' or args.data == 'driver_plus':
        model = G_Unet(width_node=16, width_kernel=32, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)
    else:
        model = G_Unet(width_node=16, width_kernel=32, ker_in=2*trunk_dim, in_channels=gnn_trunk_dim, out_channels=out_dim).float().to(device)

if args.model == 'figconv':
    if args.data == 'driver':
        hd = 16
    elif args.data == 'driver_plus':
        hd = 16
    else:
        hd = 16
    model = get_figconv_model(scale=scale, scale2=scale2, 
        aabb_max=aabb_max, aabb_min=aabb_min, 
        point_feature_dim=point_feature_dim, hidden_dim=hd).float().to(device)
if args.model == 'gifno':
    if args.data == 'driver':
        hd = 4
    elif args.data == 'driver_plus':
        hd = 8
    else:
        hd = 16
    model = get_figconv_fno_model(scale=scale, 
        aabb_max=aabb_max, aabb_min=aabb_min,
        point_feature_dim=point_feature_dim, hidden_dim=hd).float().to(device)
if args.model == 'vt':
    if args.data == 'driver':
        hd = 4
    elif args.data == 'driver_plus':
        hd = 8
    else:
        hd = 16
    model = get_figconv_vt_model(scale=scale, 
        aabb_max=aabb_max, aabb_min=aabb_min,
        point_feature_dim=point_feature_dim, hidden_dim=hd, 
        patch_shape=vt_patch_shape).float().to(device)

if args.model == 'transformer':
    model = Transformer(in_channels=gnn_trunk_dim, hidden_channels=HIDDEN_DIM, out_channels=out_dim).float().to(device)
if args.model == 'pointnet':
    model = PointNet(in_channels=gnn_trunk_dim, hidden_channels=HIDDEN_DIM, out_channels=out_dim).float().to(device)
if args.model == 'transolver':
    model = Transolver_network(space_dim=3, fun_dim=0, n_hidden=256, n_head=4, dim_head=64, slice_num=64).float().to(device)
if args.model == 'transolver_local':
    model = Transolver_local_network(space_dim=3, fun_dim=0, n_hidden=64, n_head=4, dim_head=64, slice_num=128).float().to(device)
# distribute the training
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load the pre-trained model
print('Loading pretrained model ...')
try:
    checkpoint = torch.load('./res/trained_model/{}/model_{}_{}.pt'.format(args.model, args.model, args.data), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
except:
    print('No trained model')
if args.phase == 'train':
    print('model training ...')
    st = time.time()
    if (args.model == 'figconv' or args.model == 'gifno' or args.model == 'vt') and (args.data == 'driver' or args.data == 'driver_plus'):
        train_grid_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=400)
    else:
        train_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=400)
    et = time.time()
    print('total training time:', et-st, 'seconds')
    test_model(args, model, device, test_loader, scale_factor)
if args.phase == 'test':
    test_model(args, model, device, test_loader, scale_factor)
if args.phase == 'plot':
    if args.data == 'jet':
        plot_model_jet(args, model, device, test_loader)
    if args.data == 'driver' or args.data == 'driver_plus':
        plot_model_driver(args, model, device, test_loader)





