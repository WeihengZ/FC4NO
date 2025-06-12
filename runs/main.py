import argparse
from setup import *

# arguments
parser = argparse.ArgumentParser(description='command setting')
# model architecture
parser.add_argument('--act', type=str, default='gelu', choices=['gelu', 'tanh', 'relu'])
parser.add_argument('--last_act', type=str, default='identity', choices=['identity', 'sigmoid'])
parser.add_argument('--num_layers', type=int, default=4, choices=[3,4,5,6])
parser.add_argument('--hidden_dim', type=int, default=32, choices=[4,16,64,256,512])
# IO dimension
parser.add_argument('--coor_dim', type=int, default=3)
parser.add_argument('--geo_param_dim', type=int, default=3)
parser.add_argument('--load_param_dim', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=1)
# Grid model
parser.add_argument('--aabb_min', type=list, default=[0.0,0.0,0.0])
parser.add_argument('--aabb_max', type=list, default=[1.0,1.0,1.0])
parser.add_argument('--grid_scale', type=int, default=50)
parser.add_argument('--grid_scale_low', type=int, default=5)
parser.add_argument('--patch_shape', type=list, default=[5,5,5])
parser.add_argument('--factorized_grid_scale', type=int, default=5)
# Point model
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--slice_num', type=int, default=64)
# Improved model setting
parser.add_argument('--fun_dim', type=int, default=0)

# model-specific configuration
parser.add_argument('--model_configure', type=str, default='default', choices=['default', 'branch_trunk', 'param_concat'])
parser.add_argument('--val_freq', type=int, default=5)
parser.add_argument('--epochs', type=int, default=8)
# model training setting
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'plot'])
parser.add_argument('--model', type=str, default='GNO', 
    choices=['DeepONet', 'GANO', 'SDON', 'DCON', 'geomDON', 
    'GNO', 'EAGNO', 'GUNet', 
    'figconv', 'gifno', 'vt',
    'pointnet', 'gnot', 'transolver'])
parser.add_argument('--data', type=str, default='bracket', choices=['heatsink', 'bracket_time', 'bracket', 'jet', 'driver', 'driver_plus'])
args = parser.parse_args()

# extract the configuration
args = set_arguments(args)

# load the data
data_loc = data_loc_dict[args.data]
train_loader, val_loader, test_loader = create_data_loaders(data_loc)
try:
    scale_factor = np.load(data_loc + "min_max.npz")
except:
    scale_factor = np.load(data_loc + "mean_std.npz")

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
if args.model_configure == 'default':
    model = Vanilla_Model(args).float().to(device)
elif args.model_configure == 'param_concat':
    model = Param_concat_geometric_Model(args).float().to(device)
elif args.model_configure == 'branch_trunk':
    model = Param_branch_trunk_geometric_Model(args).float().to(device)

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

# exp
if args.phase == 'train':
    print('model training ...')
    st = time.time()
    train_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=args.epochs)
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





