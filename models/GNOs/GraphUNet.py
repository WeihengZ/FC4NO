from torch_geometric.nn.models import GraphUNet
import torch

class G_Unet(torch.nn.Module):
    def __init__(self, width_node=32, width_kernel=64, ker_in=4, in_channels=2, out_channels=1):
        super(G_Unet, self).__init__()

        self.graphunet = GraphUNet(in_channels=in_channels, hidden_channels=width_node, 
            out_channels=out_channels, depth=3, act='gelu')

    def forward(self, data):

        assert hasattr(data, 'edge_index') == True, 'GUNet need connectivity information'
        assert hasattr(data, 'edge_attr') == True, 'GUNet need connectivity information'

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # add the parameter features in if possible
        if hasattr(data, 'param') == True:
            x = torch.cat((x, data.param.unsqueeze(0).repeat(x.shape[0], 1)), -1)
        
        # GCN forward
        x = self.graphunet(x, edge_index)

        return torch.sigmoid(x)