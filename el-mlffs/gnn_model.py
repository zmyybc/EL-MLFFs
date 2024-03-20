import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, JumpingKnowledge


class GNNModel(torch.nn.Module):
    def __init__(self, node_feature_dim, output_dim, n_atoms):
        super(GNNModel, self).__init__()
        self.heads = 4  
        self.conv1 = GATConv(node_feature_dim, 32, heads=self.heads, concat=True)
        self.conv2 = GATConv(32 * self.heads, 64, heads=self.heads, concat=True)
        self.conv3 = GATConv(64 * self.heads, 128, heads=self.heads, concat=True)
        self.conv4 = GATConv(128 * self.heads, 128, heads=self.heads, concat=True)
        self.conv5 = GATConv(128 * self.heads, 128, heads=self.heads, concat=True)
        self.conv6 = GATConv(128 * self.heads, 256, heads=self.heads, concat=False)
        self.conv7 = GATConv(256, 256, heads=self.heads, concat=False)
        self.conv8 = GATConv(256, 256, heads=self.heads, concat=False)

        self.jk = JumpingKnowledge(mode='cat')
        self.fc1 = torch.nn.Linear((32 * self.heads + 64 * self.heads + 128 * self.heads + 128 * self.heads + 128 * self.heads + 256 + 256 + 256), output_dim * n_atoms)

        self.res1 = torch.nn.Linear(node_feature_dim, 32 * self.heads)
        self.res2 = torch.nn.Linear(32 * self.heads, 64 * self.heads)
        self.res3 = torch.nn.Linear(64 * self.heads, 128 * self.heads)
        self.res4 = torch.nn.Identity()  
        self.res5 = torch.nn.Identity()  
        self.res6 = torch.nn.Linear(128 * self.heads, 256)

        self.n_atoms = n_atoms
        self.output_dim = output_dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index)) + self.res1(x)
        x2 = F.relu(self.conv2(x1, edge_index)) + self.res2(x1)
        x3 = F.relu(self.conv3(x2, edge_index)) + self.res3(x2)
        x4 = F.relu(self.conv4(x3, edge_index)) + self.res4(x3)
        x5 = F.relu(self.conv5(x4, edge_index)) + self.res5(x4)
        x6 = F.relu(self.conv6(x5, edge_index)) + self.res6(x5)
        x7 = F.relu(self.conv7(x6, edge_index))  
        x8 = F.relu(self.conv8(x7, edge_index))  
        x_jk = self.jk([x1, x2, x3, x4, x5, x6, x7, x8])

        x_pool = global_mean_pool(x_jk, batch)

        x_out = self.fc1(x_pool)
        x_out = x_out.view(-1, self.n_atoms, self.output_dim)
        return x_out

class GNNModel(torch.nn.Module):
    def __init__(self, node_feature_dim, output_dim, n_atoms, num_heads=4, conv_channels=[32, 64, 128, 128, 128, 256, 256, 256], concat=True, mode='cat', use_residual=True):
        super(GNNModel, self).__init__()
        self.heads = num_heads
        self.concat = concat
        self.use_residual = use_residual
        self.conv_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        
        in_channels = node_feature_dim
        for out_channels in conv_channels:
            if out_channels != conv_channels[-1] or concat: 
                self.conv_layers.append(GATConv(in_channels, out_channels // num_heads, heads=num_heads, concat=concat))
            else:
                self.conv_layers.append(GATConv(in_channels, out_channels, heads=num_heads, concat=concat))
            if use_residual:
                self.res_layers.append(torch.nn.Linear(in_channels, out_channels if concat else out_channels * num_heads) if in_channels != out_channels else torch.nn.Identity())
            else:
                self.res_layers.append(torch.nn.Identity())  
            in_channels = out_channels if concat else out_channels * num_heads

        self.jk = JumpingKnowledge(mode=mode)

        jk_dim = sum(conv_channels) if mode == 'cat' else conv_channels[-1]
        self.fc1 = torch.nn.Linear(jk_dim, output_dim * n_atoms)

        self.n_atoms = n_atoms
        self.output_dim = output_dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []

        for conv, res in zip(self.conv_layers, self.res_layers):
            x_res = res(x)
            x = F.relu(conv(x, edge_index))
            x = x + x_res if self.use_residual else x  
            xs.append(x)

        x_jk = self.jk(xs)
        x_pool = global_mean_pool(x_jk, batch)
        x_out = self.fc1(x_pool)
        x_out = x_out.view(-1, self.n_atoms, self.output_dim)

        return x_out

