import torch
from Training_utils import train_loader, train
import torch_geometric as tg

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(10, 5)
        self.l2 = torch.nn.Linear(5, 5)
        self.l3 = torch.nn.Linear(5, 5)
        self.l4 = torch.nn.Linear(5, 5)
        self.l5 = torch.nn.Linear(5, 5)
        self.l6 = torch.nn.Linear(5, 5)
        self.l7 = torch.nn.Linear(5, 5)
        self.l8 = torch.nn.Linear(5, 5)
        self.l9 = torch.nn.Linear(5, 5)
        self.l10 = torch.nn.Linear(5, 1)

    def forward(self, data) -> torch.Tensor:
        x = data.x

        # Perform two-layers of message passing:
        h = self.l1(x)
        h = h.relu()
        h = self.l2(h)
        h = h.relu()
        h = self.l3(h)
        h = h.relu()
        h = self.l4(h)
        h = h.relu()
        h = self.l5(h)
        h = h.relu()
        h = self.l6(h)
        h = h.relu()
        h = self.l7(h)
        h = h.relu()
        h = self.l8(h)
        h = h.relu()
        h = self.l9(h)
        h = h.relu()
        h = self.l10(h)

        return h
    
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tg.nn.conv.GCNConv(10, 5)
        self.conv2 = tg.nn.conv.GCNConv(5, 5)
        self.conv3 = tg.nn.conv.GCNConv(5, 5)
        self.conv4 = tg.nn.conv.GCNConv(5, 5)
        self.conv5 = tg.nn.conv.GCNConv(5, 5)
        self.conv6 = tg.nn.conv.GCNConv(5, 5)
        self.conv7 = tg.nn.conv.GCNConv(5, 5)
        self.conv8 = tg.nn.conv.GCNConv(5, 5)
        self.conv9 = tg.nn.conv.GCNConv(5, 5)
        self.conv10 = tg.nn.conv.GCNConv(5, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        h = self.conv1(x=x, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv5(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv6(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv7(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv8(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv9(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv10(x=h, edge_index=edge_index)

        return h
    
    
class SAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = tg.nn.conv.SAGEConv(10, 5)
        self.conv2 = tg.nn.conv.SAGEConv(5, 5)
        self.conv3 = tg.nn.conv.SAGEConv(5, 5)
        self.conv4 = tg.nn.conv.SAGEConv(5, 5)
        self.conv5 = tg.nn.conv.SAGEConv(5, 5)
        self.conv6 = tg.nn.conv.SAGEConv(5, 5)
        self.conv7 = tg.nn.conv.SAGEConv(5, 5)
        self.conv8 = tg.nn.conv.SAGEConv(5, 5)
        self.conv9 = tg.nn.conv.SAGEConv(5, 5)
        self.conv10 = tg.nn.conv.SAGEConv(5, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        h = self.conv1(x=x, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv5(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv6(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv7(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv8(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv9(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv10(x=h, edge_index=edge_index)


        return h
    
class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = tg.nn.conv.GATConv(10, 5)
        self.conv2 = tg.nn.conv.GATConv(5, 5)
        self.conv3 = tg.nn.conv.GATConv(5, 5)
        self.conv4 = tg.nn.conv.GATConv(5, 5)
        self.conv5 = tg.nn.conv.GATConv(5, 5)
        self.conv6 = tg.nn.conv.GATConv(5, 5)
        self.conv7 = tg.nn.conv.GATConv(5, 5)
        self.conv8 = tg.nn.conv.GATConv(5, 5)
        self.conv9 = tg.nn.conv.GATConv(5, 5)
        self.conv10 = tg.nn.conv.GATConv(5, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Perform two-layers of message passing:
        h = self.conv1(x=x, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv5(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv6(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv7(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv8(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv9(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv10(x=h, edge_index=edge_index)

        return h
    
class GATv2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = tg.nn.conv.GATv2Conv(10, 5)
        self.conv2 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv3 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv4 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv5 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv6 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv7 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv8 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv9 = tg.nn.conv.GATv2Conv(5, 5)
        self.conv10 = tg.nn.conv.GATv2Conv(5, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x=x, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv3(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv4(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv5(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv6(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv7(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv8(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv9(x=h, edge_index=edge_index)
        h = h.relu()
        h = self.conv10(x=h, edge_index=edge_index)

        return h