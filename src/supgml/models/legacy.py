import torch
import torch_geometric as tg
from torch_geometric import utils

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
    


class MIX(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = tg.nn.conv.GATv2Conv(10, 4)
        self.conv2 = tg.nn.conv.GATv2Conv(4, 4)
        self.conv3 = tg.nn.conv.GATv2Conv(4, 4)
        self.act1 = tg.nn.models.MLP(in_channels=4, hidden_channels=4,out_channels=4, num_layers=3)
        self.act2 = tg.nn.models.MLP(in_channels=4, hidden_channels=4,out_channels=4, num_layers=3)
        self.act3 = tg.nn.models.MLP(in_channels=4, hidden_channels=4,out_channels=4, num_layers=3)
        self.mlp = tg.nn.models.MLP(in_channels=4, hidden_channels=4,out_channels=1, num_layers=16)
    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x=x, edge_index=edge_index)
        h = self.act1(x=h)
        h = self.conv2(x=h, edge_index=edge_index)
        h = self.act2(x=h)
        h = self.conv3(x=h, edge_index=edge_index)
        h = self.act3(x=h)
        h=self.mlp(x=h)
        h = torch.clamp(input=h, min=torch.zeros_like(data.y), max=100*torch.ones_like(data.y))

        return h
    
class AbsRestriction(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model()
    def forward(self, data) -> torch.Tensor:
        model_out = self.model(data)
        return torch.abs(model_out)
    

class ClampRestriction(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model()
    def forward(self, data) -> torch.Tensor:
        upper = data.upper
        model_out = self.model(data)
        return torch.clamp(input=model_out, min=torch.zeros_like(data.y), max=upper)



class SigmoidRestriction(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model()
    def forward(self, data) -> torch.Tensor:
        upper = data.upper
        model_out = self.model(data)
        return upper*model_out.sigmoid()
    

class PenaltyRestriction(torch.nn.Module):
    def __init__(self, model, penalty=100):
        super().__init__()
        self.model = model()
        self.penalty = penalty
    def forward(self, data) -> torch.Tensor:
        upper = data.upper
        model_out = self.model(data)
        return model_out + self.penalty * (model_out.clamp(max=0) + model_out-model_out.clamp(max=upper))

class DirOpt(torch.nn.Module):
    def __init__(self, Psl, dtype=torch.float32):
        super().__init__()
        self.nn = torch.nn.Identity()
        self.values = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(ps.fs.yh.x.array, dtype=dtype).reshape(-1, 1))
            for ps in Psl
        ])
    def forward(self, data) -> torch.Tensor:
        return torch.clamp(self.nn(self.values[0]), min=torch.zeros_like(data.y), max=data.upper)
    


class mha(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = tg.nn.models.GAT(
            in_channels=7,
            hidden_channels=5,
            num_layers=2,
            out_channels=4, 
            v2 = True,
            edge_dim=2,
            add_self_loops=False
        )
        self.gat2 = tg.nn.models.GAT(

            in_channels=4,
            hidden_channels=4,
            num_layers=2,
            out_channels=4, 
            v2 = True,
            add_self_loops=False
        )
        self.gat3 = tg.nn.models.GAT(
            in_channels=4,
            hidden_channels=4,
            num_layers=2,
            out_channels=4, 
            v2 = True,
            edge_dim=2,
            add_self_loops=False
        )
        self.pattn1 = tg.nn.attention.PerformerAttention(channels=4, heads=1)
        self.pattn2 = tg.nn.attention.PerformerAttention(channels=4, heads=2)
        self.mlp = tg.nn.models.MLP(
            in_channels=4,
            hidden_channels=16,
            num_layers=3,
            out_channels=1,
        )

    def forward(self, data) -> torch.Tensor:
        batch, x, edge_index, edge_attr  = data.batch, data.x, data.edge_index, data.edge_attr
        h = self.gat1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        ).relu()
        h, mask = utils.to_dense_batch(x=h, batch=batch)
        h = self.pattn1(h)[mask]
        h = self.gat2(
            x=h,
            edge_index=edge_index
        ).relu()
        h, mask = utils.to_dense_batch(x=h, batch=batch)
        h = self.pattn2(h)[mask]
        h = self.gat3(
            x=h,
            edge_index=edge_index,
            edge_attr=edge_attr
        ).relu()
        h=self.mlp(h)
        return torch.abs(h)
