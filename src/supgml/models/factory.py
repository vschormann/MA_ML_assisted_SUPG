"""Small configurable regressors replacing copied notebook architectures."""

import torch
import torch_geometric as tg


_CONVOLUTIONS = {
    "gcn": tg.nn.GCNConv,
    "sage": tg.nn.SAGEConv,
    "gat": tg.nn.GATConv,
    "gatv2": tg.nn.GATv2Conv,
}


class GraphRegressor(torch.nn.Module):
    """Nodewise MLP or message-passing regression model."""

    def __init__(
        self,
        architecture,
        in_channels,
        hidden_channels=32,
        out_channels=1,
        num_layers=4,
        edge_dim=None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.architecture = architecture.lower()
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        if self.architecture == "mlp":
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(source, target)
                for source, target in zip(channels, channels[1:])
            )
        else:
            try:
                convolution = _CONVOLUTIONS[self.architecture]
            except KeyError as error:
                raise ValueError("unknown architecture: {}".format(architecture)) from error
            self.layers = torch.nn.ModuleList()
            for source, target in zip(channels, channels[1:]):
                options = {}
                if self.architecture in {"gat", "gatv2"} and edge_dim is not None:
                    options["edge_dim"] = edge_dim
                    options["add_self_loops"] = False
                self.layers.append(convolution(source, target, **options))
        self.edge_dim = edge_dim

    def forward(self, data):
        hidden = data.x
        for index, layer in enumerate(self.layers):
            if self.architecture == "mlp":
                hidden = layer(hidden)
            else:
                options = {"x": hidden, "edge_index": data.edge_index}
                if self.edge_dim is not None and self.architecture in {"gat", "gatv2"}:
                    options["edge_attr"] = data.edge_attr
                hidden = layer(**options)
            if index + 1 < len(self.layers):
                hidden = torch.relu(hidden)
        return hidden


def create_model(architecture, **parameters):
    """Create a configurable nodewise regression model."""

    return GraphRegressor(architecture=architecture, **parameters)
