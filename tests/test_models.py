import torch
from torch_geometric.data import Data

from supgml.models import BoundedOutput, create_model


def test_configurable_gnn_and_bounds():
    graph = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        upper=torch.ones(3, 1),
    )
    base = create_model("gcn", in_channels=4, hidden_channels=8, num_layers=3)
    model = BoundedOutput(base, upper="upper")
    result = model(graph)
    assert result.shape == (3, 1)
    assert torch.all(result >= 0)
    assert torch.all(result <= graph.upper)
