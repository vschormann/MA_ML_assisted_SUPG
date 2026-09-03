import torch
from torch_geometric.data import Data

from supgml.models import RevisedGATv2, RevisedMLP, combined_supervised_loss


def test_revised_mlp_has_two_prediction_heads():
    model = RevisedMLP(in_channels=9)
    main, auxiliary = model(torch.randn(5, 9))
    assert main.shape == auxiliary.shape == (5, 1)
    assert combined_supervised_loss((main, auxiliary), torch.zeros(5, 1)).ndim == 0


def test_revised_gat_respects_node_mask():
    graph = Data(
        x0=torch.randn(4, 9),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
    )
    model = RevisedGATv2(in_channels=9)
    main, auxiliary = model(graph, node_mask=torch.tensor([1, 3]))
    assert main.shape == auxiliary.shape == (2, 1)
