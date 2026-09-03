"""Wider Chapter 5 models and their two-headed supervised loss."""

import torch
import torch_geometric as tg


class _TwoHeadedRegressor(torch.nn.Module):
    def __init__(self, hidden_channels=256):
        super().__init__()
        self.main_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.auxiliary_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def _heads(self, latent):
        return self.main_head(latent), self.auxiliary_head(latent)


class RevisedMLP(_TwoHeadedRegressor):
    """Wide local predictor used for the revised supervised experiment."""

    def __init__(self, in_channels):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
        )

    def forward(self, features):
        return self._heads(self.encoder(features))


class RevisedGATv2(_TwoHeadedRegressor):
    """Four-layer, width-256 GATv2 predictor used in Chapter 5."""

    def __init__(self, in_channels):
        super().__init__()
        self.encoder = tg.nn.models.GAT(
            in_channels=in_channels,
            hidden_channels=256,
            num_layers=4,
            out_channels=256,
            v2=True,
            add_self_loops=False,
        )

    def forward(self, data, feature_name="x0", node_mask=None):
        latent = self.encoder(getattr(data, feature_name), data.edge_index)
        main, auxiliary = self._heads(latent)
        if node_mask is not None:
            main, auxiliary = main[node_mask], auxiliary[node_mask]
        return main, auxiliary


def combined_supervised_loss(prediction, target, huber_weight=0.7):
    """Chapter 5's summed Huber and squared-error objective."""

    main, auxiliary = prediction
    huber = torch.nn.functional.huber_loss(main, target, reduction="sum")
    squared = torch.nn.functional.mse_loss(auxiliary, target, reduction="sum")
    return huber_weight * huber + (1 - huber_weight) * squared
