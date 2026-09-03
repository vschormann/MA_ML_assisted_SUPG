"""Evaluate predicted SUPG parameters against FEM and supervised objectives."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelResult:
    name: str
    fem_loss: float
    supervised_mse: float
    minimum_solution: float
    maximum_solution: float
    weights: np.ndarray
    solution: np.ndarray


def evaluate_models(solver, graph, models):
    """Evaluate named models and return structured, plot-independent results."""

    import torch

    results = []
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            prediction = model(graph).reshape(-1)
        weights = prediction.detach().cpu().numpy()
        solver.set_weights(weights)
        target = graph.y.reshape(-1).to(prediction.device)
        results.append(
            ModelResult(
                name=name,
                fem_loss=float(solver.loss()),
                supervised_mse=float(torch.mean((target - prediction) ** 2)),
                minimum_solution=float(np.min(solver.uh.x.array)),
                maximum_solution=float(np.max(solver.uh.x.array)),
                weights=weights.copy(),
                solution=solver.uh.x.array.copy(),
            )
        )
    return results
