import numpy as np
import torch

from supgml.autograd import FEMObjective


class QuadraticSolver:
    def set_weights(self, weights):
        self.weights = np.asarray(weights)

    def loss(self):
        return np.sum(self.weights**2)

    def grad(self):
        return 2 * self.weights


def test_fem_objective_uses_solver_gradient():
    weights = torch.tensor([[1.0], [-2.0]], requires_grad=True)
    loss = FEMObjective(QuadraticSolver())(weights)
    loss.backward()
    torch.testing.assert_close(weights.grad, 2 * weights.detach())
