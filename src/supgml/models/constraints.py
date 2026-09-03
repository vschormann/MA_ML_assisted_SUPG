"""Composable constraints for cellwise stabilization parameters."""

import torch


class NonNegativeOutput(torch.nn.Module):
    """Map model output to non-negative values."""

    def __init__(self, model, method="softplus"):
        super().__init__()
        self.model = model
        self.method = method

    def forward(self, data):
        values = self.model(data)
        if self.method == "abs":
            return torch.abs(values)
        if self.method == "softplus":
            return torch.nn.functional.softplus(values)
        raise ValueError("method must be 'abs' or 'softplus'")


class BoundedOutput(torch.nn.Module):
    """Map output into per-node lower and upper bounds."""

    def __init__(self, model, lower=0.0, upper="upper", method="sigmoid"):
        super().__init__()
        self.model = model
        self.lower = lower
        self.upper = upper
        self.method = method

    @staticmethod
    def _resolve(value, data, reference):
        if isinstance(value, str):
            return getattr(data, value)
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    def forward(self, data):
        values = self.model(data)
        lower = self._resolve(self.lower, data, values)
        upper = self._resolve(self.upper, data, values)
        if self.method == "sigmoid":
            return lower + (upper - lower) * torch.sigmoid(values)
        if self.method == "clamp":
            return torch.maximum(torch.minimum(values, upper), lower)
        raise ValueError("method must be 'sigmoid' or 'clamp'")
