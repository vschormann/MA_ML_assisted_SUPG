"""Algebraic flux-correction limiters extracted from the research notebook."""

import numpy as np
import scipy.optimize


class ConvergenceMonitor:
    """Record residual norms reported by a SciPy root solve."""

    def __init__(self, residual):
        self.residual = residual
        self.F = residual
        self.iteration = 0
        self.residuals = []

    def __call__(self, values, *args):
        norm = np.linalg.norm(self.residual(values))
        self.residuals.append(norm)
        self.iteration += 1


class _BaseAFC:
    def optimize(self, initial, ftol=1e-12, maxiter=1000, method="krylov"):
        monitor = ConvergenceMonitor(self)
        result = scipy.optimize.root(
            self,
            initial,
            method=method,
            callback=monitor,
            options={"maxiter": maxiter, "fatol": ftol},
        )
        return result.x, monitor

    @staticmethod
    def _diffusion_matrix(matrix, rows, columns):
        diffusion = matrix.copy()
        diffusion.data[:] = 0
        pairs = {(min(i, j), max(i, j)) for i, j in zip(rows, columns) if i != j}
        for i, j in pairs:
            value = -float(max(0.0, matrix[i, j], matrix[j, i]))
            if value < 0:
                diffusion[i, j] += value
                diffusion[j, i] += value
        diffusion.setdiag(0)
        diffusion.setdiag(-diffusion.sum(axis=1))
        return diffusion.tocsr()


class KuzminAFC(_BaseAFC):
    """Kuzmin algebraic flux-correction residual."""

    def __init__(self, matrix, rhs, boundary_dofs):
        self.A = matrix.tocsr()
        self.f = np.asarray(rhs)
        self.dbc_dofs = np.asarray(boundary_dofs, dtype=int)
        coordinates = self.A.tocoo()
        self.col, self.row = coordinates.col, coordinates.row
        self.D = self._diffusion_matrix(self.A, self.row, self.col)
        self.U = self._upwind_matrix()

    def _upwind_matrix(self):
        upwind = self.A.copy()
        upwind.data[:] = 0
        pairs = {(min(i, j), max(i, j)) for i, j in zip(self.row, self.col) if i != j}
        for i, j in pairs:
            if self.A[i, j] >= self.A[j, i]:
                upwind[i, j] = 1
            else:
                upwind[j, i] = 1
        return upwind.tocsr()

    def __call__(self, values):
        flux = self.D.copy()
        flux.data = self.D.data * (values[self.col] - values[self.row])
        positive = flux.copy()
        positive.data = np.maximum(flux.data, 0)
        negative = flux.copy()
        negative.data = np.minimum(flux.data, 0)
        p_plus = np.asarray(self.U.multiply(positive).sum(axis=1)).ravel()
        p_minus = np.asarray(self.U.multiply(negative).sum(axis=1)).ravel()
        q_plus = -np.asarray(negative.sum(axis=1)).ravel()
        q_minus = -np.asarray(positive.sum(axis=1)).ravel()
        r_plus = np.minimum(np.divide(q_plus, p_plus, out=np.ones_like(q_plus), where=p_plus != 0), 1)
        r_minus = np.minimum(np.divide(q_minus, p_minus, out=np.ones_like(q_minus), where=p_minus != 0), 1)
        r_plus[self.dbc_dofs] = r_minus[self.dbc_dofs] = 1
        alpha = np.ones_like(flux.data)
        alpha[flux.data > 0] = r_plus[self.row][flux.data > 0]
        alpha[flux.data < 0] = r_minus[self.row][flux.data < 0]
        alpha *= self.U.data
        correction = self.D.copy()
        correction.data = (1 - alpha) * self.D.data
        correction.setdiag(0)
        correction.setdiag(-correction.sum(axis=1))
        return (self.A + correction).dot(values) - self.f


class BJKAFC(_BaseAFC):
    """BJK algebraic flux-correction residual."""

    def __init__(self, matrix, rhs, boundary_dofs, gamma=1.0):
        self.A = matrix.tocsr()
        if np.min(np.diff(self.A.indptr)) <= 0:
            raise ValueError("system matrix cannot contain empty rows")
        self.f = np.asarray(rhs)
        self.gamma = gamma
        self.dbc_dofs = np.asarray(boundary_dofs, dtype=int)
        coordinates = self.A.tocoo()
        self.col, self.row = coordinates.col, coordinates.row
        self.D = self._diffusion_matrix(self.A, self.row, self.col)

    def __call__(self, values):
        neighbor_values = values[self.col]
        local_max = np.maximum.reduceat(neighbor_values, self.A.indptr[:-1])
        local_min = np.minimum.reduceat(neighbor_values, self.A.indptr[:-1])
        flux = self.D.copy()
        flux.data = self.D.data * (neighbor_values - values[self.row])
        positive = flux.copy()
        positive.data = np.maximum(flux.data, 0)
        negative = flux.copy()
        negative.data = np.minimum(flux.data, 0)
        p_plus = np.asarray(positive.sum(axis=1)).ravel()
        p_minus = np.asarray(negative.sum(axis=1)).ravel()
        q_plus = -self.gamma * self.D.diagonal() * (values - local_max)
        q_minus = -self.gamma * self.D.diagonal() * (values - local_min)
        r_plus = np.minimum(np.divide(q_plus, p_plus, out=np.ones_like(q_plus), where=p_plus != 0), 1)
        r_minus = np.minimum(np.divide(q_minus, p_minus, out=np.ones_like(q_minus), where=p_minus != 0), 1)
        r_plus[self.dbc_dofs] = r_minus[self.dbc_dofs] = 1
        limiter = np.ones_like(flux.data)
        limiter[flux.data > 0] = r_plus[self.row][flux.data > 0]
        limiter[flux.data < 0] = r_minus[self.row][flux.data < 0]
        # Symmetric pairwise minimum, expressed through a sparse mirror lookup.
        mirror = self.A.copy()
        mirror.data = limiter
        limiter = np.minimum(limiter, mirror.T[self.row, self.col].A1)
        correction = self.D.copy()
        correction.data = (1 - limiter) * self.D.data
        correction.setdiag(0)
        correction.setdiag(-correction.sum(axis=1))
        return (self.A + correction).dot(values) - self.f


F_AFC_Kuzmin = KuzminAFC
F_AFC_BJK = BJKAFC
