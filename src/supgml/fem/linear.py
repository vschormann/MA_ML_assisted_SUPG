"""Reusable finite-element linear solves."""

import scipy.sparse.linalg
from dolfinx import fem, la


class LinearSolver:
    """Assemble DOLFINx forms and solve them through SciPy sparse LU.

    This preserves the serial thesis implementation. Distributed workloads
    should use a PETSc-native solver.
    """

    def __init__(self, a, L, uh, bcs):
        self.a_compiled = fem.form(a)
        self.L_compiled = fem.form(L)
        self.A = fem.create_matrix(self.a_compiled)
        self.b = fem.Function(uh.function_space)
        self.bcs = bcs
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

    def solve(self):
        self._A_scipy.data[:] = 0
        fem.assemble_matrix(self.A, self.a_compiled, bcs=self.bcs)

        self.b.x.array[:] = 0
        fem.assemble_vector(self.b.x.array, self.L_compiled)
        fem.apply_lifting(self.b.x.array, [self.a_compiled], [self.bcs])
        self.b.x.scatter_reverse(la.InsertMode.add)
        for boundary_condition in self.bcs:
            boundary_condition.set(self.b.x.array)

        factorization = scipy.sparse.linalg.splu(self._A_scipy)
        self.uh.x.array[:] = factorization.solve(self.b.x.array)
        return self.uh
