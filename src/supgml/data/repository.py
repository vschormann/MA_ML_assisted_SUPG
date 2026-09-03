"""Explicit filesystem access for graph and mesh cases."""

from pathlib import Path


class CaseRepository:
    """Load paired PyG graph and XDMF mesh files from a data directory."""

    def __init__(self, root="data"):
        self.root = Path(root)

    def load(self, number, split="train", variant="standard", comm=None):
        """Return ``(solver, graph)`` for one stored case.

        Heavy FEM and ML imports occur here rather than when the module is
        imported, keeping the base package lightweight.
        """

        import torch
        from dolfinx.io import XDMFFile
        from mpi4py import MPI

        from supgml.benchmarks import create

        split_name = {"train": "training_set", "test": "test_set"}.get(split, split)
        suffix = "" if variant == "standard" else "_{}".format(variant)
        graph_path = self.root / (split_name + suffix) / "input_values" / "raw"
        graph = torch.load(graph_path / "G_{}.pt".format(number), weights_only=False)

        mesh_path = self.root / split_name / "mesh_files" / "mesh_{}.xdmf".format(
            int(graph.mesh_id)
        )
        with XDMFFile(comm or MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
        return create(int(graph.prblm_id), mesh=mesh), graph


def Data_to_solver(num, train=True, edge_attr=False, globalizer=False, v2=False):
    """Compatibility wrapper around :class:`CaseRepository`."""

    enabled = [name for name, value in (("edge_attr", edge_attr), ("globalizer", globalizer), ("v2", v2)) if value]
    if len(enabled) > 1:
        raise ValueError("select at most one dataset variant")
    variant = enabled[0] if enabled else "standard"
    return CaseRepository().load(num, split="train" if train else "test", variant=variant)
