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

    def save(self, number, graph, mesh, split="train", variant="standard", overwrite=False):
        """Persist a graph and its mesh using the repository directory layout."""

        import torch
        from dolfinx.io import XDMFFile

        split_name = {"train": "training_set", "test": "test_set"}.get(split, split)
        suffix = "" if variant == "standard" else "_{}".format(variant)
        graph_path = self.root / (split_name + suffix) / "input_values" / "raw"
        mesh_path = self.root / split_name / "mesh_files"
        graph_file = graph_path / "G_{}.pt".format(number)
        mesh_file = mesh_path / "mesh_{}.xdmf".format(number)
        if not overwrite and (graph_file.exists() or mesh_file.exists()):
            raise FileExistsError("case {} already exists".format(number))
        graph_path.mkdir(parents=True, exist_ok=True)
        mesh_path.mkdir(parents=True, exist_ok=True)
        graph.mesh_id = int(number)
        torch.save(graph, graph_file)
        with XDMFFile(mesh.comm, str(mesh_file), "w") as xdmf:
            xdmf.write_mesh(mesh)
        return graph_file, mesh_file


def Data_to_solver(num, train=True, edge_attr=False, globalizer=False, v2=False):
    """Compatibility wrapper around :class:`CaseRepository`."""

    enabled = [name for name, value in (("edge_attr", edge_attr), ("globalizer", globalizer), ("v2", v2)) if value]
    if len(enabled) > 1:
        raise ValueError("select at most one dataset variant")
    variant = enabled[0] if enabled else "standard"
    return CaseRepository().load(num, split="train" if train else "test", variant=variant)
