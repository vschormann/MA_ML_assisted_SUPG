"""Reproduce the focused Chapter 5 MLP/GATv2 experiment."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from supgml.autograd import FEMObjective
from supgml.models import RevisedGATv2, RevisedMLP, combined_supervised_loss


def _build_fem_objective(graph, mesh_path):
    """Construct the SPDE 3 AFC-BJK comparison objective from Chapter 5."""

    import numpy as np
    import ufl
    from dolfinx import fem, mesh as msh
    from dolfinx.io import XDMFFile
    from mpi4py import MPI

    from supgml.fem import interpolate_expr
    from supgml.supg import AdjointSUPGSolver

    with XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
    space = fem.functionspace(mesh, ("CG", 1))
    solution = fem.Function(space)
    mesh.topology.create_connectivity(1, 2)
    facets = msh.exterior_facet_indices(mesh.topology)
    boundary_dofs = fem.locate_dofs_topological(space, 1, facets)
    x = ufl.SpatialCoordinate(mesh)
    velocity = ufl.as_vector(
        (
            fem.Constant(mesh, np.cos(-np.pi / 3.0)),
            fem.Constant(mesh, np.sin(-np.pi / 3.0)),
        )
    )
    boundary_expression = ufl.conditional(
        ufl.ge(x[1], 0.7 + ufl.sin(-ufl.pi / 3) / ufl.cos(-ufl.pi / 3) * x[0]),
        1,
        0,
    ) * ufl.conditional(ufl.Or(ufl.eq(x[1], 0), ufl.eq(x[0], 1)), 0, 1)
    boundary_values = interpolate_expr(boundary_expression, space)
    boundary_conditions = [fem.dirichletbc(boundary_values, boundary_dofs)]
    problem = (
        mesh,
        space,
        solution,
        fem.Constant(mesh, 1e-8),
        velocity,
        None,
        fem.Constant(mesh, 0.0),
        None,
        boundary_conditions,
    )
    target_solution = fem.Function(space)
    target_solution.x.array[:] = graph.u_bjk.detach().cpu().numpy()
    objective = (solution - target_solution) ** 2 * ufl.dx
    return FEMObjective(AdjointSUPGSolver(problem, objective))


def _checkpoint(path, model, optimizer, supervised_loss, fem_loss, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "supervised_loss": supervised_loss,
            "fem_loss": fem_loss,
            "experiment": config,
        },
        path,
    )


def run_revised_training(config):
    """Train revised models and select checkpoints by the AFC-BJK FEM loss."""

    device = torch.device(config.get("device", "cpu"))
    graph = torch.load(config["dataset"], map_location=device, weights_only=False)
    feature_name = config.get("features", "x0")
    features = getattr(graph, feature_name)
    mask = getattr(graph, config.get("node_mask", "interior_dofs")).long()
    target = getattr(graph, config.get("target", "y_optimized_z")).reshape(-1, 1)
    fem_objective = _build_fem_objective(graph, config["mesh"])
    output_dir = Path(config["output_dir"])
    selection_every = config.get("selection_every", 1000)
    huber_weight = config.get("loss", {}).get("huber_weight", 0.7)
    results = {}

    for architecture in config["architectures"]:
        if architecture == "revised-mlp":
            model = RevisedMLP(features.shape[1]).to(device)
            loader = DataLoader(
                TensorDataset(features[mask], target[mask]),
                batch_size=config["batch_size"],
                shuffle=True,
            )
        elif architecture == "revised-gatv2":
            model = RevisedGATv2(features.shape[1]).to(device)
            loader = None
        else:
            raise ValueError("unknown revised architecture: {}".format(architecture))

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5),
        )
        schedule = config.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=schedule.get("t0", 50),
            T_mult=schedule.get("tmult", 2),
        )
        best_fem = float("inf")
        samples = []

        for epoch in range(config["epochs"]):
            model.train()
            if loader is not None:
                supervised = 0.0
                for batch_features, batch_target in loader:
                    optimizer.zero_grad()
                    loss = combined_supervised_loss(
                        model(batch_features), batch_target, huber_weight
                    )
                    loss.backward()
                    optimizer.step()
                    supervised += loss.item()
                supervised /= len(loader)
            else:
                optimizer.zero_grad()
                loss = combined_supervised_loss(
                    model(graph, feature_name, mask), target[mask], huber_weight
                )
                loss.backward()
                optimizer.step()
                supervised = loss.item()
            scheduler.step(epoch + 1)

            if epoch % selection_every == 0 or epoch + 1 == config["epochs"]:
                model.eval()
                with torch.no_grad():
                    if loader is None:
                        prediction = model(graph, feature_name, mask)[0]
                    else:
                        prediction = model(features[mask])[0]
                    complete = target.detach().clone()
                    complete[mask] = prediction
                fem_loss = float(fem_objective(complete).detach().cpu())
                samples.append(
                    {"epoch": epoch, "supervised_loss": supervised, "fem_loss": fem_loss}
                )
                if fem_loss < best_fem:
                    best_fem = fem_loss
                    _checkpoint(
                        output_dir / "{}.pth".format(architecture),
                        model,
                        optimizer,
                        supervised,
                        fem_loss,
                        config,
                    )
        results[architecture] = {"best_fem_loss": best_fem, "samples": samples}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results
