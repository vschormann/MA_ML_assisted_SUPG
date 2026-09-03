"""Run the Chapter 4 architecture matrix without notebook duplication."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from supgml.autograd import BatchedFEMLoss
from supgml.data import CaseRepository, GraphDataset
from supgml.models import BoundedOutput, NonNegativeOutput, create_model
from supgml.training import self_supervised_train, train_epoch

from .config import load_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(architecture, sample, config):
    parameters = {
        "in_channels": sample.x.shape[1],
        "hidden_channels": config.get("hidden_channels", 5),
        "num_layers": config.get("num_layers", 10),
    }
    if architecture in {"gat", "gatv2"} and getattr(sample, "edge_attr", None) is not None:
        parameters["edge_dim"] = sample.edge_attr.shape[1]
    model = create_model(architecture, **parameters)
    constraint = config.get("output_constraint", "abs")
    if constraint == "abs":
        return NonNegativeOutput(model, method="abs")
    if constraint in {"sigmoid", "clamp"}:
        return BoundedOutput(model, upper="upper", method=constraint)
    if constraint == "none":
        return model
    raise ValueError("unknown output constraint: {}".format(constraint))


def _save_checkpoint(path, model, optimizer, loss, config, architecture):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
            "architecture": architecture,
            "experiment": config,
        },
        path,
    )


def run_training(config):
    """Run every architecture in a validated Chapter 4 configuration."""

    if any(name.startswith("revised-") for name in config["architectures"]):
        from .revised import run_revised_training

        return run_revised_training(config)

    seed_everything(config.get("seed", 0))
    device = torch.device(config.get("device", "cpu"))
    dataset = GraphDataset(config["dataset"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    output_dir = Path(config["output_dir"])
    results = {}

    physics_loss = None
    if config["mode"] == "self_supervised":
        repository = CaseRepository(config.get("data_root", "data"))
        physics_loss = BatchedFEMLoss.from_repository(
            repository,
            range(len(dataset)),
            split=config.get("split", "train"),
            variant=config.get("variant", "standard"),
        )

    for architecture in config["architectures"]:
        model = _build_model(architecture, dataset[0], config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
        )
        best = float("inf")
        history = []
        for _ in range(config["epochs"]):
            if physics_loss is None:
                loss = train_epoch(model, loader, optimizer, device)
            else:
                loss = self_supervised_train(model, loader, optimizer, physics_loss, device)
            history.append(loss)
            if loss < best:
                best = loss
                _save_checkpoint(
                    output_dir / "{}.pth".format(architecture),
                    model,
                    optimizer,
                    loss,
                    config,
                    architecture,
                )
        results[architecture] = {"best_loss": best, "history": history}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to an experiment JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    if arguments.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
