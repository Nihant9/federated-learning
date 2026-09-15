"""
app/fl/server.py
The central FL orchestrator. Creates simulated clients, runs FedAvg over
several rounds, and evaluates the global model each round.

This file is the plain FL baseline (Modules 1-2). No attack, detection,
or blockchain logic lives here — those hook in later without changing
this file's core structure:
  - Module 3 (attack)     -> set MALICIOUS_CLIENTS below
  - Module 4 (detection)  -> a custom Strategy wraps FedAvg's aggregate_fit
  - Module 5 (blockchain) -> called from inside that same custom Strategy

Run from the `backend/` folder with:  python -m app.fl.server
"""

import torch
from torch.utils.data import DataLoader

import flwr as fl
from app.fl.model import PneumoniaCNN
from app.data.loader import load_datasets
from app.data.partition import partition_dataset, get_client_loaders
from app.fl.client import FlowerClient
from app.fl.utils import set_parameters, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Simulation configuration ----
# NOTE: once app/config.py exists, these four lines should be replaced with
# imports from there (single source of truth) rather than defined here.
NUM_CLIENTS = 5
NUM_ROUNDS = 4

MALICIOUS_CLIENTS: list[int] = []   # e.g. [3, 6] once Module 3 is wired in

print(f"Using device: {DEVICE}")

# ---- Load and partition data once, shared across all simulated clients ----
train_dataset, test_dataset = load_datasets()
client_partitions = partition_dataset(train_dataset, num_clients=NUM_CLIENTS, iid=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def client_fn(context) -> fl.client.Client:
    """Flower calls this to spin up each simulated client by partition id."""
    cid_int = int(context.node_config["partition-id"])
    train_loader, val_loader = get_client_loaders(client_partitions[cid_int])
    model = PneumoniaCNN().to(DEVICE)
    is_malicious = cid_int in MALICIOUS_CLIENTS
    return FlowerClient(
        model, train_loader, val_loader, DEVICE,
        client_id=cid_int, malicious=is_malicious,
    ).to_client()


def get_evaluate_fn():
    """Server-side evaluation of the global model, run after each round."""
    def evaluate(server_round, parameters, config):
        model = PneumoniaCNN().to(DEVICE)
        set_parameters(model, parameters)
        loss, accuracy = test(model, test_loader, DEVICE)
        print(f"[Round {server_round:>3}] global accuracy: {accuracy:.4f}  loss: {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,          # using centralized server-side eval instead
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)

if __name__ == "__main__":
    # Note: start_simulation() prints one deprecation warning on startup
    # (Flower is migrating toward a `flwr run` / ClientApp-ServerApp project
    # structure in future versions). It is fully functional today and is
    # still what most current Flower tutorials use — safe to ignore for
    # this project.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},  # bump num_gpus if using the 3050
    )
    print("\nTraining complete.")
    print(history.metrics_centralized)
