"""
app/fl/server.py
The central FL orchestrator with Byzantine Poisoning Defense.
Creates simulated hospital clients, runs DefendedFedAvg over communication rounds,
evaluates the global model, and saves the verified global model checkpoint.

Run from the `backend/` folder with:
    python -m app.fl.server
"""

import os
import torch
from torch.utils.data import DataLoader
import flwr as fl

from app.fl.model import PneumoniaCNN
from app.data.loader import load_datasets
from app.data.partition import partition_dataset, get_client_loaders
from app.fl.client import FlowerClient
from app.fl.utils import set_parameters, test, save_checkpoint
from app.detection.strategy import DefendedFedAvg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Simulation configuration ----
NUM_CLIENTS = 5
NUM_ROUNDS = 5
MALICIOUS_CLIENTS: list[int] = [1]  # Client 1 mounts label-flipping attacks
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "global_model.pth")

print(f"Using device: {DEVICE}")
print(f"Configured {NUM_CLIENTS} simulated clients. Malicious clients: {MALICIOUS_CLIENTS}")

# Load and partition PneumoniaMNIST once
train_dataset, test_dataset = load_datasets(image_size=64)
client_partitions = partition_dataset(train_dataset, num_clients=NUM_CLIENTS, iid=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def client_fn(context) -> fl.client.Client:
    """Flower client generator for simulated hospital nodes."""
    cid_int = int(context.node_config["partition-id"])
    train_loader, val_loader = get_client_loaders(client_partitions[cid_int], batch_size=32)
    model = PneumoniaCNN().to(DEVICE)
    is_malicious = cid_int in MALICIOUS_CLIENTS
    return FlowerClient(
        model,
        train_loader,
        val_loader,
        DEVICE,
        client_id=cid_int,
        malicious=is_malicious,
    ).to_client()


def get_evaluate_fn():
    """Centralized evaluation function called at the end of each federated round."""
    best_accuracy = [0.0]

    def evaluate(server_round, parameters, config):
        model = PneumoniaCNN().to(DEVICE)
        set_parameters(model, parameters)
        loss, accuracy, metrics = test(model, test_loader, DEVICE)
        print(f"[Round {server_round:>2}] Global Test Accuracy: {accuracy*100:.2f}% | Loss: {loss:.4f} | Sensitivity: {metrics['sensitivity']*100:.1f}% | Specificity: {metrics['specificity']*100:.1f}%")

        if accuracy > best_accuracy[0]:
            best_accuracy[0] = accuracy
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"   [+] Saved new best model checkpoint ({accuracy*100:.2f}%) to {CHECKPOINT_PATH}")

        return loss, {"accuracy": accuracy, "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"]}

    return evaluate


strategy = DefendedFedAvg(
    similarity_threshold=0.0,
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)


def run_federated_simulation(num_rounds: int = NUM_ROUNDS):
    """Execute the full federated learning training run."""
    print(f"\n[+] Starting Federated Learning simulation ({num_rounds} rounds, {NUM_CLIENTS} hospitals)...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    print("\n[+] Federated Learning training complete.")
    return history


if __name__ == "__main__":
    run_federated_simulation(NUM_ROUNDS)
