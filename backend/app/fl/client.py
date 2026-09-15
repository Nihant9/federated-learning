"""
app/fl/client.py
The Flower client — represents one simulated hospital.
Handles local training and evaluation only; has no knowledge
of detection or blockchain.
"""

import flwr as fl
from app.fl.utils import get_parameters, set_parameters, train, test


class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        client_id=None,
        malicious=False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.malicious = malicious

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Load global model parameters
        set_parameters(self.model, parameters)

        # Local training (2 epochs per round for robust convergence)
        train(
            self.model,
            self.train_loader,
            self.device,
            epochs=2,
            lr=0.0004,
            malicious=self.malicious
        )

        # Return updated parameters
        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "client_id": self.client_id,
                "malicious": self.malicious
            },
        )

    def evaluate(self, parameters, config):
        # Load global model parameters
        set_parameters(self.model, parameters)

        # Evaluate local model
        loss, accuracy, metrics = test(
            self.model,
            self.val_loader,
            self.device
        )

        return (
            float(loss),
            len(self.val_loader.dataset),
            {"accuracy": float(accuracy), "sensitivity": float(metrics.get("sensitivity", 0.0))}
        )