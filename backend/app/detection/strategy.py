"""
app/detection/strategy.py

Byzantine-resilient Federated Aggregation Strategy.
Wraps Flower's FedAvg to detect and reject adversarial updates (label-flipping / poisoned weights)
using pairwise Cosine Similarity anomaly detection and generates cryptographic audit hashes.
"""

from typing import List, Tuple, Dict, Optional, Union
import hashlib
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy


class DefendedFedAvg(fl.server.strategy.FedAvg):
    """
    Byzantine-tolerant Federated Averaging strategy.
    Monitors client weight updates, computes pairwise cosine similarity,
    and discards malicious / inverted updates prior to aggregation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.0,
        blockchain_logger=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.blockchain_logger = blockchain_logger
        self.audit_log: List[Dict] = []

    def _flatten_ndarrays(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Flatten a list of parameter tensors into a single 1D vector."""
        return np.concatenate([arr.flatten() for arr in arrays])

    def _compute_hash(self, array: np.ndarray) -> str:
        """Generate SHA-256 digest of client update vector for blockchain proof."""
        return hashlib.sha256(array.tobytes()).hexdigest()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Custom aggregation step with Byzantine poisoning defense.
        """
        if not results:
            return None, {}

        # 1. Unpack client updates
        client_vectors = []
        client_metadata = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_vector = self._flatten_ndarrays(ndarrays)
            client_vectors.append(flat_vector)

            cid = fit_res.metrics.get("client_id", client_proxy.cid)
            client_metadata.append({
                "client_proxy": client_proxy,
                "fit_res": fit_res,
                "client_id": cid,
                "num_examples": fit_res.num_examples,
                "vector_norm": float(np.linalg.norm(flat_vector)),
                "hash": self._compute_hash(flat_vector),
            })

        num_clients = len(client_vectors)

        # If only 1-2 clients, fall back to default aggregation
        if num_clients < 3:
            print(f"[Round {server_round}] Too few clients for consensus defense. Aggregating all.")
            return super().aggregate_fit(server_round, results, failures)

        # 2. Compute pairwise Cosine Similarity Matrix
        norms = [np.linalg.norm(v) for v in client_vectors]
        sim_matrix = np.zeros((num_clients, num_clients))

        for i in range(num_clients):
            for j in range(i, num_clients):
                if norms[i] > 1e-9 and norms[j] > 1e-9:
                    sim = float(np.dot(client_vectors[i], client_vectors[j]) / (norms[i] * norms[j]))
                else:
                    sim = 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # 3. Calculate mean consensus similarity for each client
        avg_similarities = []
        for i in range(num_clients):
            other_sims = [sim_matrix[i, j] for j in range(num_clients) if i != j]
            avg_sim = float(np.mean(other_sims)) if other_sims else 1.0
            avg_similarities.append(avg_sim)

        # 4. Filter honest vs malicious clients
        clean_results: List[Tuple[ClientProxy, FitRes]] = []
        flagged_clients = []

        # Relative outlier threshold (e.g. median - 1.5 * std or threshold)
        median_sim = float(np.median(avg_similarities))

        for i in range(num_clients):
            client_info = client_metadata[i]
            sim_score = avg_similarities[i]
            cid = client_info["client_id"]

            # Poisoning criterion: significantly deviates from consensus cluster
            is_malicious = (sim_score < self.similarity_threshold) or (sim_score < (median_sim - 0.25))

            if is_malicious:
                flagged_clients.append(cid)
                print(f"[!] [Round {server_round}] BLOCKED MALICIOUS CLIENT {cid} (Consensus Sim: {sim_score:.4f}, Hash: {client_info['hash'][:8]}...)")
                self.audit_log.append({
                    "round": server_round,
                    "client_id": cid,
                    "status": "REJECTED_POISON",
                    "similarity_score": round(sim_score, 4),
                    "update_hash": client_info["hash"],
                })
            else:
                clean_results.append((client_info["client_proxy"], client_info["fit_res"]))
                self.audit_log.append({
                    "round": server_round,
                    "client_id": cid,
                    "status": "ACCEPTED_CLEAN",
                    "similarity_score": round(sim_score, 4),
                    "update_hash": client_info["hash"],
                })

        print(f"[Round {server_round}] Byzantine Defense: {len(clean_results)}/{num_clients} updates accepted. Blocked clients: {flagged_clients}")

        # If all were flagged (unlikely edge case), use top half
        if not clean_results:
            print(f"[Round {server_round}] Caution: Fallback to top-ranked updates.")
            top_idx = int(np.argmax(avg_similarities))
            clean_results = [(client_metadata[top_idx]["client_proxy"], client_metadata[top_idx]["fit_res"])]

        # 5. Aggregate only sanitized, honest client weights
        aggregated_parameters, metrics = super().aggregate_fit(server_round, clean_results, failures)

        # Attach defense telemetry to metrics
        metrics["accepted_clients"] = len(clean_results)
        metrics["blocked_clients"] = len(flagged_clients)

        return aggregated_parameters, metrics
