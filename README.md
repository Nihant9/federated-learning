# Federated Learning with Poisoning Attack Detection & Blockchain Defense

An end-to-end trustworthy Federated Learning framework designed for healthcare (PneumoniaMNIST X-ray diagnosis) featuring adversarial poisoning resilience, anomaly detection, and blockchain-based auditability.

---

## 🏗️ System Architecture

```
fl-poisoning-blockchain/
├── backend/
│   ├── app/
│   │   ├── data/             # PneumoniaMNIST data loaders & partitioning (IID / Non-IID)
│   │   ├── fl/               # Flower simulation engine (PneumoniaCNN, clients, server)
│   │   ├── detection/        # Anomaly detection & robust aggregation strategies
│   │   ├── blockchain/       # Audit trail, smart contracts & client reputation
│   │   └── routes/           # FastAPI REST and WebSocket endpoints
│   ├── tests/                # Unit and integration test suite
│   └── venv/                 # Virtual environment (ignored by Git)
├── frontend/                 # Web dashboard for monitoring FL rounds & blockchain
├── experiments/              # Experiment logs and benchmark results (clean/attacked/defended)
└── docs/                     # Documentation and architecture diagrams
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.x
- CUDA (optional, CPU fallback supported)

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Federated Learning Simulation
```bash
cd backend
python -m app.fl.server
```

---

## 🛡️ Core Features
- **Privacy-Preserving Training**: Distributed training using Flower (`flwr`) without raw medical data sharing.
- **Threat Model**: Data poisoning simulation via targeted label-flipping attacks.
- **Robust Defense**: Byzantine-resilient aggregation and cosine distance anomaly detection.
- **Blockchain Audit**: Immutable verification of model updates, contribution weights, and malicious participant slashing.
