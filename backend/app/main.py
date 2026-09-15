"""
app/main.py

FastAPI Application Entry Point.
Provides REST and WebSocket endpoints for real-time Pneumonia prediction,
adversarial poisoning detection, and serves the diagnostic web dashboard.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.api import router as api_router, get_model

app = FastAPI(
    title="Federated Pneumonia AI & Poisoning Defense",
    description="Privacy-Preserving Federated Healthcare with Byzantine Adversarial Poisoning Detection",
    version="2.0.0",
)

# Enable CORS for cross-origin web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount static files directory for frontend assets
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    """Warm up model and verify checkpoints on server boot."""
    print("[+] Starting Federated Healthcare Diagnostic API...")
    get_model()
    print("[+] Model loaded and ready for inference & poison auditing.")


@app.get("/")
async def serve_index():
    """Serve the interactive web diagnosis & security dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Federated Pneumonia AI API is running. UI index.html not yet found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
