"""Local training jobs and live metrics. Added with assistance from OpenAI Codex."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from threading import Lock
from time import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader

from training.data import HW2Dataset
from training.train import evaluate
from .prediction import load_model, predict_image


router = APIRouter(prefix="/training")
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "training" / "outputs" / "interactive"
lock = Lock()
processes = {}


class TrainingConfig(BaseModel):
    learning_rate: float = Field(default=0.01, ge=0.0001, le=0.1, allow_inf_nan=False)
    epochs: int = Field(default=50, ge=1, le=200)


def run_folder(run_id):
    try:
        if str(UUID(run_id)) != run_id:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Training run not found") from exc
    path = RUNS / run_id
    if not (path / "run.json").is_file():
        raise HTTPException(status_code=404, detail="Training run not found")
    return path


def read_status(path):
    info = json.loads((path / "run.json").read_text())
    process = processes.get(info["id"])
    if info["status"] == "running":
        if process is None:
            info["status"] = "interrupted"
            info["error"] = "The backend restarted. Start a new run."
        elif process.poll() is not None:
            info["status"] = "completed" if process.returncode == 0 else "failed"
            if process.returncode:
                info["error"] = "Training failed. See this run's training.log for details."
        if info["status"] != "running":
            (path / "run.json").write_text(json.dumps(info, indent=2))
    history = []
    history_path = path / "model" / "history.csv"
    if history_path.is_file():
        with history_path.open(newline="") as file:
            for row in csv.DictReader(file):
                # A poll can occur while the subprocess is writing a CSV row.
                try:
                    metric = {key: float(value) for key, value in row.items()}
                    if not all(math.isfinite(value) for value in metric.values()):
                        continue
                    metric["epoch"] = int(metric["epoch"])
                    history.append(metric)
                except (TypeError, ValueError):
                    continue
    info["history"] = history
    info["best"] = min(history, key=lambda row: row["val_loss"]) if history else None
    test_path = path / "test_results.json"
    if info["status"] == "completed" and not test_path.is_file():
        # Evaluate the selected checkpoint once, after all training has finished.
        # Test metrics never influence updates or checkpoint selection.
        save_test_results(path, info)
    info["test"] = json.loads(test_path.read_text()) if test_path.is_file() else None
    return info


def save_test_results(path, info):
    model, classes = load_model(path / "model" / "best_model.pt")
    dataset = HW2Dataset("test")
    if classes != dataset.classes:
        raise HTTPException(status_code=409, detail="Model class order does not match test data")
    loss, accuracy = evaluate(model, DataLoader(dataset, batch_size=32), nn.CrossEntropyLoss())
    result = {"loss": loss, "accuracy": accuracy, "error_rate": 1 - accuracy,
              "images": len(dataset), "epoch": info["best"]["epoch"]}
    (path / "test_results.json").write_text(json.dumps(result, indent=2))


@router.post("/runs", status_code=202)
def start_run(config: TrainingConfig):
    with lock:
        if any(process.poll() is None for process in processes.values()):
            raise HTTPException(status_code=409, detail="A training run is already active")
        # Check data before launching a process so missing data gets a clear error.
        try:
            HW2Dataset("train")
            HW2Dataset("val")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Training data is unavailable") from exc
        run_id = str(uuid4())
        path = RUNS / run_id
        path.mkdir(parents=True)
        info = {"id": run_id, "status": "running", "created_at": time(), "config": config.model_dump()}
        (path / "run.json").write_text(json.dumps(info, indent=2))
        try:
            with (path / "training.log").open("w") as log:
                processes[run_id] = subprocess.Popen([
                    sys.executable, "-u", "-m", "training.train", "--model", "mlp2",
                    "--epochs", str(config.epochs), "--lr", str(config.learning_rate),
                    "--output-dir", str(path / "model"),
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            info.update(status="failed", error="Could not start the training process")
            (path / "run.json").write_text(json.dumps(info, indent=2))
            raise HTTPException(status_code=503, detail=info["error"]) from exc
        return read_status(path)


@router.get("/runs/latest")
def latest_run():
    with lock:
        paths = list(RUNS.glob("*/run.json"))
        if not paths:
            return None
        return read_status(max(paths, key=lambda path: json.loads(path.read_text())["created_at"]).parent)


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    with lock:
        return read_status(run_folder(run_id))


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    with lock:
        path = run_folder(run_id)
        info = read_status(path)
        if info["status"] == "running":
            terminate(processes[run_id])
            info = json.loads((path / "run.json").read_text())
            info["status"] = "stopped"
            (path / "run.json").write_text(json.dumps(info, indent=2))
        return read_status(path)


def stop_all_runs():
    with lock:
        for run_id, process in processes.items():
            if process.poll() is None:
                terminate(process)
                path = RUNS / run_id / "run.json"
                info = json.loads(path.read_text())
                info["status"] = "interrupted"
                path.write_text(json.dumps(info, indent=2))


def completed_run(run_id):
    path = run_folder(run_id)
    info = read_status(path)
    if info["status"] != "completed":
        raise HTTPException(status_code=409, detail="Wait for this run to finish first")
    return path, info


@router.post("/runs/{run_id}/test")
def test_run(run_id: str):
    # Retained for API compatibility; completed status includes cached test metrics.
    with lock:
        _, info = completed_run(run_id)
        return info


@router.post("/runs/{run_id}/predict")
def predict_run(run_id: str, content: bytes = Body(default=b"", media_type="application/octet-stream")):
    with lock:
        path, _ = completed_run(run_id)
    if not content or len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Provide an image between 1 byte and 10 MB")
    try:
        return predict_image(content, path / "model" / "best_model.pt")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
