"""Read saved experiment results; added with assistance from OpenAI Codex."""

import csv
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "training" / "outputs"


@router.get("/model-results")
def get_model_results():
    try:
        report = json.loads((OUTPUTS / "test_results.json").read_text())
        with (OUTPUTS / "two_hidden" / "history.csv").open(newline="") as file:
            history = list(csv.DictReader(file))
        manifest = json.loads((ROOT / "data" / "hw2" / "metadata.json").read_text())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Saved model results are unavailable") from exc
    selected_epoch = report["models"]["two_hidden"]["selected_epoch"]
    selected = next((row for row in history if int(row["epoch"]) == selected_epoch), None)
    if selected is None:
        raise HTTPException(status_code=503, detail="Selected epoch is missing from training history")
    # Return presentation data, excluding local filesystem paths in the report.
    return {
        "classes": report["classes"],
        "architecture": [3072, 128, 64, 3],
        "epochs": len(history),
        "selected_epoch": selected_epoch,
        "selected_metrics": {key: float(selected[key]) for key in
                             ("train_loss", "train_accuracy", "val_loss", "val_accuracy")},
        "split_counts": {split: sum(row["split"] == split for row in manifest["images"])
                         for split in ("train", "val", "test")},
        "models": {name: {key: value for key, value in result.items() if key != "checkpoint"}
                   for name, result in report["models"].items()},
        "curves_url": "/model-results/curves",
    }


@router.get("/model-results/curves")
def get_model_curves():
    path = OUTPUTS / "two_hidden" / "training_curves.png"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Generate the two-hidden-layer training plot first")
    return FileResponse(path, media_type="image/png")
