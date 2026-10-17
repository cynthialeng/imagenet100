from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
from pathlib import Path

from .database import (
    annotation_rows,
    create_annotation,
    delete_annotation,
    image_exists,
    image_rows,
    initialize_database,
)
from .models import Annotation, AnnotationCreate, ImageMetadata, Prediction
from .prediction import predict_image
from .results import router as results_router
from .training import router as training_router, stop_all_runs

app = FastAPI(title="ImageNet100 Annotation API")
app.include_router(results_router)
app.include_router(training_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    initialize_database()


@app.on_event("shutdown")
def shutdown():
    stop_all_runs()


def row_to_dict(row):
    return dict(row)


HW2_DATA = Path(__file__).resolve().parent.parent / "data" / "hw2"


def prediction_samples():
    manifest = HW2_DATA / "metadata.json"
    if not manifest.is_file():
        raise HTTPException(status_code=503, detail="Prediction sample dataset is unavailable")
    return [row for row in json.loads(manifest.read_text())["images"] if row["split"] == "test"]


@app.get("/prediction-samples")
def get_prediction_samples():
    return [
        {"id": index, "filename": Path(row["path"]).name,
         "ground_truth_label": row["class_name"], "url": f"/prediction-samples/{index}/image"}
        for index, row in enumerate(prediction_samples())
    ]


@app.get("/prediction-samples/{sample_id}/image")
def get_prediction_sample_image(sample_id: int):
    samples = prediction_samples()
    if sample_id < 0 or sample_id >= len(samples):
        raise HTTPException(status_code=404, detail="Prediction sample not found")
    path = (HW2_DATA / samples[sample_id]["path"]).resolve()
    if not path.is_relative_to(HW2_DATA.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Prediction image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/predict", response_model=Prediction)
def predict(content: bytes = Body(default=b"", media_type="application/octet-stream")):
    """Send raw image bytes. Predictions are limited to the three trained classes.

    Probabilities are model scores, not a guarantee of correctness; this model
    cannot identify images outside its three supported classes.
    """
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")
    try:
        return predict_image(content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Trained model checkpoint is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/images", response_model=list[ImageMetadata])
def get_images():
    return [row_to_dict(row) for row in image_rows()]


@app.get("/images/{image_id}/annotations", response_model=list[Annotation])
def get_annotations(image_id: int):
    if not image_exists(image_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return [row_to_dict(row) for row in annotation_rows(image_id)]


@app.post(
    "/images/{image_id}/annotations",
    response_model=Annotation,
    status_code=status.HTTP_201_CREATED,
)
def add_annotation(image_id: int, annotation: AnnotationCreate):
    if not image_exists(image_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    row = create_annotation(image_id, **annotation.model_dump())
    return row_to_dict(row)


@app.delete("/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_annotation(annotation_id: int):
    if not delete_annotation(annotation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found")
