from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .database import (
    annotation_rows,
    create_annotation,
    delete_annotation,
    image_exists,
    image_rows,
    initialize_database,
)
from .models import Annotation, AnnotationCreate, ImageMetadata

app = FastAPI(title="ImageNet100 Annotation API")

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


def row_to_dict(row):
    return dict(row)


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
