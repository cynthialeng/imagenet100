from datetime import datetime
from pydantic import BaseModel, Field, model_validator

class ImageMetadata(BaseModel):
    id: int
    filename: str
    ground_truth_label: str

class ClassProbability(BaseModel):
    class_name: str
    probability: float

class Prediction(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: list[ClassProbability]
    supported_classes: list[str]

class AnnotationCreate(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def stays_inside_image(self):
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("Bounding box must stay within the image boundaries")
        return self

class Annotation(BaseModel):
    id: int
    image_id: int
    label: str
    x: float
    y: float
    width: float
    height: float
    created_at: datetime
