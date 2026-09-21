# ImageNet100 Viewer

An academic React web application for exploring a 50-image subset of the ImageNet100 dataset. The viewer displays each image, its ground-truth classification label, and user-created bounding-box annotations.

## Data

The visualized data is image classification research data. The sample images and their existing labels are stored in `public/samples/metadata.json`. User annotations contain a text label and normalized bounding-box coordinates.

## Features

- Browse and select sample images.
- View filenames and ground-truth classification labels.
- Zoom and pan the selected image.
- Draw multiple bounding boxes in Annotation Mode.
- Add text labels to regions, save them to SQLite, retrieve them, and delete them.

Annotation and backend/database requirements are implemented with a FastAPI REST API and SQLite database.

## Run the backend with Docker

From the project root:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000` and its interactive documentation is at `http://localhost:8000/docs`. The SQLite database is stored in `backend/data/`, which is mounted as a persistent Docker bind mount.

## Run the frontend

In a second terminal:

```bash
npm install
npm run dev
```

Open `http://localhost:5173/`. Start the backend before using annotations. The frontend communicates with the backend over HTTP and never accesses SQLite directly.

## Run the backend without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## Main libraries

- React and Vite for the frontend.
- FastAPI and Uvicorn for the REST backend.
- SQLite from Python's standard library for persistence.
- Docker Compose for local backend setup.
