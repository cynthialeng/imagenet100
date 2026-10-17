FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY backend ./backend
COPY public ./public
COPY training/model.py training/data.py training/train.py ./training/
COPY training/outputs/two_hidden/best_model.pt ./training/outputs/two_hidden/best_model.pt
COPY data/hw2/metadata.json ./data/hw2/metadata.json
COPY data/hw2/test ./data/hw2/test
COPY data/hw2/train ./data/hw2/train
COPY data/hw2/val ./data/hw2/val
COPY training/outputs/test_results.json ./training/outputs/test_results.json
COPY training/outputs/two_hidden/history.csv training/outputs/two_hidden/training_curves.png ./training/outputs/two_hidden/

ENV DATABASE_PATH=/app/backend/data/annotations.db
ENV METADATA_PATH=/app/public/samples/metadata.json

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
