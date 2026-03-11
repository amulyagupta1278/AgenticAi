FROM python:3.11-slim

WORKDIR /app

# system deps only if needed (none currently)
# RUN apt-get update && apt-get install -y --no-install-recommends ... && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source, model artifacts, and knowledge base (RAG index)
COPY src/ src/
COPY models/classifier.pkl        models/classifier.pkl
COPY models/tfidf_vectorizer.pkl  models/tfidf_vectorizer.pkl
COPY models/label_encoder.pkl     models/label_encoder.pkl
COPY knowledge_base/ knowledge_base/
COPY data/processed/training_dataset.csv data/processed/training_dataset.csv

EXPOSE 8000

# use gunicorn in prod, uvicorn works fine for staging
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
