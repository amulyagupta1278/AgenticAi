# Customer Support Ticket Classifier

An end-to-end MLOps pipeline that automatically classifies customer support tickets by issue type. Built with scikit-learn and TF-IDF, experiment-tracked with MLflow, and served through a FastAPI REST API.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd AgenticAi
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download and Process Dataset

The dataset is the [Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) dataset on Kaggle (50k tickets, 5 languages, 10 routing queues).

Run the download script first, then preprocess:

```bash
python src/data/download_data.py
python src/data/preprocess.py
```

`download_data.py` will try the Kaggle CLI automatically. If you don't have it set up:

1. Visit: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
2. Click Download and extract the zip
3. Place the CSV in `data/raw/`
4. If the filename differs from `customer_support_tickets.csv`, update `FILENAME` in `src/data/download_data.py`
5. Then run `python src/data/preprocess.py`

By default only English (`EN`) tickets are kept for training. To include all languages:

```bash
python src/data/preprocess.py --lang all
```

---

## Exploratory Data Analysis

Open the EDA notebook to explore the dataset before training:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This notebook covers:

1. **Data Acquisition** — load raw CSV, inspect shape, columns, and dtypes
2. **Data Cleaning and Preprocessing**
   - **2.1 Analysis of Raw Data** — null values, duplicates, raw class and language distribution
   - **2.2 Analysis of Preprocessed Data** — compare raw vs cleaned dataset (rows removed, label changes)
3. **Exploratory Data Analysis (EDA)** — text length statistics, priority patterns, language × queue relationships
4. **Visualizations** — TF-IDF keyword analysis per category, vocabulary overlap (Jaccard) heatmap

All figures are saved automatically to `notebooks/figures/`.

---

## Build Feature Store

Vectorize the processed data and cache the TF-IDF matrices to disk so training runs don't re-vectorize from scratch each time:

```bash
python -m src.features.feature_store
```

This saves the following to `data/features/v1/`:

```
X_train.npz          sparse TF-IDF matrix (train)
X_val.npz            sparse TF-IDF matrix (val)
X_test.npz           sparse TF-IDF matrix (test)
y_train.npy          label array (train)
y_val.npy            label array (val)
y_test.npy           label array (test)
tfidf_vectorizer.pkl fitted TF-IDF vectorizer
label_encoder.pkl    fitted LabelEncoder
meta.json            provenance info (split sizes, timestamp, classes)
```

Default split: 80% train / 10% val / 10% test (stratified).

---

## Usage

### 1. Model Training

Train models with MLflow tracking (run from project root):

```bash
# Train the default model (logistic regression)
python src/models/train.py

# Train a specific model
python src/models/train.py --model logistic_regression
python src/models/train.py --model linear_svc
python src/models/train.py --model random_forest

# Train all models and automatically keep the best one
python src/models/train.py --all
```

Or run the full pipeline in one shot:

```bash
python run_pipeline.py
```

**View MLflow UI:**

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001
# Open browser: http://localhost:5001
```

### 2. Run API Locally

Start the FastAPI server (make sure you've trained a model first):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Test API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"description": "My app keeps crashing every time I try to upload a file"}'
```

Example response:

```json
{
  "ticket_type": "technical_support",
  "confidence": 0.87,
  "all_scores": {
    "billing_and_payments": 0.03,
    "general_inquiry": 0.10,
    "technical_support": 0.87
  }
}
```

Additional endpoints:

```bash
# Check if the model is loaded
curl http://localhost:8000/health

# List all ticket categories the model knows
curl http://localhost:8000/classes
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## Docker Deployment

Train the model before building — the Dockerfile copies `models/` into the image.

### Build Docker Image

```bash
# Make sure Docker Desktop is running

# Build Docker image (from project root)
docker build -t ticket-classifier:latest .

# For Apple Silicon (M1/M2/M3), specify platform if needed:
docker build --platform linux/arm64 -t ticket-classifier:latest .
```

### Run Container Locally

```bash
# Run container
docker run -d --name test-api -p 8000:8000 ticket-classifier:latest

# Wait a moment for the server to start
sleep 5

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "I was charged twice for my subscription this month"}'

# View logs
docker logs test-api

# Stop and remove container
docker stop test-api && docker rm test-api
```

### Docker Compose (local dev — mounts models volume)

```bash
# Start the API (models/ is mounted, no rebuild needed after retraining)
docker compose -f deployment/docker-compose.yml up

# Also spin up the MLflow UI
docker compose -f deployment/docker-compose.yml --profile mlflow up
```
