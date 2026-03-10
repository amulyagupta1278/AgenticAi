.PHONY: setup ingest train train-all api test docker-build docker-run mlflow-ui clean \
       download build-rag generate-kb simulate monitoring

setup:
	pip install -r requirements.txt

download:
	python -m src.data.download_data

ingest:
	python -m src.data.preprocess

train:
	python -m src.models.train --model logistic_regression

train-all:
	python -m src.models.train --all

build-rag:
	python scripts/build_rag_index.py

generate-kb:
	python scripts/generate_kb.py

simulate:
	python scripts/run_simulation.py --sample 500

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

docker-build:
	docker build -t ticket-classifier:latest .

docker-run:
	docker run --rm -p 8000:8000 ticket-classifier:latest

monitoring:
	cd deployment && docker compose --profile monitoring up -d

mlflow-ui:
	mlflow ui --backend-store-uri file://$(shell pwd)/mlruns --port 5001

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; \
	echo "Cleaned"
