.PHONY: help install test train evaluate serve lint clean

help:
	@echo "Reinsurance Loss Development - Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make test        Run unit tests"
	@echo "  make train       Train ML model"
	@echo "  make evaluate    Evaluate trained model"
	@echo "  make serve       Start FastAPI server"
	@echo "  make lint        Run linting/formatting checks"
	@echo "  make clean       Remove generated files and caches"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

train:
	python scripts/train.py \
		--data data/processed/features.parquet \
		--output models/saved/ \
		--val-split 0.2 \
		--random-seed 42

evaluate:
	python scripts/evaluate.py \
		--model models/saved/model.pkl \
		--data data/processed/test.parquet \
		--divergence-threshold 0.15

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

lint:
	black . --check
	isort . --check-only
	pylint ingestion/ features/ models/ evaluation/ api/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf build/ dist/ *.egg-info
	rm -rf data/processed/*

.DEFAULT_GOAL := help
