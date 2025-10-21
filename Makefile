# Makefile for Bengali Medical Chatbot Project

.PHONY: help install setup data train evaluate demo clean test lint format

# Default target
help:
	@echo "Bengali Medical Chatbot - Available Commands:"
	@echo "============================================="
	@echo "setup          - Set up the development environment"
	@echo "install        - Install Python dependencies"
	@echo "data           - Download and prepare datasets"
	@echo "translate      - Run translation pipeline"
	@echo "train          - Train the medical chatbot model"
	@echo "evaluate       - Evaluate model performance"
	@echo "demo           - Run Streamlit demo application"
	@echo "notebook       - Start Jupyter notebook server"
	@echo "test           - Run unit tests"
	@echo "lint           - Run code linting"
	@echo "format         - Format code with black"
	@echo "clean          - Clean temporary files and caches"
	@echo "docker-build   - Build Docker image"
	@echo "docker-run     - Run Docker container"

# Environment setup
setup: install
	@echo "Setting up development environment..."
	@mkdir -p data/{raw,processed,translated,augmented}
	@mkdir -p experiments/{logs,models,results}
	@mkdir -p tests/fixtures
	@echo "✓ Directory structure created"
	@echo "✓ Setup complete!"

install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Data pipeline
data:
	@echo "Downloading datasets..."
	@python scripts/download_data.py
	@echo "✓ Data download complete"

translate:
	@echo "Running translation pipeline..."
	@python scripts/translate_data.py --config config/data_config.yaml
	@echo "✓ Translation complete"

preprocess:
	@echo "Preprocessing data..."
	@python scripts/preprocess_data.py --config config/data_config.yaml
	@echo "✓ Preprocessing complete"

# Model training and evaluation
train:
	@echo "Training Bengali medical chatbot..."
	@python scripts/train_model.py --config config/training_config.yaml
	@echo "✓ Training complete"

evaluate:
	@echo "Evaluating model performance..."
	@python scripts/evaluate_model.py --config config/model_config.yaml
	@echo "✓ Evaluation complete"

# Demo and development
demo:
	@echo "Starting Streamlit demo..."
	@streamlit run deployment/streamlit_app.py

notebook:
	@echo "Starting Jupyter notebook server..."
	@jupyter notebook notebooks/

# Code quality
test:
	@echo "Running unit tests..."
	@python -m pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests complete"

lint:
	@echo "Running code linting..."
	@flake8 src/ scripts/ tests/
	@mypy src/
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	@black src/ scripts/ tests/
	@isort src/ scripts/ tests/
	@echo "✓ Code formatted"

# Docker
docker-build:
	@echo "Building Docker image..."
	@docker build -t bengali-medical-chatbot -f deployment/docker/Dockerfile .
	@echo "✓ Docker image built"

docker-run:
	@echo "Running Docker container..."
	@docker run -p 8501:8501 bengali-medical-chatbot
	@echo "✓ Docker container started"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@rm -rf dist/
	@rm -rf build/
	@rm -rf *.egg-info/
	@echo "✓ Cleanup complete"

# Full pipeline
full-pipeline: setup data translate preprocess train evaluate
	@echo "✓ Full pipeline complete!"

# Development workflow
dev-setup: setup
	@echo "Setting up development environment..."
	@pre-commit install
	@echo "✓ Development setup complete"

# Research workflow
research: data translate
	@echo "Starting research workflow..."
	@jupyter notebook notebooks/01_data_exploration.ipynb
	@echo "✓ Research environment ready"

# Production deployment
deploy-prep: test lint
	@echo "Preparing for deployment..."
	@python scripts/export_model.py
	@echo "✓ Deployment preparation complete"

# Quick start for new users
quickstart:
	@echo "Bengali Medical Chatbot - Quick Start"
	@echo "====================================="
	@echo "1. Setting up environment..."
	@make setup
	@echo "2. Downloading sample data..."
	@python scripts/download_data.py --dataset bangla_health
	@echo "3. Starting demo..."
	@make demo
	@echo "✓ Quick start complete!"

# Help for specific commands
help-data:
	@echo "Data Commands:"
	@echo "  data          - Download all datasets"
	@echo "  translate     - Translate English datasets to Bengali"
	@echo "  preprocess    - Clean and prepare data for training"

help-train:
	@echo "Training Commands:"
	@echo "  train         - Train the model with current config"
	@echo "  evaluate      - Evaluate trained model"
	@echo "  train-debug   - Train with debug settings"

help-dev:
	@echo "Development Commands:"
	@echo "  test          - Run all tests"
	@echo "  lint          - Check code quality"
	@echo "  format        - Format code"
	@echo "  notebook      - Start Jupyter server"

# Advanced targets
train-debug:
	@echo "Training in debug mode..."
	@python scripts/train_model.py --config config/training_config.yaml --debug
	@echo "✓ Debug training complete"

profile:
	@echo "Profiling model performance..."
	@python -m cProfile -o profile_results.prof scripts/train_model.py
	@echo "✓ Profiling complete"

benchmark:
	@echo "Running benchmarks..."
	@python scripts/benchmark_model.py
	@echo "✓ Benchmarking complete"
