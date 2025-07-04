.PHONY: help install dev test lint format clean build run demo docs deploy

# Default target
help:
	@echo "Alpha Architecture Agent - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install    Install dependencies"
	@echo "  dev        Set up development environment"
	@echo "  demo       Run architecture generation demo"
	@echo "  notebook   Start Jupyter notebook server"
	@echo ""
	@echo "Code Quality:"
	@echo "  test       Run test suite"
	@echo "  lint       Run linting checks"
	@echo "  format     Format code with black and isort"
	@echo "  typecheck  Run type checking with mypy"
	@echo ""
	@echo "Docker:"
	@echo "  build      Build Docker images"
	@echo "  run        Start all services with Docker Compose"
	@echo "  stop       Stop all services"
	@echo "  logs       View service logs"
	@echo "  clean      Clean up Docker resources"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy     Deploy to production"
	@echo "  docs       Build documentation"
	@echo ""

# Development setup
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

dev: install
	@echo "Setting up development environment..."
	cp .env.example .env
	@echo "Please edit .env with your API keys"
	@echo "Development environment ready!"

# Demo and testing
demo:
	@echo "Running architecture generation demo..."
	python examples/demo_architecture_generation.py

notebook:
	@echo "Starting Jupyter notebook server..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick tests..."
	pytest tests/ -x --ff

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 src/ tests/ examples/
	mypy src/

format:
	@echo "Formatting code..."
	black src/ tests/ examples/
	isort src/ tests/ examples/

typecheck:
	@echo "Running type checking..."
	mypy src/

# Docker operations
build:
	@echo "Building Docker images..."
	docker-compose build

run:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started. Access points:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - Grafana: http://localhost:3001"
	@echo "  - MLflow: http://localhost:5000"

stop:
	@echo "Stopping all services..."
	docker-compose down

logs:
	@echo "Viewing service logs..."
	docker-compose logs -f

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Database operations
db-init:
	@echo "Initializing database..."
	docker-compose exec postgres psql -U stock_user -d stockprediction -f /docker-entrypoint-initdb.d/init-db.sql

db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

db-reset:
	@echo "Resetting database..."
	docker-compose down postgres
	docker volume rm uki_postgres_data
	docker-compose up -d postgres
	sleep 10
	make db-init

# MLflow operations
mlflow-ui:
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-clean:
	@echo "Cleaning MLflow experiments..."
	mlflow gc --backend-store-uri sqlite:///mlflow.db

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"

metrics:
	@echo "Displaying current metrics..."
	curl -s http://localhost:9090/api/v1/query?query=up | jq .

# Data operations
data-download:
	@echo "Downloading sample data..."
	python scripts/download_sample_data.py

data-preprocess:
	@echo "Preprocessing data..."
	python scripts/preprocess_data.py

data-validate:
	@echo "Validating data quality..."
	python scripts/validate_data.py

# Architecture operations
generate-architectures:
	@echo "Generating architecture suite..."
	python scripts/generate_architectures.py --num-architectures 70

validate-architectures:
	@echo "Validating generated architectures..."
	python scripts/validate_architectures.py

train-models:
	@echo "Training all models..."
	python scripts/train_models.py

backtest:
	@echo "Running backtesting..."
	python scripts/run_backtest.py

# API operations
api-dev:
	@echo "Starting API in development mode..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-test:
	@echo "Testing API endpoints..."
	curl -X GET "http://localhost:8000/health"
	curl -X GET "http://localhost:8000/api/v1/architectures"

# Documentation
docs:
	@echo "Building documentation..."
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation built in docs/_build/html/"

docs-serve:
	@echo "Serving documentation..."
	cd docs/_build/html && python -m http.server 8080

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf docs/_build/

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod:
	@echo "Deploying to production..."
	@echo "This requires production configuration"
	# Add production deployment commands here

# Security
security-scan:
	@echo "Running security scan..."
	bandit -r src/
	safety check

# Performance
profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile.stats scripts/profile_generation.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark:
	@echo "Running benchmarks..."
	python scripts/benchmark_architectures.py

# Utilities
shell:
	@echo "Starting Python shell with project context..."
	python -c "from src.core.config import get_config; from src.agents.architecture_agent import ArchitectureAgent; print('Project context loaded. Available: get_config(), ArchitectureAgent()')"
	python

version:
	@echo "Project version information:"
	@python -c "from src.core.config import get_config; print(f'Version: {get_config().project_version}')"

status:
	@echo "System status:"
	@docker-compose ps
	@echo ""
	@echo "Available endpoints:"
	@echo "  - API: http://localhost:8000/docs"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - Grafana: http://localhost:3001"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"

# Cleanup
clean-all: clean
	@echo "Deep cleaning project..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete