# RL-Snake Makefile

.PHONY: help install install-dev test train play evaluate clean lint format

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	python -m pytest tests/ -v

test-coverage:  ## Run tests with coverage
	python -m pytest tests/ --cov=rl_snake --cov-report=html

train:  ## Train a PPO model (default settings)
	python scripts/train.py -m PPO -g 15 -x 5

train-dqn:  ## Train a DQN model
	python scripts/train.py -m DQN -g 15 -x 5

play:  ## Play with the default trained model
	python scripts/play.py -m PPO_snake

evaluate:  ## Evaluate the default trained model
	python scripts/evaluate.py -m PPO_snake -e 50

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

lint:  ## Run linting
	flake8 src/ tests/ scripts/
	mypy src/

format:  ## Format code
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

setup-env:  ## Set up development environment
	python -m venv env
	@echo "Activate with: source env/bin/activate"

tensorboard:  ## Start TensorBoard
	tensorboard --logdir=logs

# Training variations
train-small:  ## Train on small grid (10x10)
	python scripts/train.py -m PPO -g 10 -x 3

train-large:  ## Train on large grid (25x25)
	python scripts/train.py -m PPO -g 25 -x 7

train-fast:  ## Quick training session
	python scripts/train.py -m PPO -g 15 -x 1

# Development tasks
check:  ## Run all checks (lint, test)
	$(MAKE) lint
	$(MAKE) test

build:  ## Build the package
	python -m build


