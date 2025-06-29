VENV_NAME = .venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

.PHONY: help install dataset model run clean fclean

help:
	@echo "Available commands:"
	@echo "  install - Create Python 3.11 environment and install dependencies"
	@echo "  dataset - Convert and execute data preprocessing notebook"
	@echo "  model - Convert and execute MobileNetV2 model notebook"
	@echo "  clean - Remove cache files only"
	@echo "  fclean - Full clean (remove everything)"

install:
	@echo "Installing Python 3.11 and dependencies..."
	@if ! command -v python3.11 >/dev/null 2>&1; then \
		echo "Python 3.11 not found. Installing..."; \
		sudo apt update; \
		sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils; \
	fi
	@echo "Creating Python 3.11 virtual environment..."
	python3.11 -m venv $(VENV_NAME)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Installation complete!"

dataset: $(VENV_NAME)
	@echo "Executing data preprocessing notebook..."
	$(PYTHON) -m nbconvert --to notebook --execute src/data_preprocessing.ipynb
	@echo "Data preprocessing complete!"

model: $(VENV_NAME)
	@echo "Executing MobileNetV2 model notebook..."
	$(PYTHON) -m nbconvert --to notebook --execute models/MobileNetV2.ipynb
	@echo "Model training complete!"

run: $(VENV_NAME)
	@echo "Starting Streamlit dashboard..."
	$(PYTHON) -m streamlit run src/dashboard.py

clean:
	@echo "Cleaning cache and generated files..."
	rm -rf src/__pycache__
	rm -rf models/__pycache__
	rm -rf */__pycache__
	
fclean: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_NAME)
	rm -f models/model_classification.h5
	rm -f data/dataset.pkl
	@echo "Full clean complete!"


$(VENV_NAME):
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Virtual environment not found. Please run 'make install' first."; \
		exit 1; \
	fi