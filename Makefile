.PHONY: run clean test lint

run:
    @echo "Avvio pipeline..."
    @python scripts/run_pipeline.py

clean:
    @echo "Pulizia ambiente..."
    @rm -rf venv
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type f -name "*.pyc" -delete

test:
    @echo "Esecuzione test..."
    @pytest tests/ -v

lint:
    @echo "Verifica codice..."
    @flake8 src/ tests/
    @mypy src/ tests/