.PHONY: preprocess train app docker-build docker-run test lint

preprocess:
	python scripts/preprocess.py

train:
	python scripts/train.py

app:
	streamlit run app.py

docker-build:
	docker build -t french-sentiment-camembert .

docker-run:
	docker run -p 8501:8501 french-sentiment-camembert

test:
	pytest tests/ -v

lint:
	ruff check .
