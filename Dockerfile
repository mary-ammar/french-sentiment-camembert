FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit transformers \
        scikit-learn sentencepiece protobuf numpy && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY src/ src/

# Model weights are not committed to git (too large).
# Run `make train` before `docker build` to generate them.
COPY results/models/ results/models/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
