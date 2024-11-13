FROM python:3.12.6-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app
COPY requirements.txt .
COPY ./src .

RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding, MultiTaskTextEmbedding; \
    SparseTextEmbedding('Qdrant/bm42-all-minilm-l6-v2-attentions'); \
    LateInteractionTextEmbedding('colbert-ir/colbertv2.0'); \
    MultiTaskTextEmbedding(model_name='jinaai/jina-embeddings-v3')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
