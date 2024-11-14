import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from qdrant_client import QdrantClient, models
from fastembed import (
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    MultiTaskTextEmbedding,
)
from time import time
from embedding import JinaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize models
# models_cache_dir = "/app/models"
jina = JinaEmbeddings()
# jina = MultiTaskTextEmbedding(
#     model_name="jinaai/jina-embeddings-v3", local_files_only=True
# )
print("Jina Embedding loaded.")
bm42 = SparseTextEmbedding(
    "Qdrant/bm42-all-minilm-l6-v2-attentions", local_files_only=True
)
print("BM42 loaded.")
colbert = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", local_files_only=True)
print("Colbert loaded.")

qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
HYBRID_COLLECTION_NAME = "aaps_hybrid_vector"

app = FastAPI()


# Input model
class QueryRequest(BaseModel):
    query_text: str
    limit: int


# Output model
class Hit(BaseModel):
    payload: dict
    score: float


class QueryResponse(BaseModel):
    hits: List[Hit]


@app.get("/search", response_model=QueryResponse)
async def query_embeddings(
    query_text: str = Query(..., description="The search query text"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
):
    """
    Search endpoint that processes a query and returns results from Qdrant.
    """
    try:
        start = time()
        sparse_embedding = list(bm42.query_embed(query_text))[0]
        print(f"Sparse embedding: {round(time() - start, 2)}s")
        start = time()
        # dense_embedding = list(
        #     jina.task_embed(query_text, task_type="retrieval.query")
        # )[0].embedding
        dense_embedding = jina.encode(query_text, task="retrieval.query").tolist()[0]
        print(f"Dense embedding: {round(time() - start, 2)}s")
        start = time()
        colbert_embedding = list(colbert.query_embed(query_text))[0]
        print(f"Colbert embedding: {round(time() - start, 2)}s")
        start = time()

        hits = qdrant_client.query_points(
            collection_name=HYBRID_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=sparse_embedding.as_object(),
                    using="bm42",
                    limit=max(5, int(limit * 1.5)),
                ),
                models.Prefetch(
                    query=dense_embedding,
                    using="jina_dense",
                    limit=max(5, int(limit * 1.5)),
                ),
            ],
            query=colbert_embedding.tolist(),
            using="jina_colbert",
            limit=limit,
        ).points

        response_hits = [Hit(payload=hit.payload, score=hit.score) for hit in hits]
        print(f"Qdrant search: {round(time() - start, 2)}s")
        return QueryResponse(hits=response_hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
