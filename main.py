from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import re
from fastapi import FastAPI, HTTPException
from typing import Any, Optional, List, Dict
from pydantic import BaseModel
import uvicorn 

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333, timeout=35)
COLLECTION_NAME='arxiv_papers'
VECTOR_SIZE = 1536
vectors_config = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
EMBEDDING_MODEL = "text-embedding-ada-002"

tiny_api_key = os.getenv ("TINY_API_KEY")
if not tiny_api_key:
    raise RuntimeError ("TINY_API_KEY is not set.")
tiny_base_url = os.getenv ("TINY_BASE_URL") or "https://litellm.aks-hs-prod.int.hyperskill.org/openai/"
client_openAI = OpenAI(
    api_key=tiny_api_key,
    base_url=tiny_base_url
)

app = FastAPI()

# Pydantic classes for request and response models in FastAPI
class SearchRequest(BaseModel):
    """
    Model representing a search request sent by a client.
    Attributes:
        query (str): The search query string provided by the client.
        top_n (Optional[int]): The number of top search results to return. Defaults to 5 if not provided.
    """
    query: str
    top_n: Optional[int] = 5

class SearchResult(BaseModel):
    """
    Model representing an individual search result.
    Attributes:
        id (str): Unique identifier for the search result item.
        payload (Dict): A dictionary containing additional data or metadata about the item.
        score (float): The relevance score of the search result, typically used for ranking.
    """
    id: str
    payload: Dict
    score: float

class SearchResponse(BaseModel):
    """
    Model representing the response returned to a client after a search.
    Attributes:
        results (List[SearchResult]): A list containing the top matching search results.
    """
    results: List[SearchResult]

# Define the /search POST endpoint
@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    # Perform search using request.query and request.top_n
    if not request.query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if request.top_n is None or request.top_n <= 0:
        raise HTTPException(status_code=400, detail="top_n must be a positive integer")

    results: List[SearchResult] = []
    try:
        query_vector = get_embedding(request.query)
        raw_results = similar_entries_author(
            COLLECTION_NAME,
            request.query,
            query_vector,
            request.top_n,
            )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results: List[SearchResult] = []

    for item in raw_results[:request.top_n]:
        # Less than top_n results if Qdrant has fewer matching points
        payload = item.payload or {}
        result_id = str(payload.get("id", item.id))
        score = getattr(item, "score", 1.0)
        if score is None:
            score = 1.0

        results.append (
            SearchResult (
                id=result_id,
                payload=payload,
                score=float (score),
            )
        )

    return SearchResponse(results=results)


# Vector functions
def get_embedding(qry: str) -> list[float]:
    """Remove newlines from a query and create embeddings to return a list of floats.
    input: query string with newlines
    result: list of floats
    """
    # Remove newlines from a query
    cleaned_qry = qry.replace("\n", "")
    # Make a call to the OpenAI API with preprocessed query
    response = client_openAI.embeddings.create(
        input=cleaned_qry,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def find_author(qry_author: str) -> str | None:
    """ Find author
    input: query which normally contains 'by' followed by author name
    result: author name or None
    """
    if 'by' in qry_author:
        match = re.search (r'by\s+([A-Za-z\s\-]+)', qry_author)
        if match:
            author_name: str | Any = match.group (1)
            return author_name
    return None

def similar_entries_author(collection_name: str, qry: str, query_vector: list[float], top_k: int = 3):
    """
    Find the similar entry for an author or the top-k most similar entries to the generated embedding
    input: for the collection_name and author we provide the vector result of get_embedding() to search k entries
    result: top-k most similar points that match the author or query vector list
    """
    if top_k <= 0:
        raise ValueError ("top_k must be a positive integer")
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError (f"query_vector must have same dimension as Qdrant vector of length {VECTOR_SIZE}")
    author_found = find_author(qry)
    if author_found:
        found_author_papers, _next_page_offset = client.scroll (
            collection_name=collection_name,
            scroll_filter=models.Filter (
                must=[
                    models.FieldCondition (
                        key="authors",
                        match=models.MatchText(text=author_found),
                    ),
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return found_author_papers

    else:
        # Normal vector search - After NOT finding author
        vector_search_result = similar_entries(collection_name, query_vector, top_k)
        if vector_search_result:
            return vector_search_result
        else:
            raise ValueError(f"No results found for {author_found} in the database.")

def similar_entries(collection_name: str, query_vector: list[float], top_k: int = 5):
    """
    Find the top-k most similar entries to the generated embedding
    input: for the collection_name we provide the vector result of get_embedding() to search k entries
    result: top-k most similar points that match the query vector list
    """
    if top_k <= 0:
        raise ValueError ("top_k must be a positive integer")
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError (f"query_vector must have same dimension as Qdrant vector of length {VECTOR_SIZE}")
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return search_results.points


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000)
    QUERY = "Mentions of point clouds by Tian-Xing Xu"
    embeddings = get_embedding(QUERY)
    result_entries = similar_entries_author(COLLECTION_NAME, QUERY, embeddings, 3)

    paper_ids = [result.payload.get("id", None)
                 if result.payload else None
                 for result in result_entries
                ]
    print(paper_ids)
