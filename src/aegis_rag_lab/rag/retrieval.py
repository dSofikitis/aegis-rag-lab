from aegis_rag_lab.rag.embeddings import Embedder
from aegis_rag_lab.rag.models import DocumentChunk
from aegis_rag_lab.rag.vector_store import VectorStore


def retrieve_documents(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    k: int,
    min_similarity: float = 0.0,
) -> list[tuple[float, DocumentChunk]]:
    query_embedding = embedder.embed_query(question)
    return store.similarity_search(query_embedding, k, min_similarity)
