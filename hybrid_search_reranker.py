import math
import logging
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("HybridSearchReranker")

class HybridSearchEngine:
    """
    Combines Keyword-based (Sparse) search with Vector (Dense) search,
    followed by Re-ranking using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents
        logger.info(f"HybridSearchEngine initialized with {len(documents)} documents.")

    def _keyword_search(self, query: str) -> List[Tuple[int, float]]:
        """Simulates BM25 / Keyword overlap scoring."""
        query_terms = set(query.lower().split())
        scores = []
        for idx, doc in enumerate(self.documents):
            doc_terms = doc.lower().split()
            matches = sum(1 for term in query_terms if term in doc_terms)
            score = matches / max(len(query_terms), 1)
            scores.append((idx, score))
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _vector_search(self, query: str) -> List[Tuple[int, float]]:
        """Simulates semantic vector similarity search."""
        query_len = len(query)
        scores = []
        for idx, doc in enumerate(self.documents):
            # Mock similarity based on character length proximity for simulation
            diff = abs(len(doc) - query_len)
            sim_score = 1.0 / (1.0 + (diff / 50.0))
            scores.append((idx, sim_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def reciprocal_rank_fusion(self, keyword_results: List[Tuple[int, float]], 
                               vector_results: List[Tuple[int, float]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Applies Reciprocal Rank Fusion (RRF) to merge sparse and dense rankings.
        RRF Score = sum( 1 / (k + rank_i) )
        """
        rrf_scores: Dict[int, float] = {}

        # Process keyword ranks
        for rank, (doc_idx, _) in enumerate(keyword_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        # Process vector ranks
        for rank, (doc_idx, _) in enumerate(vector_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        # Sort combined results by highest RRF score
        sorted_fusion = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fusion

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        logger.info(f"Executing hybrid search for query: '{query}'")
        
        kw_res = self._keyword_search(query)
        vec_res = self._vector_search(query)
        
        fused_rankings = self.reciprocal_rank_fusion(kw_res, vec_res)
        
        final_results = []
        for doc_idx, score in fused_rankings[:top_k]:
            final_results.append({
                "document_id": doc_idx,
                "text": self.documents[doc_idx],
                "rrf_score": round(score, 4)
            })
        return final_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID SEARCH & RE-RANKING SIMULATOR")
    print("="*60 + "\n")

    corpus = [
        "PostgreSQL window functions like ROW_NUMBER and DENSE_RANK optimize heavy analytics queries.",
        "Python asyncio enables concurrent non-blocking network operations and fast data pipelines.",
        "FastAPI framework provides automatic Swagger docs, dependency injection, and Pydantic validation.",
        "Hybrid search combines sparse BM25 keyword matching with dense vector semantic embeddings."
    ]

    engine = HybridSearchEngine(corpus)
    query_text = "How does FastAPI handle data validation and docs?"
    
    results = engine.search(query_text, top_k=2)
    print(f"Query: '{query_text}'\n")
    for res in results:
        print(f"[{res['rrf_score']}] Doc #{res['document_id']}: {res['text']}")