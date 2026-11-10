import math
import logging
from typing import List, Dict, Any, Tuple

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("LangChainAgentSim")

class MockVectorStore:
    """
    Simulates a Vector Database (like Chroma, FAISS, or Pinecone).
    Stores text chunks and their mock embedding vectors, performing Cosine Similarity search.
    """
    def __init__(self) -> None:
        self.store: List[Dict[str, Any]] = []
        logger.info("MockVectorStore initialized.")

    def _get_mock_embedding(self, text: str) -> List[float]:
        """Simulates embedding generation by mapping keyword lengths to a vector."""
        # Simple deterministic vector representation for simulation
        length = len(text)
        v1 = float(length % 10) / 10.0
        v2 = float(text.count('python') * 2 + 1) / 5.0
        v3 = float(text.count('sql') * 2 + 1) / 5.0
        return [v1, v2, v3]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
        mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot_product / (mag_a * mag_b)

    def add_documents(self, documents: List[str]) -> None:
        """Embeds and stores documents."""
        for doc in documents:
            embedding = self._get_mock_embedding(doc)
            self.store.append({"text": doc, "embedding": embedding})
        logger.info(f"Added {len(documents)} documents to Vector Store.")

    def similarity_search(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """Performs semantic similarity search against stored documents."""
        logger.info(f"Executing vector similarity search for query: '{query}'")
        query_embedding = self._get_mock_embedding(query)
        
        scored_results = []
        for item in self.store:
            similarity = self._cosine_similarity(query_embedding, item["embedding"])
            scored_results.append((item["text"], similarity))
            
        # Sort by highest similarity score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]


class SimpleAgentOrchestrator:
    """
    Simulates LangChain / LangGraph chain execution and state routing.
    """
    def __init__(self, vector_store: MockVectorStore) -> None:
        self.vector_store = vector_store
        logger.info("SimpleAgentOrchestrator initialized.")

    def run_agent_chain(self, user_query: str) -> str:
        logger.info("Agent Chain started: Step 1 - Retrieving context from Vector Store...")
        
        # Step 1: Retrieve relevant context
        retrieved_docs = self.vector_store.similarity_search(user_query, top_k=1)
        context = retrieved_docs[0][0] if retrieved_docs else "No context found."
        score = retrieved_docs[0][1] if retrieved_docs else 0.0

        logger.info(f"Agent Chain: Step 2 - Synthesizing final response (Relevance Score: {score:.2f})...")
        
        # Step 2: Formulating chain response
        response = f"""
        [Agent Execution Summary]
        - User Query: {user_query}
        - Retrieved Context: '{context}'
        - Match Confidence: {score:.2f}
        - Final Decision: Processed securely via structured LangChain-style pipeline.
        """
        return response


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LANGCHAIN & VECTOR DATABASE PIPELINE SIMULATION")
    print("="*60 + "\n")

    # 1. Setup Vector Store
    vector_db = MockVectorStore()
    
    # Ingest technical knowledge documents
    kb_docs = [
        "Python backend architectures leverage asyncio and generators for memory efficiency.",
        "Advanced SQL databases use window functions like DENSE_RANK and indexing for query performance.",
        "LangChain and LangGraph manage multi-agent state loops and LLM orchestration."
    ]
    vector_db.add_documents(kb_docs)

    # 2. Initialize Agent Orchestrator
    agent = SimpleAgentOrchestrator(vector_db)

    # 3. Run Query
    query = "How does LangChain manage agent state?"
    print(f"Query: {query}\n")
    
    final_output = agent.run_agent_chain(query)
    print(final_output)