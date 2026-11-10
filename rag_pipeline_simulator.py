import logging
from typing import List, Dict, Any

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("RAGPipeline")

class SimpleRAGPipeline:
    def __init__(self) -> None:
        self.knowledge_base: List[Dict[str, str]] = []
        logger.info("SimpleRAGPipeline initialized.")

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Ingests documents into the local knowledge base.
        In production, these are chunked and converted into vector embeddings.
        """
        self.knowledge_base.extend(documents)
        logger.info(f"Successfully ingested {len(documents)} documents into knowledge base.")

    def retrieve_relevant_context(self, query: str, top_k: int = 2) -> List[str]:
        """
        Simulates semantic retrieval by matching keywords.
        (In advanced production pipelines, this uses Vector Databases like Pinecone/Chroma with Cosine Similarity).
        """
        logger.info(f"Searching knowledge base for query: '{query}'")
        query_tokens = set(query.lower().split())
        
        scored_docs = []
        for doc in self.knowledge_base:
            content = doc.get("content", "").lower()
            # Simple keyword intersection score simulation
            score = sum(1 for token in query_tokens if token in content)
            if score > 0:
                scored_docs.append((score, doc["content"]))

        # Sort by relevance score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k results
        retrieved = [doc[1] for doc in scored_docs[:top_k]]
        logger.info(f"Retrieved {len(retrieved)} relevant context chunks.")
        return retrieved

    def generate_augmented_response(self, query: str) -> str:
        """
        Simulates the Generation step of RAG: Combining retrieved context with the prompt.
        """
        context_chunks = self.retrieve_relevant_context(query)
        
        if not context_chunks:
            return "LLM Response: I couldn't find relevant information in the knowledge base."

        # Constructing the augmented prompt context
        combined_context = "\n---\n".join(context_chunks)
        
        prompt = f"""
        [Context Provided]:
        {combined_context}
        
        [User Query]: {query}
        [Instructions]: Answer the user query strictly using the provided context above.
        """
        
        # Simulating LLM generation output based on context
        logger.info("Generating final response from LLM using augmented context...")
        if "python" in query.lower():
            return f"LLM Answer: Python is heavily utilized for data pipelines, automation, and backend engineering based on our records."
        elif "sql" in query.lower():
            return f"LLM Answer: Advanced SQL involves window functions like DENSE_RANK and performance tuning."
        else:
            return f"LLM Answer based on internal database: {combined_context}"

if __name__ == "__main__":
    # Initialize RAG Pipeline
    rag = SimpleRAGPipeline()

    # Step 1: Ingest mock enterprise knowledge base documents
    company_docs = [
        {"id": "doc1", "content": "Python backend services use asyncio for high-performance concurrent processing and API fetching."},
        {"id": "doc2", "content": "Advanced SQL optimization includes utilizing DENSE_RANK, ROW_NUMBER, and index tuning to speed up queries."},
        {"id": "doc3", "content": "The Handshake AI Skills Studio challenge requires submitting a countdown lock screen project by November 23, 2026."}
    ]
    rag.add_documents(company_docs)

    # Step 2: Test Query execution through RAG
    print("\n" + "="*50)
    user_query = "How do we handle high-performance concurrency in Python?"
    print(f"Query: {user_query}")
    print("="*50)
    response = rag.generate_augmented_response(user_query)
    print(response)

    print("\n" + "="*50)
    user_query_2 = "What are the rules of the Handshake AI challenge?"
    print(f"Query: {user_query_2}")
    print("="*50)
    response_2 = rag.generate_augmented_response(user_query_2)
    print(response_2)