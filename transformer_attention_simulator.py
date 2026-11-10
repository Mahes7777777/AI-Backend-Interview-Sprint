import numpy as np
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("TransformerAttention")

def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    Simulates the core Self-Attention mechanism formula used in Transformers:
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    """
    logger.info("Calculating Scaled Dot-Product Attention...")
    
    # Get dimension of keys for scaling
    d_k = query.shape[-1]
    
    # Step 1: Matrix multiplication of Query and Transposed Key
    matmul = np.dot(query, key.T)
    
    # Step 2: Scale by square root of d_k to prevent gradients from exploding
    scaled_scores = matmul / np.sqrt(d_k)
    
    # Step 3: Apply Softmax to get attention weights (probabilities summing to 1)
    weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=-1, keepdims=True)
    
    logger.info(f"Attention Weights Matrix:\n{weights}")
    
    # Step 4: Multiply by Value matrix to get final attention-weighted representation
    output = np.dot(weights, value)
    return output

if __name__ == "__main__":
    # Mock embedding vectors for words in a sentence (e.g., Sequence length = 3, Embedding dimension = 4)
    np.random.seed(42)
    
    # Query (Q), Key (K), and Value (V) matrices derived from input tokens
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 4)

    print("\n" + "="*50)
    print("TRANSFORMER SELF-ATTENTION SIMULATION")
    print("="*50)
    
    attention_output = scaled_dot_product_attention(Q, K, V)
    
    print("\nFinal Attention Output Shape:", attention_output.shape)
    print("Output Representation:\n", attention_output)