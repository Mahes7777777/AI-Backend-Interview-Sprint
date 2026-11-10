import time
import logging
from collections import OrderedDict
from typing import Any

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("SystemDesign")

class LRUCache:
    """
    Least Recently Used (LRU) Cache implementation.
    Essential in AI systems to cache expensive vector search or LLM generation outputs.
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache = OrderedDict()
        logger.info(f"LRUCache initialized with capacity: {capacity}")

    def get(self, key: str) -> Any:
        if key not in self.cache:
            logger.info(f"Cache Miss for key: '{key}'")
            return None
        # Move accessed key to end to mark as recently used
        self.cache.move_to_end(key)
        logger.info(f"Cache Hit for key: '{key}'")
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Evict oldest item if capacity exceeded
        if len(self.cache) > self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.info(f"Cache capacity reached. Evicted oldest key: '{evicted_key}'")


class TokenBucketRateLimiter:
    """
    Token Bucket algorithm for API rate limiting.
    Prevents abuse and manages concurrent LLM token consumption limits.
    """
    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate_per_sec
        self.last_refill_time = time.time()
        logger.info(f"RateLimiter initialized: Capacity={capacity}, RefillRate={refill_rate_per_sec}/s")

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill_time
        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill_time = now

    def allow_request(self, tokens_needed: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            logger.info(f"Request allowed. Remaining tokens: {self.tokens:.2f}")
            return True
        logger.warning("Rate limit exceeded! Request rejected.")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SYSTEM DESIGN: CACHING & RATE LIMITING SIMULATOR")
    print("="*60 + "\n")

    # 1. Testing LRU Cache
    print("--- Testing LRU Cache ---")
    cache = LRUCache(capacity=2)
    cache.put("query_1", "Result_Vector_A")
    cache.put("query_2", "Result_Vector_B")
    
    # Access query_1 so it becomes recently used
    cache.get("query_1") 
    
    # Adding a 3rd item should evict query_2 (since query_1 was accessed recently)
    cache.put("query_3", "Result_Vector_C")
    
    print(f"Cache state check -> query_2 present: {cache.get('query_2')}") # Should be None (evicted)
    print(f"Cache state check -> query_1 present: {cache.get('query_1')}") # Should be found

    print("\n" + "-"*60 + "\n")

    # 2. Testing Rate Limiter
    print("--- Testing Token Bucket Rate Limiter ---")
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=1.0)
    
    # Send rapid requests
    for i in range(4):
        print(f"Attempt {i+1}: ", end="")
        limiter.allow_request()
        time.sleep(0.1) # Short delay