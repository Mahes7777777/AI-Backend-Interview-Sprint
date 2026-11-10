import asyncio
import time
import logging
from typing import List, Dict, Any

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AsyncPipeline")

async def fetch_mock_external_api(user_id: int) -> Dict[str, Any]:
    """
    Simulates an asynchronous network call (e.g., fetching user data from an external microservice or LLM API).
    Non-blocking `asyncio.sleep` represents network latency.
    """
    logger.info(f"Starting API fetch for User ID: {user_id}")
    
    # Simulate network delay of 1 second
    await asyncio.sleep(1.0) 
    
    mock_data = {
        "user_id": user_id,
        "score": user_id * 12.5,
        "status": "SUCCESS"
    }
    logger.info(f"Completed API fetch for User ID: {user_id}")
    return mock_data

async def process_batch_concurrently(user_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Processes multiple tasks concurrently using asyncio.gather().
    This is critical for high-performance backend pipelines.
    """
    logger.info(f"Spawning concurrent tasks for {len(user_ids)} users...")
    
    # Create a list of coroutines
    tasks = [fetch_mock_external_api(uid) for uid in user_ids]
    
    # Execute all tasks concurrently and wait for results
    results = await asyncio.gather(*tasks)
    
    logger.info("All concurrent tasks finished successfully.")
    return results

if __name__ == "__main__":
    target_user_ids = [101, 102, 103, 104, 105]
    
    start_time = time.time()
    
    # Run the async event loop
    loop = asyncio.get_event_loop()
    final_results = loop.run_until_complete(process_batch_concurrently(target_user_ids))
    
    end_time = time.time()
    
    print("\n--- Async Execution Summary ---")
    for res in final_results:
        print(res)
    print(f"\nTotal Execution Time (Async Concurrency): {end_time - start_time:.2f} seconds")