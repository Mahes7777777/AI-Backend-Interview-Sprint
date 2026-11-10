import logging
import queue
import time
from typing import Dict, Any

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("DatabaseScaling")

class MockDatabaseConnection:
    """Simulates an individual database connection instance."""
    def __init__(self, conn_id: int) -> None:
        self.conn_id = conn_id
        logger.info(f"Database Connection #{self.conn_id} established.")

    def execute_query(self, query: str) -> str:
        # Simulate small query execution delay
        time.sleep(0.05)
        return f"Result for '{query}' executed on Connection #{self.conn_id}"


class ConnectionPool:
    """
    Implements a Connection Pool to manage and reuse database connections efficiently.
    Prevents connection exhaustion under high concurrent load.
    """
    def __init__(self, max_connections: int = 3) -> None:
        self.max_connections = max_connections
        self.pool: queue.Queue = queue.Queue(maxsize=max_connections)
        
        # Pre-initialize connections
        for i in range(1, max_connections + 1):
            conn = MockDatabaseConnection(conn_id=i)
            self.pool.put(conn)
            
        logger.info(f"ConnectionPool initialized with max capacity: {max_connections}")

    def acquire_connection(self) -> MockDatabaseConnection:
        """Acquires a connection from the pool (blocks if none available)."""
        logger.info("Acquiring database connection from pool...")
        conn = self.pool.get(block=True, timeout=2.0)
        return conn

    def release_connection(self, conn: MockDatabaseConnection) -> None:
        """Returns the connection back to the pool for reuse."""
        self.pool.put(conn)
        logger.info(f"Connection #{conn.conn_id} returned back to pool.")


class DatabaseShardingRouter:
    """
    Simulates Horizontal Database Sharding using Consistent Hashing / Modulo Strategy.
    Routes queries to different database nodes based on User ID.
    """
    def __init__(self, num_shards: int = 2) -> None:
        self.num_shards = num_shards
        self.shards: Dict[int, str] = {
            i: f"postgresql://cluster_node_{i}.internal:5432/db_shard_{i}" 
            for i in range(num_shards)
        }
        logger.info(f"DatabaseShardingRouter initialized with {num_shards} nodes.")

    def get_shard_node(self, user_id: int) -> str:
        """Determines which shard node owns the data using a deterministic modulo function."""
        shard_id = user_id % self.num_shards
        node_url = self.shards[shard_id]
        logger.info(f"Routing User ID {user_id} -> Shard Node #{shard_id} ({node_url})")
        return node_url


if __name__ == "__main__":
    print("\n" + "="*65)
    print("DATABASE SCALING: CONNECTION POOLING & SHARDING SIMULATOR")
    print("="*65 + "\n")

    # 1. Testing Connection Pooling
    print("--- Testing Connection Pool ---")
    db_pool = ConnectionPool(max_connections=2)
    
    # Acquire and release connections simulating concurrent queries
    conn1 = db_pool.acquire_connection()
    print(conn1.execute_query("SELECT * FROM users;"))
    
    conn2 = db_pool.acquire_connection()
    print(conn2.execute_query("SELECT * FROM transactions;"))
    
    # Release them back
    db_pool.release_connection(conn1)
    db_pool.release_connection(conn2)

    print("\n" + "-"*65 + "\n")

    # 2. Testing Database Sharding Router
    print("--- Testing Database Sharding ---")
    router = DatabaseShardingRouter(num_shards=3)
    
    sample_user_ids = [1001, 1002, 1003, 1004, 1005]
    for uid in sample_user_ids:
        router.get_shard_node(uid)