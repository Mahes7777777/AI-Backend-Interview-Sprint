import time
import logging
from contextlib import contextmanager
from typing import Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("AdvancedPython")

# --- 1. CUSTOM DECORATOR (Measuring Execution Time) ---
def measure_execution_time(func):
    """
    A custom decorator that logs the execution time of any function.
    Highly asked in backend interviews for performance optimization tracking.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting execution of '{func.__name__}'...")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {elapsed_time:.4f} seconds.")
        return result
    return wrapper


# --- 2. CUSTOM CONTEXT MANAGER (Managing Resources securely) ---
@contextmanager
def managed_database_connection(db_name: str) -> Generator[str, None, None]:
    """
    A custom context manager using generator and @contextmanager.
    Ensures setup and teardown (like opening and closing DB connections) happen safely.
    """
    logger.info(f"Connecting to database: '{db_name}'...")
    connection_status = f"CONNECTED_TO_{db_name.upper()}"
    try:
        # Yield control back to the 'with' block
        yield connection_status
    except Exception as e:
        logger.error(f"Database error occurred: {e}")
        raise
    finally:
        # Guaranteed cleanup phase (Teardown)
        logger.info(f"Closing connection to database: '{db_name}'. Cleanup completed safely.")


# --- 3. ADVANCED OBJECT-ORIENTED PROGRAMMING (OOP) ---
class BasePipeline:
    """Base class demonstrating encapsulation and inheritance."""
    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name

    def execute(self) -> None:
        raise NotImplementedError("Subclasses must implement this abstract method.")

class EnterpriseDataIngestionPipeline(BasePipeline):
    """Derived class implementing polymorphism and robust OOP design."""
    def __init__(self, pipeline_name: str, batch_size: int) -> None:
        super().__init__(pipeline_name)
        self.batch_size = batch_size

    @measure_execution_time
    def execute(self) -> None:
        logger.info(f"Executing pipeline '{self.pipeline_name}' with batch size {self.batch_size}")
        # Simulating data processing workload
        time.sleep(0.5)
        logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMONSTRATING OOP, DECORATORS, & CONTEXT MANAGERS")
    print("="*50 + "\n")

    # 1. Running OOP with Decorator attached
    pipeline = EnterpriseDataIngestionPipeline(pipeline_name="User_Sync_V1", batch_size=100)
    pipeline.execute()

    print("\n" + "-"*50 + "\n")

    # 2. Using Custom Context Manager (`with` statement)
    with managed_database_connection("production_users_db") as db_conn:
        logger.info(f"Active session state: {db_conn}")
        logger.info("Executing safe database queries inside context...")