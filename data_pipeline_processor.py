import logging
import csv
from typing import Generator, Dict, Any, List
from datetime import datetime

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PipelineProcessor")

class DataProcessingError(Exception):
    """Custom exception raised for errors during data transformation."""
    pass

class IndustryDataPipeline:
    def __init__(self, batch_size: int = 3) -> None:
        self.batch_size = batch_size
        logger.info("IndustryDataPipeline initialized successfully.")

    def raw_data_stream(self, records: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """
        Memory-efficient generator yielding records one by one.
        Simulates streaming large datasets from a database or file.
        """
        try:
            for record in records:
                yield record
        except Exception as e:
            logger.error(f"Error reading stream: {e}")
            raise DataProcessingError("Stream reading failed.") from e

    def transform_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes and normalizes user records with validation and error isolation.
        """
        processed_output: List[Dict[str, Any]] = []
        
        logger.info("Starting record transformation phase...")
        for record in self.raw_data_stream(records):
            try:
                # Validation check
                if not record.get("id") or not record.get("salary"):
                    logger.warning(f"Skipping invalid record due to missing fields: {record}")
                    continue
                
                # Business Logic Transformation
                transformed = {
                    "user_id": record["id"],
                    "full_name": record["name"].strip().title(),
                    "annual_salary_usd": float(record["salary"]),
                    "processed_timestamp": datetime.utcnow().isoformat(),
                    "status": "ACTIVE" if record.get("is_active", True) else "INACTIVE"
                }
                
                processed_output.append(transformed)
            except Exception as err:
                logger.error(f"Failed to process record {record.get('id')}: {err}")
                continue

        logger.info(f"Successfully processed {len(processed_output)} records.")
        return processed_output

    def export_to_csv(self, records: List[Dict[str, Any]], filename: str = "processed_users.csv") -> None:
        """
        Exports the transformed records to a CSV file securely.
        """
        if not records:
            logger.warning("No records available to export.")
            return

        try:
            keys = records[0].keys()
            with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Successfully exported {len(records)} records to {filename}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise DataProcessingError("CSV export failed.") from e

# --- Execution Block ---
if __name__ == "__main__":
    # Mock database or file records
    mock_database_records = [
        {"id": 101, "name": "maheswar rao", "salary": 75000, "is_active": True},
        {"id": 102, "name": "  alex smith ", "salary": 92000, "is_active": True},
        {"id": 103, "name": "incomplete record"}, # Will trigger warning/skip
        {"id": 104, "name": "sarah Connor", "salary": 115000, "is_active": False}
    ]

    pipeline = IndustryDataPipeline(batch_size=2)
    final_data = pipeline.transform_records(mock_database_records)
    
    # Exporting data to CSV file
    pipeline.export_to_csv(final_data, filename="processed_users.csv")
    
    print("\n--- Final Cleaned Output ---")
    for item in final_data:
        print(item)