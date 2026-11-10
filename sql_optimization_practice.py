import sqlite3
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("SQLOptimizer")

def setup_in_memory_db() -> sqlite3.Connection:
    """Creates an in-memory SQLite database and populates dummy enterprise sales data."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create Sales Table
    cursor.execute("""
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            employee_name TEXT,
            department TEXT,
            amount REAL,
            sale_date TEXT
        )
    """)

    # Seed Data
    sales_data = [
        (1, 'Alice', 'Tech', 12000.00, '2026-10-01'),
        (2, 'Bob', 'Tech', 15000.00, '2026-10-02'),
        (3, 'Charlie', 'Tech', 9000.00, '2026-10-03'),
        (4, 'David', 'Sales', 22000.00, '2026-10-01'),
        (5, 'Eve', 'Sales', 22000.00, '2026-10-02'),
        (6, 'Frank', 'Sales', 18000.00, '2026-10-03'),
        (7, 'Grace', 'Marketing', 11000.00, '2026-10-01'),
    ]
    cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sales_data)
    conn.commit()
    logger.info("Database initialized with sales records.")
    return conn

def execute_window_function_query(conn: sqlite3.Connection) -> List[Tuple]:
    """
    Industry Query Pattern: Top Performers per Department using DENSE_RANK()
    Matches SQL Windowing required for data engineering/backend interviews.
    """
    cursor = conn.cursor()
    query = """
        WITH RankedSales AS (
            SELECT 
                employee_name,
                department,
                amount,
                DENSE_RANK() OVER (
                    PARTITION BY department 
                    ORDER BY amount DESC
                ) as rank_in_dept
            FROM sales
        )
        SELECT department, employee_name, amount, rank_in_dept
        FROM RankedSales
        WHERE rank_in_dept <= 2;
    """
    cursor.execute(query)
    return cursor.fetchall()

if __name__ == "__main__":
    db_conn = setup_in_memory_db()
    top_performers = execute_window_function_query(db_conn)

    print("\n--- Top 2 Highest Earners per Department (Window Function Output) ---")
    print(f"{'Department':<12} | {'Employee':<10} | {'Amount ($)':<10} | {'Rank'}")
    print("-" * 50)
    for row in top_performers:
        print(f"{row[0]:<12} | {row[1]:<10} | {row[2]:<10.2f} | {row[3]}")
    
    db_conn.close()