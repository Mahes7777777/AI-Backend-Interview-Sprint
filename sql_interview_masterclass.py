import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("SQLMasterclass")

def run_sql_interview_masterclass():
    # Setup In-Memory Database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 1. Create Departments Table
    cursor.execute("""
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT
        )
    """)
    
    # 2. Create Employees Table
    cursor.execute("""
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT,
            dept_id INTEGER,
            salary REAL,
            hire_date TEXT,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        )
    """)

    # Seed Data - Departments
    cursor.executemany("INSERT INTO departments VALUES (?, ?)", [
        (1, 'Tech'), 
        (2, 'Sales'), 
        (3, 'Marketing')
    ])

    # Seed Data - Employees (Intentional duplicate salaries to test RANK vs DENSE_RANK)
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", [
        (101, 'Alice', 1, 90000, '2024-01-15'),
        (102, 'Bob', 1, 120000, '2023-05-10'),
        (103, 'Charlie', 1, 120000, '2025-02-20'), # Tie salary with Bob
        (104, 'David', 2, 85000, '2022-11-01'),
        (105, 'Eve', 2, 95000, '2023-08-14'),
        (106, 'Frank', 3, 70000, '2024-06-01')
    ])
    conn.commit()
    logger.info("Database and tables populated successfully.")

    # --- TOPIC 1: JOINS & SUBQUERIES ---
    print("\n" + "="*60)
    print("1. JOINS & SUBQUERIES (Employees earning more than their Dept Average)")
    print("="*60)
    
    # Subquery inside WHERE combined with INNER JOIN
    sub_join_query = """
        SELECT e.name, d.dept_name, e.salary
        FROM employees e
        JOIN departments d ON e.dept_id = d.dept_id
        WHERE e.salary > (
            SELECT AVG(sub.salary) 
            FROM employees sub 
            WHERE sub.dept_id = e.dept_id
        );
    """
    cursor.execute(sub_join_query)
    for row in cursor.fetchall():
        print(f"Employee: {row[0]:<10} | Dept: {row[1]:<10} | Salary: ${row[2]:,.2f}")


    # --- TOPIC 2: ROW_NUMBER vs RANK vs DENSE_RANK ---
    print("\n" + "="*60)
    print("2. WINDOW FUNCTIONS: ROW_NUMBER, RANK, DENSE_RANK")
    print("="*60)
    
    window_query = """
        SELECT 
            name,
            salary,
            ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num,
            RANK() OVER (ORDER BY salary DESC) as rank_val,
            DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank_val
        FROM employees;
    """
    cursor.execute(window_query)
    print(f"{'Name':<10} | {'Salary':<8} | {'ROW_NUM':<7} | {'RANK':<6} | {'DENSE_RANK'}")
    print("-" * 55)
    for row in cursor.fetchall():
        print(f"{row[0]:<10} | {row[1]:<8.0f} | {row[2]:<7} | {row[3]:<6} | {row[4]}")


    # --- TOPIC 3: LAG and LEAD ---
    print("\n" + "="*60)
    print("3. ANALYTIC FUNCTIONS: LAG and LEAD (Comparing Salaries)")
    print("="*60)
    
    lag_lead_query = """
        SELECT 
            name,
            salary,
            LAG(salary, 1, 0) OVER (ORDER BY salary DESC) as higher_salary_above,
            LEAD(salary, 1, 0) OVER (ORDER BY salary DESC) as lower_salary_below
        FROM employees;
    """
    cursor.execute(lag_lead_query)
    print(f"{'Name':<10} | {'Salary':<8} | {'Salary Above':<12} | {'Salary Below'}")
    print("-" * 55)
    for row in cursor.fetchall():
        print(f"{row[0]:<10} | {row[1]:<8.0f} | ${row[2]:<11,.0f} | ${row[3]:<10,.0f}")

    conn.close()

if __name__ == "__main__":
    run_sql_interview_masterclass()