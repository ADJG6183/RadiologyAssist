"""
Seed demo data into RadiologyAI database.
Reads .env for connection details, then executes seed_demo.sql.
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def get_connection_string():
    """Build MS SQL Server connection string from .env"""
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "1433")
    user = os.getenv("DATABASE_USER", "sa")
    password = os.getenv("DATABASE_PASSWORD")
    database = os.getenv("DATABASE_NAME", "RadiologyAI")
    
    # Format: mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
    connection_string = (
        f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
    )
    return connection_string

def main():
    """Connect and execute seed_demo.sql"""
    try:
        print("Loading .env configuration...")
        connection_string = get_connection_string()
        
        print("Connecting to MS SQL Server...")
        engine = create_engine(connection_string)
        
        with engine.connect() as connection:
            print("✓ Connected successfully!")
            
            # Read seed_demo.sql
            script_dir = os.path.dirname(os.path.abspath(__file__))
            seed_file = os.path.join(script_dir, "seed_demo.sql")
            
            print(f"📖 Reading {seed_file}...")
            with open(seed_file, "r") as f:
                sql_script = f.read()
            
            print("🌱 Executing seed_demo.sql...")
            # Execute script (split by GO for SQL Server batches)
            for batch in sql_script.split("GO"):
                if batch.strip():
                    connection.execute(text(batch))
            
            connection.commit()
            print("✓ Seed data loaded successfully!")
            
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
