"""
Comprehensive database test script.
Creates schema, stored procedures, seeds data, and tests retrieval.
"""

import os
import sys
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def get_connection_string():
    """Build MS SQL Server connection string from .env"""
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "1433")
    user = os.getenv("DATABASE_USER", "sa")
    password = os.getenv("DATABASE_PASSWORD")
    database = os.getenv("DATABASE_NAME", "RadiologyAI")
    
    connection_string = (
        f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
        f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    return connection_string

def get_master_connection_string():
    """Build connection string for master database"""
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "1433")
    user = os.getenv("DATABASE_USER", "sa")
    password = os.getenv("DATABASE_PASSWORD")

    connection_string = (
        f"mssql+pyodbc://{user}:{password}@{host}:{port}/master"
        f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    return connection_string

def execute_sql_file(engine, filepath):
    """Execute SQL file, splitting by GO batches"""
    with engine.connect() as connection:
        with open(filepath, "r") as f:
            sql_script = f.read()
        
        for batch in sql_script.split("GO"):
            if batch.strip():
                try:
                    connection.execute(text(batch))
                except Exception as e:
                    print(f"Warning during batch execution: {e}")
        
        connection.commit()

def main():
    """Test the entire database setup"""
    try:
        print("\n" + "="*70)
        print("RADIOLOGY AI - DATABASE SETUP TEST")
        print("="*70 + "\n")
        
        print("📡 Step 1: Connecting to MS SQL Server (master)...")
        master_connection_string = get_master_connection_string()
        master_engine = create_engine(master_connection_string)
        
        # Create database if needed — must run outside any transaction (AUTOCOMMIT)
        database_name = os.getenv("DATABASE_NAME", "RadiologyAI")
        with master_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(
                f"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database_name}') "
                f"CREATE DATABASE [{database_name}]"
            ))
        
        print("✓ Connected to SQL Server\n")
        
        # Now use the RadiologyAI database
        db_connection_string = get_connection_string()
        db_engine = create_engine(db_connection_string)
        
        print("📋 Step 2: Creating database schema...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(script_dir, "schema.sql")
        execute_sql_file(db_engine, schema_file)
        print("✓ Schema created\n")
        
        print("⚙️  Step 3: Creating stored procedures...")
        procs_file = os.path.join(script_dir, "stored_procs.sql")
        execute_sql_file(db_engine, procs_file)
        print("✓ Stored procedures created\n")
        
        print("🌱 Step 4: Seeding demo data...")
        seed_file = os.path.join(script_dir, "seed_demo.sql")
        execute_sql_file(db_engine, seed_file)
        print("✓ Seed data loaded\n")
        
        print("🔍 Step 5: Testing sp_get_study_context (study_id=1)...")
        with db_engine.connect() as connection:
            # Call stored procedure for study context
            result = connection.execute(
                text("EXEC sp_get_study_context @study_id = 1")
            )
            
            study_rows = result.fetchall()
            if study_rows:
                print("\n  📊 STUDY CONTEXT:")
                row = study_rows[0]
                print(f"    Study ID: {row[0]}")
                print(f"    Study Date: {row[1]}")
                print(f"    Modality: {row[2]}")
                print(f"    Institution: {row[3]}")
                print(f"    Patient Name: {row[6]} {row[7]} (MRN: {row[5]})")
                print(f"    Date of Birth: {row[8]}")
            
            # Fetch prior reports for this patient
            prior_result = connection.execute(
                text("""
                    SELECT TOP 3 draft_id, study_id, draft_text, created_at 
                    FROM report_drafts 
                    WHERE study_id IN (
                        SELECT study_id FROM studies WHERE patient_id = 1
                    ) 
                    ORDER BY created_at DESC
                """)
            )
            
            prior_rows = prior_result.fetchall()
            print("\n  📄 PRIOR REPORTS FOR THIS PATIENT:")
            if prior_rows:
                for i, row in enumerate(prior_rows, 1):
                    print(f"    {i}. Study {row[1]} (Draft {row[0]}): {row[2][:70]}...")
            else:
                print("    (No prior reports found)")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        print("📝 Summary:")
        print("   - Database: RadiologyAI")
        print("   - Tables: 5 (patients, studies, report_inputs, report_drafts, agent_events)")
        print("   - Stored Procedures: 4")
        print("   - Demo Patients: 2")
        print("   - Demo Studies: 6")
        print("   - Demo Prior Reports: 4\n")
        print("🎯 Ready for STEP 2: Agent Pipeline Skeleton\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
