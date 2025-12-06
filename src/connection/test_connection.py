import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()
    
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✅ Connected to PostgreSQL!")
    print(f"Version: {version[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM predictions;")
    count = cursor.fetchone()[0]
    print(f"📊 Current predictions in database: {count}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")