"""PostgreSQL Database Configuration
==================================
Loads database credentials from environment variables or .env file.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DB_NAME = os.getenv("LOCAL_DB_NAME", "retail_ai_analytics")
DB_USER = os.getenv("LOCAL_DB_USER", "postgres")
DB_PASSWORD = os.getenv("LOCAL_DB_PASSWORD", "123456")
DB_HOST = os.getenv("LOCAL_DB_HOST", "172.17.4.199")
DB_PORT = int(os.getenv("LOCAL_DB_PORT", "5432"))
ENABLE_POSTGRES = os.getenv("ENABLE_POSTGRES", "true").strip().lower() in ("true", "1", "yes")
SHOP_ID = int(os.getenv("SHOP_ID", "1"))


def get_connection_params() -> Dict[str, Any]:
    """Return dictionary of connection keyword arguments for psycopg2.connect."""
    return {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
        "connect_timeout": 5,
    }
