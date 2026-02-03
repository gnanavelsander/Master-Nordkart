import pandas as pd
import logging
import psycopg2

logger = logging.getLogger(__name__)




class Database:
    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self.conn = None
    
    def connect(self) -> "Database":
        try:
            logger.info("Connecting to database...")
            self.conn = psycopg2.connect(self.conn_str)
            logger.info("Database connected successfully")
            return self
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("Database disconnected")
    
    def query(self, sql: str) -> pd.DataFrame:
        try:
            logger.info(f"Executing query: {sql[:100]}...")
            with self.conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
            logger.info(f"Query returned {len(rows)} rows")
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise