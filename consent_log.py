import hashlib
import logging
import os
import re
import json
from datetime import datetime, timezone
from psycopg2 import pool

logger = logging.getLogger(__name__)

HASH_SALT = os.getenv("HASH_SALT", "agri_intellect_default_salt_2026")
DATABASE_URL = os.getenv("DATABASE_URL")


def normalize_phone(phone: str) -> str:
    """Normalizes phone numbers to standard E.164 format before hashing."""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"+91{digits}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    return f"+{digits}"


def hash_phone_number(phone: str) -> str:
    """Creates an irreversible hash for DPDPA compliance."""
    clean_phone = normalize_phone(phone)
    salted_string = f"{clean_phone}:{HASH_SALT}"
    return hashlib.sha256(salted_string.encode("utf-8")).hexdigest()


class ConsentLogger:
    def __init__(self):
        self.db_url = DATABASE_URL
        self.connection_pool = None
        
        if self.db_url:
            try:
                self.connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.db_url
                )
            except Exception as e:
                logger.error(f"Failed to create connection pool: {e}")
                
        self._init_db()

    def _init_db(self):
        if not self.connection_pool:
            logger.warning("No connection pool available. Skipping DB table creation.")
            return

        create_table_query = """
        CREATE TABLE IF NOT EXISTS consent_audit_logs (
            id SERIAL PRIMARY KEY,
            hashed_phone VARCHAR(64) NOT NULL,
            consent_type VARCHAR(50) NOT NULL,
            granted BOOLEAN NOT NULL,
            language VARCHAR(10) DEFAULT 'mr',
            timestamp TIMESTAMPTZ NOT NULL,
            context_meta JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_hashed_phone ON consent_audit_logs(hashed_phone);
        """
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
            conn.commit()
            logger.info("Consent audit log database table initialized via pool.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize consent log table: {e}")
        finally:
            self.connection_pool.putconn(conn)

    def log_consent(
        self,
        phone: str,
        consent_type: str,
        granted: bool,
        language: str = "mr",
        metadata: dict = None
    ) -> bool:
        if not self.connection_pool:
            logger.error("Database connection pool is not initialized.")
            return False

        hashed_phone = hash_phone_number(phone)
        current_time = datetime.now(timezone.utc)

        query = """
        INSERT INTO consent_audit_logs 
        (hashed_phone, consent_type, granted, language, timestamp, context_meta)
        VALUES (%s, %s, %s, %s, %s, %s);
        """

        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        hashed_phone,
                        consent_type,
                        granted,
                        language,
                        current_time,
                        json.dumps(metadata or {})
                    )
                )
            conn.commit()
            logger.info(f"Consent logged for type '{consent_type}' (Granted: {granted})")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error writing consent log: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def close(self):
        """Call this on app shutdown to prevent connection leaks."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Consent log connection pool closed.")