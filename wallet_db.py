import logging
import os
from datetime import datetime, timezone
from psycopg2 import pool
from typing import Optional, Tuple, Dict, Any
from consent_log import hash_phone_number, normalize_phone  # Reuse your existing hash function

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL")

class WalletDB:
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
                self._init_db()
            except Exception as e:
                logger.error(f"Failed to create WalletDB connection pool: {e}")

    def _init_db(self):
        """Creates the User Credits ledger and the Top-up Audit table."""
        if not self.connection_pool:
            return

        # Using PostgreSQL syntax (SERIAL, TIMESTAMPTZ, JSONB)
        schema_query = """
        -- 1. The actual balance ledger for the user
        CREATE TABLE IF NOT EXISTS user_credits (
            hashed_phone VARCHAR(64) PRIMARY KEY,
            credit_balance INTEGER NOT NULL DEFAULT 0,
            tier_last_purchased VARCHAR(50),
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        -- 2. The immutable audit log for the x402 Challenge
        CREATE TABLE IF NOT EXISTS topup_transactions (
            id SERIAL PRIMARY KEY,
            hashed_phone VARCHAR(64) NOT NULL,
            session_id VARCHAR(100),
            payment_source VARCHAR(50) NOT NULL CHECK (payment_source IN ('WHATSAPP_UPI', 'B2B_DIRECT_X402')),
            
            package_id VARCHAR(50) NOT NULL,
            fiat_amount NUMERIC(10, 2),
            credits_granted INTEGER NOT NULL,
            
            razorpay_payment_id VARCHAR(100) UNIQUE,
            razorpay_order_id VARCHAR(100),
            razorpay_payment_status VARCHAR(50),
            razorpay_timestamp TIMESTAMPTZ,
            
            x402_tx_id VARCHAR(100) UNIQUE,
            x402_payer VARCHAR(100),
            x402_payto VARCHAR(100),
            x402_network VARCHAR(100),
            x402_amount_atomic VARCHAR(50),
            x402_timestamp TIMESTAMPTZ,
            x402_settlement_status VARCHAR(50),
            
            credit_grant_status VARCHAR(50) NOT NULL,
            credit_balance_after INTEGER,
            
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_topup_hashed_phone ON topup_transactions(hashed_phone);
        CREATE INDEX IF NOT EXISTS idx_topup_rzp_id ON topup_transactions(razorpay_payment_id);
        """
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(schema_query)
            conn.commit()
            logger.info("WalletDB tables initialized successfully.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize WalletDB tables: {e}")
        finally:
            self.connection_pool.putconn(conn)

    def get_balance(self, phone: str) -> int:
        """Returns the current query credit balance for a farmer."""
        if not self.connection_pool: return 0
        
        hashed_phone = hash_phone_number(phone)
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT credit_balance FROM user_credits WHERE hashed_phone = %s", (hashed_phone,))
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
        finally:
            self.connection_pool.putconn(conn)

    def deduct_credit(self, phone: str) -> bool:
        """Deducts 1 credit if balance > 0. Returns True on success, False if empty."""
        if not self.connection_pool: return False
        
        hashed_phone = hash_phone_number(phone)
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # PostgreSQL atomic update with returning clause
                cursor.execute("""
                    UPDATE user_credits 
                    SET credit_balance = credit_balance - 1, updated_at = CURRENT_TIMESTAMP
                    WHERE hashed_phone = %s AND credit_balance > 0
                    RETURNING credit_balance;
                """, (hashed_phone,))
                row = cursor.fetchone()
                conn.commit()
                return bool(row) # True if update succeeded and returned a row
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deducting credit: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def grant_credits_and_log(
        self, 
        phone: str, 
        package_id: str, 
        credits_to_add: int,
        payment_source: str,
        fiat_amount: float = None,
        session_id: str = None,
        rzp_data: dict = None,
        x402_data: dict = None
    ) -> bool:
        """
        The Master Transaction Function.
        Upserts the user's balance AND writes the massive audit log in ONE database transaction.
        """
        if not self.connection_pool: return False
        
        hashed_phone = hash_phone_number(phone)
        rzp = rzp_data or {}
        x402 = x402_data or {}
        current_time = datetime.now(timezone.utc)
        
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # 1. UPSERT the user balance
                cursor.execute("""
                    INSERT INTO user_credits (hashed_phone, credit_balance, tier_last_purchased, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (hashed_phone) DO UPDATE 
                    SET credit_balance = user_credits.credit_balance + EXCLUDED.credit_balance,
                        tier_last_purchased = EXCLUDED.tier_last_purchased,
                        updated_at = EXCLUDED.updated_at
                    RETURNING credit_balance;
                """, (hashed_phone, credits_to_add, package_id, current_time))
                
                new_balance = cursor.fetchone()[0]

                # 2. Write the Audit Log
                cursor.execute("""
                    INSERT INTO topup_transactions (
                        hashed_phone, session_id, payment_source, package_id, fiat_amount, credits_granted,
                        razorpay_payment_id, razorpay_order_id, razorpay_payment_status, razorpay_timestamp,
                        x402_tx_id, x402_payer, x402_payto, x402_network, x402_amount_atomic, x402_timestamp, x402_settlement_status,
                        credit_grant_status, credit_balance_after
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s
                    );
                """, (
                    hashed_phone, session_id, payment_source, package_id, fiat_amount, credits_to_add,
                    rzp.get('payment_id'), rzp.get('order_id'), rzp.get('status'), rzp.get('timestamp'),
                    x402.get('tx_id'), x402.get('payer'), x402.get('payto'), x402.get('network'), x402.get('amount_atomic'), x402.get('timestamp'), x402.get('status', 'PENDING'),
                    'COMPLETED', new_balance
                ))
            
            conn.commit()
            logger.info(f"Granted {credits_to_add} credits to {phone[-4:]}. New balance: {new_balance}")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to grant credits and log transaction: {e}")
            return False
        finally:
            self.connection_pool.putconn(conn)

    def close(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("WalletDB connection pool closed.")