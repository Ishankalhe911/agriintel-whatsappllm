import logging
import os
from datetime import datetime, timezone
import psycopg2
from typing import Optional, Tuple, Dict, Any
from consent_log import hash_phone_number  # Reuse your existing hash function

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL")

class WalletDB:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL is not set!")
        self._init_db()

    def _get_connection(self):
        # 🚀 Helper method to get a fresh connection every time
        # Prevents "Zombie Connection" drops from Neon/Supabase
        return psycopg2.connect(self.db_url)

    def _init_db(self):
        """Creates the User Credits ledger and the Audit table using the Subject Identity Model."""
        schema_query = """
        -- 1. The B2B/B2C Balance Ledger
        CREATE TABLE IF NOT EXISTS user_credits (
            subject_type VARCHAR(30) NOT NULL,
            subject_id VARCHAR(128) NOT NULL,
            hashed_phone VARCHAR(64), -- Kept for legacy compatibility
            credit_balance INTEGER NOT NULL DEFAULT 0,
            tier_last_purchased VARCHAR(50),
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (subject_type, subject_id)
        );

        -- 2. The Immutable Audit Log
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            
            subject_type VARCHAR(30) NOT NULL,
            subject_id VARCHAR(128) NOT NULL,
            hashed_phone VARCHAR(64),
            session_id VARCHAR(100),
            
            payment_source VARCHAR(50) NOT NULL CHECK (
                payment_source IN ('WHATSAPP_UPI', 'B2B_DIRECT_X402', 'AGENT_X402')
            ),
            
            package_id VARCHAR(50) NOT NULL,
            fiat_amount NUMERIC(10, 2),
            credits_granted INTEGER NOT NULL,
            
            razorpay_payment_id VARCHAR(100) UNIQUE,
            razorpay_order_id VARCHAR(100),
            razorpay_payment_status VARCHAR(50),
            razorpay_timestamp TIMESTAMPTZ,
            
            x402_tx_id VARCHAR(100) UNIQUE,
            x402_payer VARCHAR(128),
            x402_payto VARCHAR(128),
            x402_network VARCHAR(100),
            x402_amount_atomic VARCHAR(50),
            x402_timestamp TIMESTAMPTZ,
            x402_settlement_status VARCHAR(50),
            
            credit_grant_status VARCHAR(50) NOT NULL,
            credit_balance_after INTEGER,
            
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_tx_subject ON transactions(subject_type, subject_id);
        CREATE INDEX IF NOT EXISTS idx_tx_rzp_id ON transactions(razorpay_payment_id);
        """
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(schema_query)
            conn.commit()
            logger.info("WalletDB tables initialized successfully.")
        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Failed to initialize WalletDB tables: {e}")
        finally:
            if conn and not conn.closed:
                conn.close()

    # ------------------------------------------------------------------
    # SUBJECT IDENTITY HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def whatsapp_subject(phone: str) -> tuple[str, str]:
        """Resolves a human WhatsApp user."""
        return "WHATSAPP", hash_phone_number(phone)

    @staticmethod
    def agent_subject(wallet_address: str) -> tuple[str, str]:
        """Resolves a B2B AI Agent / Drone."""
        return "AGENT", wallet_address.strip()

    # ------------------------------------------------------------------
    # READ & DEDUCT (Core Query Logic)
    # ------------------------------------------------------------------

    def get_subject_balance(self, subject_type: str, subject_id: str) -> int:
        """Internal generic balance check."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT credit_balance FROM user_credits WHERE subject_type = %s AND subject_id = %s",
                    (subject_type, subject_id)
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
        finally:
            if conn and not conn.closed:
                conn.close()

    def get_balance(self, phone: str) -> int:
        """Legacy Wrapper for orchestrator.py & wallet_monitor.py"""
        subject_type, subject_id = self.whatsapp_subject(phone)
        return self.get_subject_balance(subject_type, subject_id)

    def deduct_subject_credit(self, subject_type: str, subject_id: str) -> bool:
        """Internal generic deduction logic."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE user_credits 
                    SET credit_balance = credit_balance - 1, updated_at = CURRENT_TIMESTAMP
                    WHERE subject_type = %s AND subject_id = %s AND credit_balance > 0
                    RETURNING credit_balance;
                """, (subject_type, subject_id))
                row = cursor.fetchone()
                conn.commit()
                return bool(row)
        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Error deducting credit: {e}")
            return False
        finally:
            if conn and not conn.closed:
                conn.close()

    def deduct_credit(self, phone: str) -> bool:
        """Legacy Wrapper for wallet_monitor.py"""
        subject_type, subject_id = self.whatsapp_subject(phone)
        return self.deduct_subject_credit(subject_type, subject_id)

    # ------------------------------------------------------------------
    # GRANT & LOG (Top-ups and Transactions)
    # ------------------------------------------------------------------

    def grant_topup(
        self,
        subject_type: str,
        subject_id: str,
        package_id: str,
        credits: int,
        payment_source: str,
        session_id: Optional[str] = None,
        fiat_amount: Optional[float] = None,
        razorpay_payment_id: Optional[str] = None,
        razorpay_order_id: Optional[str] = None,
        razorpay_status: Optional[str] = None,
        razorpay_timestamp: Optional[datetime] = None,
        x402_tx_id: Optional[str] = None,
        x402_payer: Optional[str] = None,
        x402_payto: Optional[str] = None,
        x402_network: Optional[str] = None,
        x402_amount_atomic: Optional[str] = None,
        x402_timestamp: Optional[datetime] = None,
        x402_settlement_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        The Master Transaction Function.
        Handles pure fiat top-ups, pure crypto B2B top-ups, and hybrid queries safely.
        """
        if credits <= 0:
            raise ValueError("credits must be > 0")

        now = datetime.now(timezone.utc)
        hashed_phone = subject_id if subject_type == "WHATSAPP" else None

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 1. IDEMPOTENCY CHECKS
                if razorpay_payment_id:
                    cursor.execute("SELECT id, credit_balance_after FROM transactions WHERE razorpay_payment_id = %s", (razorpay_payment_id,))
                    existing = cursor.fetchone()
                    if existing:
                        return {"success": True, "duplicate": True, "credit_balance": existing[1]}

                if x402_tx_id:
                    cursor.execute("SELECT id, credit_balance_after FROM transactions WHERE x402_tx_id = %s", (x402_tx_id,))
                    existing = cursor.fetchone()
                    if existing:
                        return {"success": True, "duplicate": True, "credit_balance": existing[1]}

                # 2. UPSERT BALANCE
                cursor.execute("""
                    INSERT INTO user_credits (subject_type, subject_id, hashed_phone, credit_balance, tier_last_purchased, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (subject_type, subject_id) DO UPDATE 
                    SET credit_balance = user_credits.credit_balance + EXCLUDED.credit_balance,
                        tier_last_purchased = EXCLUDED.tier_last_purchased,
                        updated_at = EXCLUDED.updated_at
                    RETURNING credit_balance;
                """, (subject_type, subject_id, hashed_phone, credits, package_id, now))
                
                new_balance = cursor.fetchone()[0]

                # 3. WRITE AUDIT LOG
                cursor.execute("""
                    INSERT INTO transactions (
                        subject_type, subject_id, hashed_phone, session_id, payment_source, package_id, fiat_amount, credits_granted,
                        razorpay_payment_id, razorpay_order_id, razorpay_payment_status, razorpay_timestamp,
                        x402_tx_id, x402_payer, x402_payto, x402_network, x402_amount_atomic, x402_timestamp, x402_settlement_status,
                        credit_grant_status, credit_balance_after
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s
                    ) RETURNING id;
                """, (
                    subject_type, subject_id, hashed_phone, session_id, payment_source, package_id, fiat_amount, credits,
                    razorpay_payment_id, razorpay_order_id, razorpay_status, razorpay_timestamp,
                    x402_tx_id, x402_payer, x402_payto, x402_network, x402_amount_atomic, x402_timestamp, x402_settlement_status,
                    'COMPLETED', new_balance
                ))
                
                transaction_id = cursor.fetchone()[0]

            conn.commit()
            logger.info(f"✅ Granted {credits} credits to {subject_type}:{subject_id[:8]}. New balance: {new_balance}")
            
            return {
                "success": True,
                "duplicate": False,
                "transaction_id": transaction_id,
                "credit_balance": new_balance,
                "credits_granted": credits,
            }
            
        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Failed to grant credits and log transaction: {e}")
            return {"success": False}
        finally:
            if conn and not conn.closed:
                conn.close()

    def grant_credits_and_log(
        self, phone: str, package_id: str, credits_to_add: int,
        payment_source: str, fiat_amount: float = None, session_id: str = None,
        rzp_data: dict = None, x402_data: dict = None
    ) -> bool:
        """Legacy wrapper used by wallet_monitor.py for single query payments."""
        subject_type, subject_id = self.whatsapp_subject(phone)
        rzp = rzp_data or {}
        x402 = x402_data or {}
        
        result = self.grant_topup(
            subject_type=subject_type,
            subject_id=subject_id,
            package_id=package_id,
            credits=credits_to_add,
            payment_source=payment_source,
            session_id=session_id,
            fiat_amount=fiat_amount,
            razorpay_payment_id=rzp.get('payment_id'),
            razorpay_order_id=rzp.get('order_id'),
            razorpay_status=rzp.get('status'),
            razorpay_timestamp=rzp.get('timestamp'),
            x402_tx_id=x402.get('tx_id'),
            x402_payer=x402.get('payer'),
            x402_payto=x402.get('payto'),
            x402_network=x402.get('network'),
            x402_amount_atomic=x402.get('amount_atomic'),
            x402_timestamp=x402.get('timestamp'),
            x402_settlement_status=x402.get('status', 'PENDING')
        )
        return result.get("success", False)

    def close(self):
        logger.info("WalletDB using short-lived connections. Shutdown complete.")