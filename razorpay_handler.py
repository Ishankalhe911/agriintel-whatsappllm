"""
razorpay_handler.py
────────────────────
Fiat/UPI payment layer (Razorpay) — sits alongside x402_client.py (which
handles USDC/Algorand). Same job, different rail: collect payment from a
farmer before delivery.py releases the actual advisory.

Responsibilities:
    1. Create a Payment Link for a session (farmer taps it in WhatsApp,
       via whatsapp.send_payment_link()).
    2. Verify Razorpay's webhook signature (POST /webhook, called by main.py
       — SAME webhook URL as WhatsApp, or a separate one, main.py's routing
       decides; this file only cares about the Razorpay payload).
    3. Parse the webhook event into a normalized dict.
    4. (Defense-in-depth) Re-fetch a payment link's status directly from
       Razorpay's API — useful if a webhook is ever missed.

CRITICAL RULES (same conventions as x402_client.py / whatsapp.py):
    - RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET / RAZORPAY_WEBHOOK_SECRET come
      from env vars only.
    - Razorpay auth = HTTP Basic Auth (KEY_ID as username, KEY_SECRET as
      password) — NOT a Bearer token, unlike WhatsApp/Gemini.
    - Never trust a webhook without verify_webhook_signature() passing —
      same reasoning as whatsapp.verify_signature(): this endpoint literally
      controls whether a farmer gets a paid service for free.
    - reference_id on the Payment Link = session_id (from session_store.py)
      — this is how the webhook maps back to "which farmer, which crop,
      which service" without a second lookup table.
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─── Config (env-only) ───────────────────────────────────────────────────────

RAZORPAY_KEY_ID        = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET    = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

BASE_URL = "https://api.razorpay.com/v1"

# Events we actually act on. Razorpay sends many more (payment.captured,
# payment.failed, refund.processed, ...) — we ignore anything not listed
# here since Payment Links already give us a clean paid/expired/cancelled
# lifecycle without needing the lower-level payment.* events.
_HANDLED_EVENTS = {
    "payment_link.paid": "paid",
    "payment_link.expired": "expired",
    "payment_link.cancelled": "cancelled",
}


def _err(error_type: str, reason: str, status_code: int = 0) -> dict:
    payload = {"error": True, "error_type": error_type, "error_reason": reason}
    if status_code:
        payload["status_code"] = status_code
    return payload


def _auth() -> tuple:
    return (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)


# ─── 1. Create a Payment Link ────────────────────────────────────────────────

async def create_payment_link(
    session_id: str,
    phone: str,
    amount_rupees: float,
    service_type: str,
    description: Optional[str] = None,
) -> dict:
    """
    Creates a Razorpay Payment Link tied to a session.

    session_id  -> becomes reference_id (webhook uses this to find the
                   session in Redis via session_store.get_session()).
    phone       -> stored in notes (backup identifier) AND as customer
                   contact (Razorpay needs E.164-ish local format, no '+').
    amount_rupees -> converted to paise internally (Razorpay's API unit).

    Returns on success:
        {"payment_link_id": "plink_xxx", "short_url": "https://rzp.io/...", "amount": <paise>}
    Returns on failure:
        {"error": True, "error_type": ..., "error_reason": ...}

    Caller (main.py, after orchestrator + delivery decide a paid service is
    needed) should pass the short_url straight into
    whatsapp.send_payment_link(to=phone, url=short_url, ...).
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return _err("CONFIG_ERROR", "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")

    amount_paise = int(round(amount_rupees * 100))
    clean_phone = phone.lstrip("+")

    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "reference_id": session_id,
        "description": description or f"AgriIntel {service_type} advisory",
        "customer": {"contact": clean_phone},
        "notify": {"sms": False, "email": False},  # we notify via WhatsApp ourselves
        "reminder_enable": False,
        "notes": {
            "session_id": session_id,
            "phone": clean_phone,
            "service_type": service_type,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.post(
                f"{BASE_URL}/payment_links/", auth=_auth(), json=payload
            )

        if response.status_code in (200, 201):
            data = response.json()
            logger.info(f"[Razorpay] Payment link created for session {session_id}")
            return {
                "payment_link_id": data.get("id"),
                "short_url": data.get("short_url"),
                "amount": data.get("amount"),
            }

        logger.error(f"[Razorpay] Create link failed {response.status_code}: {response.text}")
        return _err("CREATE_LINK_FAILED", response.text, response.status_code)

    except Exception as e:
        logger.error(f"[Razorpay] Create link pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))


# ─── 2. Webhook signature verification ───────────────────────────────────────

def verify_webhook_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """
    Razorpay signs every webhook POST with X-Razorpay-Signature: <hex>,
    computed as HMAC-SHA256(webhook_secret, raw_body) — no "sha256=" prefix
    (unlike Meta's header), just the raw hex digest.

    MUST be called with the RAW request body (bytes), same rule as
    whatsapp.verify_signature() — read body before json-parsing it.
    """
    if not signature_header:
        logger.warning("[Razorpay] Missing signature header")
        return False

    if not RAZORPAY_WEBHOOK_SECRET:
        logger.error("[Razorpay] RAZORPAY_WEBHOOK_SECRET not set — cannot verify")
        return False

    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()

    valid = hmac.compare_digest(expected, signature_header)
    if not valid:
        logger.warning("[Razorpay] Signature mismatch — rejecting payload")
    return valid


# ─── 3. Parse webhook payload ────────────────────────────────────────────────

def parse_webhook_event(payload: dict) -> Optional[dict]:
    """
    Normalizes a Razorpay webhook payload into:
        {
            "event": "paid" | "expired" | "cancelled",
            "session_id": str,      # from reference_id -> session_store.get_session()
            "payment_link_id": str,
            "amount_paid": int,     # paise
            "phone": str or None,   # backup, from notes
        }

    Returns None for events we don't act on (main.py should just 200-ack
    and ignore those — Razorpay retries on non-2xx, not on "we ignored it").
    """
    event = payload.get("event")
    if event not in _HANDLED_EVENTS:
        return None

    try:
        entity = payload["payload"]["payment_link"]["entity"]
    except (KeyError, TypeError):
        logger.warning(f"[Razorpay] Could not find payment_link entity in payload: {event}")
        return None

    return {
        "event": _HANDLED_EVENTS[event],
        "session_id": entity.get("reference_id"),
        "payment_link_id": entity.get("id"),
        "amount_paid": entity.get("amount_paid"),
        "phone": (entity.get("notes") or {}).get("phone"),
    }


# ─── 4. Defense-in-depth: direct status check ────────────────────────────────

async def get_payment_link_status(payment_link_id: str) -> dict:
    """
    Directly asks Razorpay "what's the status of this link right now?"
    Use this as a reconciliation check (e.g. a cron job, or before
    re-sending a payment link) — NOT as the primary payment-confirmation
    path. The webhook is the primary path; this is a backup in case a
    webhook delivery is ever lost.
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return _err("CONFIG_ERROR", "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.get(
                f"{BASE_URL}/payment_links/{payment_link_id}", auth=_auth()
            )

        if response.status_code == 200:
            return response.json()

        logger.error(f"[Razorpay] Status check failed {response.status_code}: {response.text}")
        return _err("STATUS_CHECK_FAILED", response.text, response.status_code)

    except Exception as e:
        logger.error(f"[Razorpay] Status check pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))
