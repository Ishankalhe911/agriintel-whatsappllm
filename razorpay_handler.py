"""
razorpay_handler.py
────────────────────
Fiat/UPI payment layer (Razorpay) — sits alongside x402_client.py (which
handles USDC/Algorand). Same job, different rail: collect ₹ from a farmer
before delivery.py releases the advisory.

Responsibilities:
    1. Create a UPI Payment Link for a session (farmer taps it in WhatsApp
       via whatsapp.send_payment_link()).
    2. Verify Razorpay webhook signature (POST /razorpay-webhook in main.py).
    3. Parse webhook event into a normalized dict.
    4. Defense-in-depth: re-fetch a payment link status directly from API.

CRITICAL RULES:
    - RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET / RAZORPAY_WEBHOOK_SECRET from
      env vars only — never hardcoded.
    - Razorpay auth = HTTP Basic Auth (KEY_ID:KEY_SECRET) — not Bearer.
    - Never trust a webhook without verify_webhook_signature() passing.
    - reference_id on the Payment Link = session_id — this maps the webhook
      back to the correct farmer session in Redis with no second lookup table.
    - UPI Payment Links work on Android only (Razorpay limitation).
      iOS farmers get a standard link. Body text should mention UPI app names.
    - expire_by must be at least 15 min from now (Razorpay hard constraint).
      We use 20 min to avoid edge-case rejections from processing delay.
    - UPI Payment Links only work in LIVE mode, not test mode.
"""

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─── Config (env-only) ────────────────────────────────────────────────────────

RAZORPAY_KEY_ID         = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET     = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

BASE_URL = "https://api.razorpay.com/v1"

# ─── INR prices per service (single source of truth) ─────────────────────────
# These are the farmer-facing ₹ prices (Razorpay rail).
# x402_client.py has the separate USDC prices for the float wallet paying endpoints.

SERVICE_PRICES_INR: dict[str, float] = {
    "weather":    2.5,
    "mandi":      6.0,
    "fertilizer": 5.0,
}

def get_service_price(service_type: str) -> float:
    """Returns INR price for a service. Falls back to 10.0 if unknown."""
    return SERVICE_PRICES_INR.get(service_type, 10.0)


# ─── Events we act on ────────────────────────────────────────────────────────
# payment_link.paid     → farmer paid → trigger delivery
# payment_link.expired  → 20-min window elapsed → re-prompt farmer
# payment_link.cancelled → rare but handle gracefully
# Everything else (payment.*, refund.*, etc.) → ignore, return 200 to Razorpay

_HANDLED_EVENTS = {
    "payment_link.paid":      "paid",
    "payment_link.expired":   "expired",
    "payment_link.cancelled": "cancelled",
}


def _err(error_type: str, reason: str, status_code: int = 0) -> dict:
    payload = {"error": True, "error_type": error_type, "error_reason": reason}
    if status_code:
        payload["status_code"] = status_code
    return payload


def _auth() -> tuple:
    """Basic Auth tuple for httpx. Razorpay uses KEY_ID:KEY_SECRET, not Bearer."""
    return (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)


# ─── 1. Create a UPI Payment Link ────────────────────────────────────────────

async def create_payment_link(
    session_id: str,
    phone: str,
    service_type: str,
    amount_rupees: Optional[float] = None,
    description: Optional[str] = None,
) -> dict:
    """
    Creates a Razorpay UPI Payment Link tied to a session.

    session_id    → reference_id (max 40 chars — session_id is ~12 chars, safe).
                    Webhook uses this to look up the farmer's Redis session.
    phone         → customer contact in notes (backup identifier).
                    Stripped of leading '+' — Razorpay expects local format.
    service_type  → "weather" | "mandi" | "fertilizer"
                    Used to auto-lookup price from SERVICE_PRICES_INR.
    amount_rupees → override price if needed (optional — defaults to SERVICE_PRICES_INR).

    Returns on success:
        {
            "payment_link_id": "plink_xxx",
            "short_url": "https://rzp.io/...",
            "amount_paise": <int>,
            "expires_at": <unix_timestamp>
        }
    Returns on failure:
        {"error": True, "error_type": ..., "error_reason": ...}

    Caller (main.py) passes short_url to whatsapp.send_payment_link().

    NOTE: UPI Payment Links are Android-only (Razorpay limitation).
    expire_by = now + 20 min (Razorpay requires minimum 15 min from creation).
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return _err("CONFIG_ERROR", "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")

    # Price resolution — caller can override, otherwise use service table
    price = amount_rupees if amount_rupees is not None else get_service_price(service_type)
    amount_paise = int(round(price * 100))
    clean_phone = phone.lstrip("+")

    # expire_by: Razorpay minimum is 15 min from now (hard constraint).
    # We use 20 min to absorb any processing delay between this call and
    # Razorpay's server receiving it. Sending exactly 15 min risks rejection
    # if even 1 second elapses in transit.
    expires_at = int(time.time()) + (20 * 60)

    payload = {
        "upi_link":    True,           # REQUIRED — without this, creates a standard link (cards/netbanking)
        "amount":      amount_paise,
        "currency":    "INR",
        "reference_id": session_id,    # maps webhook → Redis session (max 40 chars)
        "description": description or f"AgriIntellect {service_type} माहिती",
        "expire_by":   expires_at,     # Unix timestamp — Razorpay min is 15 min
        "customer": {
            "contact": clean_phone,    # E.164 without '+' e.g. 919876543210
        },
        "notify": {
            "sms":   False,            # We notify via WhatsApp ourselves
            "email": False,
        },
        "reminder_enable": False,
        "notes": {
            "session_id":   session_id,
            "phone":        clean_phone,
            "service_type": service_type,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.post(
                f"{BASE_URL}/payment_links",   # NO trailing slash — causes 301 + auth drop
                auth=_auth(),
                json=payload,
            )

        if response.status_code in (200, 201):
            data = response.json()
            link_id = data.get("id")
            short_url = data.get("short_url")
            logger.info(
                f"[Razorpay] ✅ UPI link created | session={session_id} | "
                f"service={service_type} | ₹{price} | id={link_id}"
            )
            return {
                "payment_link_id": link_id,
                "short_url":       short_url,
                "amount_paise":    data.get("amount"),
                "expires_at":      expires_at,
            }

        logger.error(
            f"[Razorpay] Create link failed {response.status_code}: {response.text}"
        )
        return _err("CREATE_LINK_FAILED", response.text, response.status_code)

    except Exception as e:
        logger.error(f"[Razorpay] Create link pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))


# ─── 2. Webhook signature verification ───────────────────────────────────────

def verify_webhook_signature(
    raw_body: bytes, signature_header: Optional[str]
) -> bool:
    """
    Razorpay signs every webhook POST with:
        X-Razorpay-Signature: <hex_digest>
    computed as HMAC-SHA256(webhook_secret, raw_body).

    NOTE: Unlike Meta's X-Hub-Signature-256 header (which has a "sha256=" prefix),
    Razorpay's header is RAW hex only — no prefix. Do NOT strip anything.

    MUST receive raw bytes body — call BEFORE json.loads() in main.py.
    Same rule as whatsapp.verify_signature().
    """
    if not signature_header:
        logger.warning("[Razorpay] Missing X-Razorpay-Signature header")
        return False

    if not RAZORPAY_WEBHOOK_SECRET:
        logger.error("[Razorpay] RAZORPAY_WEBHOOK_SECRET not set — cannot verify")
        return False

    expected = hmac.new(
        RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    valid = hmac.compare_digest(expected, signature_header)
    if not valid:
        logger.warning("[Razorpay] Signature mismatch — rejecting payload")
    return valid


# ─── 3. Parse webhook payload ─────────────────────────────────────────────────

def parse_webhook_event(payload: dict) -> Optional[dict]:
    """
    Normalizes a Razorpay webhook into a unified dict.
    Extracts both payment_link entity and payment entity.
    """
    event = payload.get("event")
    if event not in _HANDLED_EVENTS:
        return None

    try:
        payload_data = payload.get("payload", {})
        link_entity = payload_data.get("payment_link", {}).get("entity", {})
        payment_entity = payload_data.get("payment", {}).get("entity", {})
    except (KeyError, TypeError):
        logger.warning(f"[Razorpay] Could not extract entities from event: {event}")
        return None

    # Guard: reference_id could be None if the link wasn't created by us
    session_id = link_entity.get("reference_id")
    if not session_id:
        logger.warning(
            f"[Razorpay] Webhook event '{event}' has no reference_id — not our link, ignoring"
        )
        return None

    notes = link_entity.get("notes") or {}

    return {
        "event":           _HANDLED_EVENTS[event],
        "session_id":      session_id,
        "payment_link_id": link_entity.get("id"),
        "amount_paid":     link_entity.get("amount_paid") or payment_entity.get("amount") or 0,
        "phone":           notes.get("phone"),
        "service_type":    notes.get("service_type"),
        # 🚀 ADDED FOR FINANCIAL AUDIT LOGGING:
        "payment_id":      payment_entity.get("id"),       # e.g., 'pay_PnXXXXX'
        "order_id":        payment_entity.get("order_id") or link_entity.get("order_id"), # e.g., 'order_PnXXXXX'
    }


# ─── 4. Defense-in-depth: direct status fetch ────────────────────────────────

async def get_payment_link_status(payment_link_id: str) -> dict:
    """
    Directly polls Razorpay: "what is the current status of this link?"

    Use for:
        - Reconciliation / cron jobs
        - Before re-sending an expiry reminder (verify it's actually expired)
        - When a webhook is suspected to have been missed

    NOT the primary payment confirmation path — webhooks are primary.
    This is a backup / audit tool only.

    Possible status values in response: created | paid | expired | cancelled
    """
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return _err("CONFIG_ERROR", "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.get(
                f"{BASE_URL}/payment_links/{payment_link_id}",
                auth=_auth(),
            )

        if response.status_code == 200:
            return response.json()

        logger.error(
            f"[Razorpay] Status check failed {response.status_code}: {response.text}"
        )
        return _err("STATUS_CHECK_FAILED", response.text, response.status_code)

    except Exception as e:
        logger.error(f"[Razorpay] Status check pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))