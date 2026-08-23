"""
wallet_monitor.py
──────────────────
Handles Razorpay webhook events AFTER main.py has already:
    1. Verified the X-Razorpay-Signature
    2. Called razorpay_handler.parse_webhook_event()
    3. Got back a normalized event dict

This file owns exactly one job:
    handle_payment_event(event, store, consent_logger) → None

For "paid"  → mark paid → call delivery → format → send to farmer
For "expired"  → mark expired → send expiry message → offer fresh link
For "cancelled" → mark cancelled → send cancellation message

EDGE CASES HANDLED:
    - Session missing from Redis (race: TTL expired before webhook arrived)
    - Double webhook (Razorpay retries on non-2xx) — idempotent payment_status check
    - delivery.deliver() returns x402 error → send clean Marathi error, don't clear session
    - x402 endpoint 503 (DATA_UNAVAILABLE) → "try again" message, session kept alive
    - Stale expired webhook arriving after farmer already re-paid on fresh session

CRITICAL RULES:
    - Never re-verify signatures here (main.py already did that)
    - Never import main.py (circular)
    - All paths are async — no sync blocking calls
    - SessionStore is passed in from main.py (singleton pattern, not created here)
    - ConsentLogger is passed in from main.py (same singleton)
    - payment_status idempotency check BEFORE any processing — prevents double delivery
"""

import asyncio
import logging
from typing import Optional

from delivery import deliver
from marathi_formatter import format_response_for_whatsapp
from razorpay_handler import create_payment_link
from session_store import SessionStore
from whatsapp import send_text, send_payment_link

logger = logging.getLogger(__name__)


# ─── Farmer-facing messages (three languages, Marathi default) ────────────────

_MESSAGES = {

    # ── Payment expired ────────────────────────────────────────────────────
    "expired": {
        "mr": (
            "⏰ *पेमेंट वेळ संपला*\n\n"
            "तुमची पेमेंट लिंक कालबाह्य झाली.\n"
            "पुन्हा माहिती मिळवण्यासाठी नवीन लिंक खाली आहे 👇"
        ),
        "hi": (
            "⏰ *पेमेंट समय समाप्त*\n\n"
            "आपका पेमेंट लिंक एक्सपायर हो गया।\n"
            "नीचे नया लिंक है 👇"
        ),
        "en": (
            "⏰ *Payment link expired*\n\n"
            "Your payment window closed. A fresh link is below 👇"
        ),
    },

    # ── Payment cancelled ──────────────────────────────────────────────────
    "cancelled": {
        "mr": (
            "❌ *पेमेंट रद्द केले*\n\n"
            "तुमचे पेमेंट रद्द झाले.\n"
            "पुन्हा माहिती हवी असल्यास तुमचा प्रश्न पाठवा. 🌾"
        ),
        "hi": (
            "❌ *पेमेंट रद्द*\n\n"
            "आपका पेमेंट कैंसिल हो गया।\n"
            "दोबारा जानकारी के लिए अपना सवाल भेजें। 🌾"
        ),
        "en": (
            "❌ *Payment cancelled*\n\n"
            "Your payment was cancelled.\n"
            "Send your question again to get a new link. 🌾"
        ),
    },

    # ── Delivery error (endpoint returned error, payment was taken) ────────
    "delivery_error": {
        "mr": (
            "⚠️ *माहिती मिळवण्यात अडचण*\n\n"
            "तुमचे पेमेंट झाले, पण सध्या डेटा उपलब्ध नाही.\n"
            "काळजी करू नका — *5 मिनिटांत पुन्हा प्रयत्न होईल.*\n\n"
            "समस्या कायम राहिल्यास तुमचा प्रश्न पुन्हा पाठवा. 🙏"
        ),
        "hi": (
            "⚠️ *जानकारी लाने में दिक्कत*\n\n"
            "आपका पेमेंट हो गया, लेकिन अभी डेटा उपलब्ध नहीं है।\n"
            "*5 मिनट में दोबारा कोशिश होगी।*\n\n"
            "समस्या बनी रहे तो सवाल फिर से भेजें। 🙏"
        ),
        "en": (
            "⚠️ *Data temporarily unavailable*\n\n"
            "Your payment went through, but data is unavailable right now.\n"
            "*We'll retry in 5 minutes.*\n\n"
            "If this persists, please send your question again. 🙏"
        ),
    },

    # ── Session not found (Redis TTL race) ────────────────────────────────
    "session_missing": {
        "mr": (
            "⚠️ *सत्र कालबाह्य झाले*\n\n"
            "तुमचे पेमेंट झाले, पण सत्र कालबाह्य झाले.\n"
            "कृपया तुमचा प्रश्न पुन्हा पाठवा — आम्ही मदत करू. 🌾"
        ),
        "hi": (
            "⚠️ *सेशन एक्सपायर*\n\n"
            "पेमेंट हो गया, लेकिन सेशन एक्सपायर हो गया।\n"
            "कृपया अपना सवाल दोबारा भेजें। 🌾"
        ),
        "en": (
            "⚠️ *Session expired*\n\n"
            "Payment received but session expired.\n"
            "Please send your question again. 🌾"
        ),
    },

    # ── Expiry resend button label ─────────────────────────────────────────
    "pay_button": {
        "mr": "💳 UPI ने भरा",
        "hi": "💳 UPI से भरें",
        "en": "💳 Pay via UPI",
    },

    # ── Expiry resend header ───────────────────────────────────────────────
    "pay_header": {
        "mr": "AgriIntel माहिती",
        "hi": "AgriIntel जानकारी",
        "en": "AgriIntel Advisory",
    },
}


def _msg(key: str, lang: str) -> str:
    """Safe message lookup — falls back to Marathi if lang not found."""
    lang_map = _MESSAGES.get(key, {})
    return lang_map.get(lang, lang_map.get("mr", ""))


# ─── Main entry point ─────────────────────────────────────────────────────────

async def handle_payment_event(
    event: dict,
    store: SessionStore,
    consent_logger=None,
) -> None:
    """
    Called by main.py after webhook verification and parsing.

    event dict (from razorpay_handler.parse_webhook_event):
        {
            "event":           "paid" | "expired" | "cancelled",
            "session_id":      str,
            "payment_link_id": str,
            "amount_paid":     int,   # paise
            "phone":           str | None,
            "service_type":    str | None,
        }

    store: SessionStore instance (shared singleton from main.py)
    consent_logger: ConsentLogger instance or None (DPDPA logging)
    """
    event_type   = event.get("event")
    session_id   = event.get("session_id")
    phone_backup = event.get("phone")          # from Razorpay notes — backup only
    amount_paid  = event.get("amount_paid", 0)

    logger.info(f"[WalletMonitor] Event='{event_type}' session={session_id}")

    # ── Fetch session from Redis ───────────────────────────────────────────
    session = store.get_session(session_id) if session_id else None

    if not session:
        # Race condition: Redis TTL expired before Razorpay webhook arrived,
        # OR session_id is None (shouldn't happen — parse_webhook_event guards this)
        logger.warning(
            f"[WalletMonitor] Session {session_id} not found in Redis "
            f"for '{event_type}' event — sending fallback message"
        )
        phone = phone_backup
        if phone and event_type == "paid":
            # Farmer paid but we can't serve them — be honest
            await send_text(phone, _msg("session_missing", "mr"))
        return

    phone        = session.get("phone") or phone_backup
    service_type = session.get("service_type", "")
    lang         = session.get("language", "mr")
    farmer_msg   = session.get("original_message", "")

    if not phone:
        logger.error(
            f"[WalletMonitor] No phone in session {session_id} or event — cannot reply"
        )
        return

    # ── Route by event type ────────────────────────────────────────────────
    if event_type == "paid":
        await _handle_paid(
            session=session,
            session_id=session_id,
            phone=phone,
            service_type=service_type,
            lang=lang,
            farmer_msg=farmer_msg,
            amount_paid=amount_paid,
            store=store,
            consent_logger=consent_logger,
        )

    elif event_type == "expired":
        await _handle_expired(
            session=session,
            session_id=session_id,
            phone=phone,
            service_type=service_type,
            lang=lang,
            store=store,
        )

    elif event_type == "cancelled":
        await _handle_cancelled(
            session_id=session_id,
            phone=phone,
            lang=lang,
            store=store,
        )

    else:
        # Should never reach here — parse_webhook_event only returns handled events
        logger.warning(f"[WalletMonitor] Unhandled event type: '{event_type}'")


# ─── paid handler ─────────────────────────────────────────────────────────────

async def _handle_paid(
    session: dict,
    session_id: str,
    phone: str,
    service_type: str,
    lang: str,
    farmer_msg: str,
    amount_paid: int,
    store: SessionStore,
    consent_logger,
) -> None:
    """
    Idempotent: checks payment_status before processing.
    Razorpay may deliver the same 'paid' webhook more than once.
    """

    # ── Idempotency check — Razorpay retries on non-2xx ───────────────────
    current_status = session.get("payment_status")
    if current_status == "paid":
        logger.info(
            f"[WalletMonitor] Session {session_id} already marked paid — "
            f"skipping duplicate webhook"
        )
        return

    # ── Mark paid immediately before any async work ────────────────────────
    # Mark first so a second concurrent webhook sees "paid" and exits early
    store.update_payment_status(session_id, "paid")
    logger.info(
        f"[WalletMonitor] ✅ Payment confirmed | session={session_id} | "
        f"service={service_type} | ₹{amount_paid / 100:.2f}"
    )

    # ── DPDPA consent log ─────────────────────────────────────────────────
    if consent_logger:
        try:
            consent_logger.log_consent(
                phone=phone,
                consent_type="payment_completed",
                granted=True,
                language=lang,
                metadata={
                    "session_id":   session_id,
                    "service_type": service_type,
                    "amount_paise": amount_paid,
                },
            )
        except Exception as e:
            # Non-fatal — log and continue, farmer must get their data
            logger.warning(f"[WalletMonitor] Consent log failed (non-fatal): {e}")

    # ── Call delivery (x402 → endpoint) ───────────────────────────────────
    try:
        result = await deliver(session)
    except Exception as e:
        logger.error(f"[WalletMonitor] deliver() raised exception: {e}")
        await send_text(phone, _msg("delivery_error", lang))
        return

    # ── Handle delivery errors ─────────────────────────────────────────────
    if result.get("error") is True:
        error_type = result.get("error_type", "UNKNOWN")
        logger.error(
            f"[WalletMonitor] Delivery error for session {session_id}: "
            f"{error_type} — {result.get('error_reason', '')}"
        )

        if error_type == "DATA_UNAVAILABLE":
            # Endpoint 503 — schedule a retry in 5 minutes
            # Reset payment_status back to "paid" so retry logic can re-trigger
            # (main.py retry scheduler checks payment_status="paid" + result_ready=False)
            store.update_session_data(session_id, result_ready=False, retry_scheduled=True)
            asyncio.create_task(
                _retry_delivery_after_delay(
                    session_id=session_id,
                    phone=phone,
                    lang=lang,
                    store=store,
                    delay_seconds=300,   # 5 minutes
                )
            )
        elif error_type == "SESSION_INCOMPLETE":
            # This means orchestrator missed a required field — ask farmer to re-send
            await send_text(
                phone,
                (
                    "⚠️ *माहिती अपूर्ण*\n\n"
                    "कृपया तुमचा प्रश्न पुन्हा पाठवा — "
                    "यावेळी पीक आणि ठिकाण दोन्ही नमूद करा. 🌾"
                    if lang == "mr" else
                    _msg("delivery_error", lang)
                ),
            )
        else:
            await send_text(phone, _msg("delivery_error", lang))
        return

    # ── Format and send to farmer ──────────────────────────────────────────
    try:
        # DEBUG: log what the formatter actually receives — remove after confirming
        # mandi hallucination bug is fixed
        logger.info(
            f"[WalletMonitor] Formatter input | service={service_type} | "
            f"result_keys={list(result.keys())} | "
            f"top_mandis={[m.get('market') for m in result.get('top_mandis', [])]}"
            if service_type == 'mandi' else
            f"[WalletMonitor] Formatter input | service={service_type} | "
            f"result_keys={list(result.keys())}"
        )
        formatted = await format_response_for_whatsapp(
            service_type=service_type,
            raw_data=result,
            original_user_text=farmer_msg,
        )
    except Exception as e:
        logger.error(f"[WalletMonitor] Formatter failed: {e}")
        # Formatter crashed — send raw summary rather than silence
        await send_text(
            phone,
            "✅ *माहिती मिळाली* — पण फॉर्मेट करताना अडचण आली.\n"
            "कृपया तुमचा प्रश्न पुन्हा पाठवा. 🙏",
        )
        return

    await send_text(phone, formatted)

    # ── Mark result delivered ──────────────────────────────────────────────
    store.update_session_data(session_id, result_ready=True, retry_scheduled=False)
    logger.info(f"[WalletMonitor] ✅ Response delivered to {phone[-4:]} for session {session_id}")


# ─── expired handler ──────────────────────────────────────────────────────────

async def _handle_expired(
    session: dict,
    session_id: str,
    phone: str,
    service_type: str,
    lang: str,
    store: SessionStore,
) -> None:
    """
    Payment link expired (20-min window elapsed).
    Mark session expired, send message, issue a fresh payment link.
    Fresh link reuses same session_id — no re-extraction needed.
    """
    # ── Idempotency guards ─────────────────────────────────────────────────
    current_status = session.get("payment_status")

    if current_status == "paid":
        # Stale expired webhook arriving after farmer already paid — skip entirely.
        # Do NOT clear the session; wallet_monitor._handle_paid already delivered.
        logger.info(
            f"[WalletMonitor] Expired webhook for already-paid session "
            f"{session_id} — ignoring"
        )
        return

    if current_status == "expired":
        # Razorpay retry of the same expired event — already handled.
        logger.info(
            f"[WalletMonitor] Duplicate expired webhook for session "
            f"{session_id} — ignoring"
        )
        return

    # Bug 2 fix: original code checked payment_link_id to decide whether to
    # send an "awaiting payment" reminder — this was wrong.  A session that
    # Razorpay just told us expired should always be marked expired and a fresh
    # link issued.  The payment_link_id guard was silently swallowing the expiry
    # event for every normal session (which always has a link_id by the time
    # Razorpay fires the expiry webhook).

    store.update_payment_status(session_id, "expired")
    logger.info(f"[WalletMonitor] Payment link expired for session {session_id}")

    # ── Send expiry message ────────────────────────────────────────────────
    await send_text(phone, _msg("expired", lang))

    # ── Issue a fresh payment link for same session ────────────────────────
    try:
        link_result = await create_payment_link(
            session_id=session_id,
            phone=phone,
            service_type=service_type,
        )

        if link_result.get("error"):
            logger.error(
                f"[WalletMonitor] Could not create fresh link for expired session "
                f"{session_id}: {link_result.get('error_reason')}"
            )
            # Can't re-issue link — tell farmer to re-send their message
            await send_text(
                phone,
                (
                    "नवीन लिंक तयार करता आली नाही. कृपया तुमचा प्रश्न पुन्हा पाठवा. 🙏"
                    if lang == "mr" else
                    "Could not issue a new link. Please send your question again. 🙏"
                ),
            )
            return

        new_url = link_result.get("short_url", "")
        button_label = _msg("pay_button", lang)
        header_text  = _msg("pay_header", lang)

        await send_payment_link(
            to=phone,
            body_text=(
                "तुमची माहिती तयार आहे. खाली UPI ने पेमेंट करा 👇"
                if lang == "mr" else
                "Your advisory is ready. Pay via UPI below 👇"
            ),
            button_label=button_label,
            url=new_url,
            header_text=header_text,
        )

        # Update session with new payment link id
        store.update_session_data(
            session_id,
            payment_link_id=link_result.get("payment_link_id"),
            payment_status="pending",   # reset to pending for the new link
        )
        logger.info(
            f"[WalletMonitor] Fresh link issued for expired session {session_id}"
        )

    except Exception as e:
        logger.error(f"[WalletMonitor] Fresh link creation failed: {e}")
        await send_text(phone, "कृपया तुमचा प्रश्न पुन्हा पाठवा. 🙏")


# ─── cancelled handler ────────────────────────────────────────────────────────

async def _handle_cancelled(
    session_id: str,
    phone: str,
    lang: str,
    store: SessionStore,
) -> None:
    """
    Farmer cancelled the payment link (rare — Razorpay allows manual cancellation).
    No re-issue — farmer must send fresh message to restart flow.
    """
    store.update_payment_status(session_id, "cancelled")
    logger.info(f"[WalletMonitor] Payment cancelled for session {session_id}")
    await send_text(phone, _msg("cancelled", lang))


# ─── Retry delivery after delay (for 503 endpoint errors) ────────────────────

async def _retry_delivery_after_delay(
    session_id: str,
    phone: str,
    lang: str,
    store: SessionStore,
    delay_seconds: int = 300,
) -> None:
    """
    Background task — waits delay_seconds then retries delivery once.
    Only fires when x402 endpoint returned DATA_UNAVAILABLE (503).
    Farmer was already told "retry in 5 minutes" before this task starts.

    If retry also fails → send final error message, no further retries.
    One retry only — don't spam the farmer.
    """
    logger.info(
        f"[WalletMonitor] Retry scheduled for session {session_id} "
        f"in {delay_seconds}s"
    )
    await asyncio.sleep(delay_seconds)

    # Re-fetch session — it may have been updated or cleared
    session = store.get_session(session_id)
    if not session:
        logger.warning(
            f"[WalletMonitor] Retry: session {session_id} gone from Redis — aborting"
        )
        return

    # Skip if somehow already delivered (e.g. farmer re-sent and got new session)
    if session.get("result_ready"):
        logger.info(
            f"[WalletMonitor] Retry: session {session_id} already delivered — skipping"
        )
        return

    service_type = session.get("service_type", "")
    farmer_msg   = session.get("original_message", "")

    logger.info(f"[WalletMonitor] Retrying delivery for session {session_id}")

    try:
        result = await deliver(session)
    except Exception as e:
        logger.error(f"[WalletMonitor] Retry deliver() failed: {e}")
        await send_text(phone, _msg("delivery_error", lang))
        return

    if result.get("error") is True:
        logger.error(
            f"[WalletMonitor] Retry also failed for {session_id}: "
            f"{result.get('error_type')}"
        )
        await send_text(phone, _msg("delivery_error", lang))
        return

    try:
        formatted = await format_response_for_whatsapp(
            service_type=service_type,
            raw_data=result,
            original_user_text=farmer_msg,
        )
    except Exception as e:
        logger.error(f"[WalletMonitor] Retry formatter failed: {e}")
        await send_text(phone, _msg("delivery_error", lang))
        return

    await send_text(phone, formatted)
    store.update_session_data(session_id, result_ready=True, retry_scheduled=False)
    logger.info(
        f"[WalletMonitor] ✅ Retry successful — delivered to {phone[-4:]} "
        f"for session {session_id}"
    )