"""
main.py
────────
FastAPI application — the single entry point for the AgriIntel WhatsApp agent.

Two webhook routes:
    GET  /webhook          → Meta verification handshake (one-time setup)
    POST /webhook          → Incoming WhatsApp messages from farmers
    POST /razorpay-webhook → Razorpay payment events (paid/expired/cancelled)
    GET  /health           → Render health check (no auth)

Message flow (POST /webhook):
    1. Verify X-Hub-Signature-256 (reject if invalid)
    2. Parse payload → normalized message dict
    3. Dedup check (Meta retries webhooks)
    4. Mark as read (blue tick UX)
    5. Load or create session
    6. Check for special commands (delete my data — DPDPA)
    7. Route by message type:
        text       → orchestrate() → ack + location button OR payment link
        location   → save lat/lon → send payment link
        button/list reply → treat as text re-entry
        unsupported → polite ignore
    8. Session state machine drives what happens next:
        NEW / needs_intent  → orchestrate
        awaiting_location   → save location, send payment
        awaiting_payment    → ignore (Razorpay webhook handles this)

Payment flow (POST /razorpay-webhook):
    1. Verify X-Razorpay-Signature (reject if invalid)
    2. Parse event dict
    3. wallet_monitor.handle_payment_event() owns everything after this

CRITICAL DESIGN RULES:
    - main.py NEVER awaits heavy work inline — orchestrate/deliver are always
      wrapped in asyncio.create_task where the webhook must return 200 fast
    - WhatsApp webhook MUST return 200 within ~5s or Meta retries
    - Razorpay webhook MUST return 200 or Razorpay retries for 24h
    - main.py is the ONLY place SessionStore and ConsentLogger are instantiated
      (singleton pattern — passed into all functions that need them)
    - Raw body bytes read ONCE, before any JSON parsing, for signature checks
    - DPDPA: consent logged at conversation start (first message from new farmer)
    - DPDPA: "delete my data" command honoured immediately, session cleared
"""

import asyncio
import json
import logging
import os



from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from consent_log import ConsentLogger
from orchestrator import orchestrate
from razorpay_handler import (
    create_payment_link,
    parse_webhook_event,
    verify_webhook_signature,
)
from session_store import SessionStore, normalize_phone
from wallet_monitor import handle_payment_event
from whatsapp import (
    is_duplicate_message,
    mark_as_read,
    parse_incoming_webhook,
    send_location_request,
    send_payment_link,
    send_text,
    verify_signature,
    verify_webhook_subscription,
)

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Singletons (created once at startup, passed everywhere) ─────────────────

store          = SessionStore()
consent_logger = ConsentLogger()

# ─── DPDP data deletion triggers (all variants a farmer might type) ───────────
# Site privacy page promises: "message 'delete my data' to withdraw consent"
# DPDPA Article 12 makes this a legal obligation — must be honoured

_DELETE_TRIGGERS = {
    "delete my data", "delete data", "data delete",
    "माझा डेटा काढा", "maza data kadha", "data kadha",
    "delete karo", "data hatao", "mera data delete karo",
    "डेटा डिलीट", "data delete kara",
}

# ─── Location request message (sent after orchestrator ack) ──────────────────

_LOCATION_REQUEST_TEXT = {
    "mr": "📍 खालील बटणावर क्लिक करून तुमचे स्थान शेअर करा.\n\nकिंवा तुमचे *गाव, तालुका, जिल्हा* टाइप करा.",
    "hi": "📍 नीचे बटन दबाकर अपना स्थान शेयर करें.\n\nया अपना *गाँव, तालुका, जिला* टाइप करें।",
    "en": "📍 Tap the button below to share your location.\n\nOr type your *village, taluka, district*.",
}

# ─── Payment link body text ───────────────────────────────────────────────────

_PAYMENT_BODY = {
    "mr": "✅ माहिती तयार आहे!\n\nखाली UPI ने पेमेंट करा — PhonePe, GPay, Paytm किंवा कोणताही UPI app.\n\n⚠️ iOS वापरकर्ते: लिंक browser मध्ये उघडा.",
    "hi": "✅ जानकारी तैयार है!\n\nनीचे UPI से पेमेंट करें — PhonePe, GPay, Paytm या कोई भी UPI app.\n\n⚠️ iOS यूजर: लिंक browser में खोलें।",
    "en": "✅ Your advisory is ready!\n\nPay via UPI below — PhonePe, GPay, Paytm or any UPI app.\n\n⚠️ iOS users: open the link in your browser.",
}

_PAYMENT_BUTTON = {
    "mr": "💳 UPI ने भरा",
    "hi": "💳 UPI से भरें",
    "en": "💳 Pay via UPI",
}

_PAYMENT_HEADER = {
    "mr": "AgriIntel माहिती",
    "hi": "AgriIntel जानकारी",
    "en": "AgriIntel Advisory",
}

# ─── Unsupported message response ─────────────────────────────────────────────

_UNSUPPORTED_MSG = {
    "mr": "माफ करा, आम्ही फक्त मजकूर संदेश आणि स्थान स्वीकारतो.\nकृपया तुमचा प्रश्न मजकूरात पाठवा. 🌾",
    "hi": "माफ करें, हम सिर्फ text और location स्वीकार करते हैं.\nकृपया अपना सवाल text में भेजें। 🌾",
    "en": "Sorry, we only accept text messages and location.\nPlease send your question as text. 🌾",
}

# ─── Awaiting payment message (farmer sends text while payment pending) ───────

_AWAITING_PAYMENT_MSG = {
    "mr": "⏳ तुमचे पेमेंट अपेक्षित आहे.\nपेमेंट पूर्ण करा — वर पाठवलेल्या UPI लिंकवर क्लिक करा.\n\nनवीन प्रश्नासाठी पेमेंट पूर्ण होण्याची वाट पाहा किंवा 'रद्द' टाइप करा.",
    "hi": "⏳ आपका पेमेंट बाकी है.\nऊपर भेजे गए UPI लिंक से पेमेंट करें.\n\nनए सवाल के लिए पेमेंट करें या 'cancel' टाइप करें।",
    "en": "⏳ Your payment is pending.\nPlease complete payment via the UPI link sent above.\n\nFor a new question, complete payment or type 'cancel'.",
}

# ─── Cancel command triggers ──────────────────────────────────────────────────

_CANCEL_TRIGGERS = {
    "cancel", "रद्द", "radd", "band karo", "stop",
    "nako", "नको", "quit",
}

_CANCEL_REPLY = {
    "mr": "✅ रद्द केले. तुमचा नवीन प्रश्न पाठवा. 🌾",
    "hi": "✅ रद्द किया। नया सवाल भेजें। 🌾",
    "en": "✅ Cancelled. Send your new question. 🌾",
}


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌾 AgriIntel WhatsApp agent starting up")
    yield
    # Shutdown — close DB connection pool cleanly
    consent_logger.close()
    logger.info("🌾 AgriIntel WhatsApp agent shut down cleanly")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgriIntel WhatsApp Agent",
    description="Agricultural advisory backend — x402 + Razorpay payment gated",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "service": "whatsapp-agent"}


# ─── GET /webhook — Meta verification handshake ───────────────────────────────

@app.get("/webhook")
async def whatsapp_verify(request: Request):
    """
    Meta calls this once when you register the webhook URL in the
    WhatsApp Business dashboard. Must respond with the challenge string.
    """
    params = request.query_params
    challenge = verify_webhook_subscription(
        mode      = params.get("hub.mode"),
        token     = params.get("hub.verify_token"),
        challenge = params.get("hub.challenge"),
    )
    if challenge:
        return PlainTextResponse(challenge, status_code=200)
    return PlainTextResponse("Forbidden", status_code=403)


# ─── POST /webhook — Incoming WhatsApp messages ───────────────────────────────

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Receives all WhatsApp events: farmer messages, delivery receipts, etc.
    Must return 200 quickly — all heavy work is fire-and-forget via create_task.
    """
    # 1. Read raw bytes FIRST — signature check needs raw body before parsing
    raw_body = await request.body()

    # 2. Verify Meta signature — reject anything that doesn't match
    sig_header = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(raw_body, sig_header):
        logger.warning("[Main] WhatsApp signature verification failed — rejected")
        # Return 200 anyway — returning 403 makes Meta retry aggressively
        return Response(status_code=200)

    # 3. Parse JSON
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.error("[Main] Invalid JSON in WhatsApp webhook body")
        return Response(status_code=200)

    # 4. Parse into normalized message dict
    msg = parse_incoming_webhook(payload)
    if not msg:
        # Status callback (delivered/read receipt) — acknowledge and ignore
        return Response(status_code=200)

    message_id = msg.get("message_id")
    phone      = msg.get("phone")
    msg_type   = msg.get("type")

    if not phone:
        return Response(status_code=200)

    # 5. Dedup — Meta retries on slow responses; prevent double processing
    if is_duplicate_message(message_id):
        logger.info(f"[Main] Duplicate message {message_id} — skipping")
        return Response(status_code=200)

    # 6. Fire-and-forget processing — return 200 to Meta immediately
    asyncio.create_task(
        _process_whatsapp_message(msg, phone, msg_type)
    )

    return Response(status_code=200)


async def _process_whatsapp_message(msg: dict, phone: str, msg_type: str) -> None:
    """
    Full message processing — runs as a background task so the webhook
    endpoint can return 200 to Meta within the timeout window.
    """
    message_id = msg.get("message_id")

    # Mark as read — blue tick so farmer knows bot received it
    await mark_as_read(message_id)

    # Load existing session by phone
    session = store.get_session_by_phone(phone)
    lang    = session.get("language", "mr") if session else "mr"

    # ── Special command check (runs before everything else) ───────────────
    if msg_type == "text":
        text_lower = (msg.get("text") or "").lower().strip()

        # DPDPA delete request — legal obligation, must honour immediately
        if any(trigger in text_lower for trigger in _DELETE_TRIGGERS):
            await _handle_data_deletion(phone, session, lang)
            return

        # Cancel pending payment
        if any(trigger in text_lower for trigger in _CANCEL_TRIGGERS):
            if session:
                store.clear_session(session.get("session_id", ""))
            await send_text(phone, _CANCEL_REPLY.get(lang, _CANCEL_REPLY["mr"]))
            return

    # ── Route by message type ─────────────────────────────────────────────
    if msg_type == "text":
        # button_reply / list_reply also arrive with text-like content
        text = msg.get("text") or ""
        await _handle_text_message(phone, text, session, lang)

    elif msg_type in ("button_reply", "list_reply"):
        # Treat interactive reply as a fresh text message
        text = msg.get("reply_title") or msg.get("reply_id") or ""
        await _handle_text_message(phone, text, session, lang)

    elif msg_type == "location":
        lat = msg.get("lat")
        lon = msg.get("lon")
        if lat is not None and lon is not None:
            await _handle_location_message(phone, lat, lon, session, lang)
        else:
            await send_text(phone, _LOCATION_REQUEST_TEXT.get(lang, _LOCATION_REQUEST_TEXT["mr"]))

    else:
        # image, audio, video, document, sticker — politely ignore
        await send_text(phone, _UNSUPPORTED_MSG.get(lang, _UNSUPPORTED_MSG["mr"]))


# ─── Text message handler ─────────────────────────────────────────────────────

async def _handle_text_message(
    phone: str,
    text: str,
    session: dict | None,
    lang: str,
) -> None:
    """
    Handles all text messages from farmers.

    State machine:
        No session        → create session skeleton, run orchestrate
        payment_status=pending → farmer sending text while payment pending
        payment_status=paid    → result already delivered, re-orchestrate
        otherwise              → re-orchestrate (new question)
    """
    clean_phone = normalize_phone(phone)

    # ── If payment is pending, don't re-orchestrate ────────────────────────
    # ── Handle existing session states ─────────────────────────────────────
    if session:
        current_status = session.get("payment_status")
        
        if current_status == "pending":
            # Check if they have a payment link id — if so, remind them
            if session.get("payment_link_id"):
                await send_text(phone, _AWAITING_PAYMENT_MSG.get(lang, _AWAITING_PAYMENT_MSG["mr"]))
                return
                
        elif current_status == "paid":
            # PREVENT STATE LEAK: Old session is finished. 
            # Clear it from Redis and set local variable to None so a fresh one is created below.
            store.clear_session(session.get("session_id"))
            session = None

    # ── Create a skeleton session if none exists ───────────────────────────
    # create_session requires crop, qty, intent, service_type but we don't
    # know these yet — pass empty strings, orchestrator fills them via
    # update_session_data() immediately after
    if not session:
        session_id = store.create_session(
            phone        = clean_phone,
            crop         = "",
            qty          = "",
            intent       = "",
            service_type = "",
        )
        if not session_id:
            logger.error(f"[Main] Failed to create session for {clean_phone[-4:]}")
            await send_text(phone, "⚠️ तांत्रिक अडचण. कृपया पुन्हा प्रयत्न करा.")
            return

        # DPDPA: log consent at conversation start (first message)
        # Site privacy page says "Starting a chat records your consent"
        try:
            consent_logger.log_consent(
                phone        = clean_phone,
                consent_type = "conversation_started",
                granted      = True,
                language     = lang,
                metadata     = {"session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"[Main] Consent log failed (non-fatal): {e}")

    else:
        session_id = session.get("session_id")

    # ── Run orchestrator — 3-stage pipeline ───────────────────────────────
    try:
        result = await orchestrate(
            message       = text,
            session_store = store,
            session_id    = session_id,
        )
    except Exception as e:
        logger.error(f"[Main] Orchestrator exception: {e}")
        await send_text(phone, "⚠️ तांत्रिक अडचण. कृपया पुन्हा प्रयत्न करा.")
        return

    status        = result.get("status")
    reply_message = result.get("reply_message", "")
    needs_location = result.get("needs_location", False)
    detected_lang  = result.get("detected_language", lang)

    # ── Send orchestrator reply ────────────────────────────────────────────
    if reply_message:
        await send_text(phone, reply_message)

    # ── If not routed, we're done ──────────────────────────────────────────
    if status != "routed":
        # not_agri / not_handled / coming_soon / needs_clarification
        # orchestrator already sent the right message — nothing more to do
        return

    # ── Routed — decide next step based on needs_location ─────────────────
    if needs_location:
        # weather or mandi — need location before payment
        # Send DPDPA consent text first, then location button
        dpdpa_consent = (
            "📍 *स्थान माहिती*: चांगल्या सेवेसाठी आम्हाला आपले स्थान आवश्यक आहे.\n"
            "आपले स्थान फक्त या विनंतीसाठी वापरले जाईल."
            if detected_lang == "mr" else
            "📍 *Location*: We need your location for accurate results.\n"
            "It will only be used for this request."
        )
        await send_text(phone, dpdpa_consent)
        await send_location_request(
            to        = phone,
            body_text = _LOCATION_REQUEST_TEXT.get(detected_lang, _LOCATION_REQUEST_TEXT["mr"]),
        )
    else:
        # fertilizer — no location needed, go straight to payment
        await _send_payment(phone, session_id, detected_lang)


# ─── Location message handler ─────────────────────────────────────────────────

async def _handle_location_message(
    phone: str,
    lat: float,
    lon: float,
    session: dict | None,
    lang: str,
) -> None:
    """
    Farmer tapped the "Send Location" button.
    Save lat/lon to session, then send payment link.
    """
    if not session:
        # Location arrived but no active session — ask them to re-send question
        await send_text(
            phone,
            (
                "📍 स्थान मिळाले, पण तुमचा प्रश्न आढळला नाही.\n"
                "कृपया आधी तुमचा प्रश्न पाठवा, मग स्थान शेअर करा. 🌾"
                if lang == "mr" else
                "📍 Location received, but no active question found.\n"
                "Please send your question first, then share location. 🌾"
            ),
        )
        return

    session_id     = session.get("session_id")
    payment_status = session.get("payment_status")

    # 1. Block if already paid
    if payment_status == "paid":
        logger.info(f"[Main] Location received but session {session_id} is already paid — ignoring")
        return

    # 2. Block if a payment link was ALREADY sent (prevents link duplicate spam)
    if session.get("payment_link_id"):
        logger.info(f"[Main] Location received, but payment link already exists for {session_id} — ignoring")
        return

    # Save location to session
    success = store.update_location(session_id, lat=float(lat), lon=float(lon))
    if not success:
        logger.error(f"[Main] Failed to save location for session {session_id}")
        await send_text(
            phone,
            "⚠️ स्थान जतन करताना अडचण आली. पुन्हा प्रयत्न करा." if lang == "mr"
            else "⚠️ Could not save location. Please try again."
        )
        return

    logger.info(f"[Main] Location saved for session {session_id}: lat={lat}, lon={lon}")

    # Send payment link
    await _send_payment(phone, session_id, lang)


# ─── Send payment link ────────────────────────────────────────────────────────

async def _send_payment(phone: str, session_id: str, lang: str) -> None:
    """
    Creates a Razorpay UPI payment link and sends it to the farmer via WhatsApp.
    service_type is read from the session — razorpay_handler looks up price automatically.
    """
    session = store.get_session(session_id)
    if not session:
        logger.error(f"[Main] Session {session_id} gone before payment link creation")
        return

    service_type = session.get("service_type", "")

    link_result = await create_payment_link(
        session_id   = session_id,
        phone        = phone,
        service_type = service_type,
        # amount_rupees not passed — razorpay_handler uses SERVICE_PRICES_INR
    )

    if link_result.get("error"):
        logger.error(
            f"[Main] Payment link creation failed for session {session_id}: "
            f"{link_result.get('error_reason')}"
        )
        await send_text(
            phone,
            (
                "⚠️ पेमेंट लिंक तयार करताना अडचण आली.\nकृपया थोड्या वेळाने पुन्हा प्रयत्न करा. 🙏"
                if lang == "mr" else
                "⚠️ Could not create payment link. Please try again shortly. 🙏"
            ),
        )
        return

    short_url       = link_result.get("short_url", "")
    payment_link_id = link_result.get("payment_link_id", "")

    # Save payment link id to session
    store.update_session_data(
        session_id,
        payment_link_id = payment_link_id,
        payment_status  = "pending",
    )

    await send_payment_link(
        to           = phone,
        body_text    = _PAYMENT_BODY.get(lang, _PAYMENT_BODY["mr"]),
        button_label = _PAYMENT_BUTTON.get(lang, _PAYMENT_BUTTON["mr"]),
        url          = short_url,
        header_text  = _PAYMENT_HEADER.get(lang, _PAYMENT_HEADER["mr"]),
    )

    logger.info(
        f"[Main] ✅ Payment link sent | session={session_id} | "
        f"service={service_type} | id={payment_link_id}"
    )


# ─── DPDPA data deletion handler ──────────────────────────────────────────────

async def _handle_data_deletion(
    phone: str,
    session: dict | None,
    lang: str,
) -> None:
    """
    DPDPA Article 12 — right to withdraw consent and delete data.
    Site privacy page explicitly promises this. Must work.
    Clears Redis session and logs the deletion event to Neon PostgreSQL.
    """
    clean_phone = normalize_phone(phone)

    if session:
        session_id = session.get("session_id", "")
        store.clear_session(session_id)
        logger.info(f"[Main] DPDPA deletion — cleared session {session_id}")

    # Log the deletion request (we keep the audit log itself per DPDPA Article 8)
    try:
        consent_logger.log_consent(
            phone        = clean_phone,
            consent_type = "data_deletion_request",
            granted      = True,
            language     = lang,
            metadata     = {"action": "session_cleared", "requested_by": "farmer"},
        )
    except Exception as e:
        logger.warning(f"[Main] Deletion consent log failed (non-fatal): {e}")

    confirmation = {
        "mr": "✅ *तुमचा सर्व डेटा काढला गेला आहे.*\n\nतुमचे स्थान, प्रश्न आणि सत्र माहिती काढली गेली आहे.\nपुन्हा माहिती हवी असल्यास कधीही संपर्क करा. 🌾",
        "hi": "✅ *आपका सभी डेटा हटा दिया गया है।*\n\nआपकी location, सवाल और session जानकारी हटा दी गई है।\nदोबारा जानकारी के लिए कभी भी संपर्क करें। 🌾",
        "en": "✅ *All your data has been deleted.*\n\nYour location, questions and session data have been removed.\nFeel free to contact us again anytime. 🌾",
    }
    await send_text(phone, confirmation.get(lang, confirmation["mr"]))


# ─── POST /razorpay-webhook — Payment events ──────────────────────────────────

@app.post("/razorpay-webhook")
async def razorpay_webhook(request: Request):
    """
    Receives payment events from Razorpay.
    Must return 200 quickly — Razorpay retries for 24h on non-2xx.
    All processing is fire-and-forget via create_task.
    """
    # 1. Read raw bytes before parsing (signature needs raw body)
    raw_body = await request.body()

    # 2. Verify Razorpay signature
    sig_header = request.headers.get("X-Razorpay-Signature")
    if not verify_webhook_signature(raw_body, sig_header):
        logger.warning("[Main] Razorpay signature verification failed — rejected")
        # Return 200 — returning 4xx causes Razorpay to retry
        return Response(status_code=200)

    # 3. Parse JSON
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.error("[Main] Invalid JSON in Razorpay webhook body")
        return Response(status_code=200)

    # 4. Parse event
    event = parse_webhook_event(payload)
    if not event:
        # Unhandled event type (payment.captured, refund.*, etc.) — ack and ignore
        return Response(status_code=200)

    # 5. Fire-and-forget — return 200 to Razorpay immediately
    asyncio.create_task(
        handle_payment_event(
            event          = event,
            store          = store,
            consent_logger = consent_logger,
        )
    )

    return Response(status_code=200)


# ─── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)