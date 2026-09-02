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
import random
import asyncio
import json
from google import genai
from google.genai import types
from whatsapp import get_media_url, download_media  # add to existing whatsapp imports
import logging
import os
from typing import Optional, Tuple
from datetime import date, timedelta
import httpx
from wallet_db import WalletDB
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
    create_topup_payment_link,
    verify_webhook_signature,
)
from session_store import SessionStore, normalize_phone
from wallet_monitor import handle_payment_event
from whatsapp import (
    is_duplicate_message,
    mark_as_read,
    parse_incoming_webhook,
    send_buttons,
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
wallet_db      = WalletDB()   

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
    wallet_db.close()
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
    # ── Route by message type ─────────────────────────────────────────────
    if msg_type == "text":
        text = msg.get("text") or ""
        await _handle_text_message(phone, text, session, lang)
    elif msg_type == "audio":
        audio_id = msg.get("audio_id")
        media_url = await get_media_url(audio_id) if audio_id else None
        
        if not media_url:
            await send_text(phone, "⚠️ ऑडिओ डाउनलोड करता आला नाही. कृपया टाईप करा." if lang == "mr" else "⚠️ Audio download failed. Please type.")
            return
            
        audio_bytes = await download_media(media_url)
        transcribed_text = await _transcribe_audio(audio_bytes)
        
        if not transcribed_text:
            await send_text(phone, "⚠️ ऑडिओ समजला नाही. कृपया टाईप करा." if lang == "mr" else "⚠️ Could not understand audio. Please type.")
            return
            
        logger.info(f"[Main] 🎙️ Audio transcribed for {phone[-4:]}: '{transcribed_text[:60]}'")
        
        # QA Catch: Check if they said "Cancel" or "Delete" via voice
        text_lower = transcribed_text.lower().strip()
        if any(trigger in text_lower for trigger in _DELETE_TRIGGERS):
            await _handle_data_deletion(phone, session, lang)
            return
        if any(trigger in text_lower for trigger in _CANCEL_TRIGGERS):
            if session:
                store.clear_session(session.get("session_id", ""))
            await send_text(phone, _CANCEL_REPLY.get(lang, _CANCEL_REPLY["mr"]))
            return

        # Treat as standard text flow
        await _handle_text_message(phone, transcribed_text, session, lang)

    elif msg_type in ("button_reply", "list_reply"):
        reply_id    = msg.get("reply_id") or ""
        reply_title = msg.get("reply_title") or ""

        # ── Horizon button reply ──────────────────────────────────────────
        if reply_id in ("horizon_15d", "horizon_1m", "horizon_2m") and session:
            session_id = session.get("session_id", "")
            
            today = date.today()
            horizon_map = {
                "horizon_15d": timedelta(days=15),
                "horizon_1m":  timedelta(days=30),
                "horizon_2m":  timedelta(days=60),
            }
            harvest_date = (today + horizon_map[reply_id]).strftime("%Y-%m-%d")
            success = store.update_session_data(
                session_id,
                harvest_date  = harvest_date,
                horizon_asked = False,
            )
            logger.info(
                f"[Main] Horizon resolved: {reply_id} → harvest_date={harvest_date} "
                f"for session {session_id} (save={'ok' if success else 'FAILED'})"
            )
            dpdpa_consent = (
                "📍 *स्थान माहिती*: चांगल्या सेवेसाठी आम्हाला आपले स्थान आवश्यक आहे.\n"
                "आपले स्थान फक्त या विनंतीसाठी वापरले जाईल."
                if lang == "mr" else
                "📍 *Location*: We need your location for accurate results.\n"
                "It will only be used for this request."
            )
            await send_text(phone, dpdpa_consent)
            await send_location_request(
                to        = phone,
                body_text = _LOCATION_REQUEST_TEXT.get(lang, _LOCATION_REQUEST_TEXT["mr"]),
            )
            return

        if session:
            store.update_session_data(
                session.get("session_id", ""),
                last_button_reply_id=reply_id,
            )
        # ── For topup package buttons, use reply_id not reply_title ──────
        # reply_id = "PACK_20"/"PACK_30", reply_title = "₹20 — 5 प्रश्न"
        # Passing reply_title would re-trigger topup_select via ₹20 keyword
        text = reply_id or reply_title
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


# ─── Geocoding helper (text location fallback — Bug 1) ───────────────────────

async def _geocode_text_location(text: str) -> Optional[Tuple[float, float]]:
    """
    Tries to geocode a free-text location string (village / taluka / district)
    using the Nominatim API (no key required, rate-limited to 1 req/s).

    Returns (lat, lon) tuple if confident, None if not found or ambiguous.
    Called only when session needs location and farmer typed text instead of
    tapping the location button.

    Fail-open on every error — worst case farmer gets the "awaiting location"
    reminder and taps the button instead.
    """
    if not text or len(text.strip()) > 80:
        return None

    # ── Agri keyword guard — don't geocode a new crop/pest question ───────
    _AGRI_SIGNALS = {
        # Marathi
        "पीक", "शेत", "रोग", "कीड", "खत", "फवारणी", "बियाणे", "पाऊस",
        "हवामान", "मंडी", "भाव", "किंमत", "मावा", "करपा", "भुरी",
        # English + transliterated
        "crop", "pest", "disease", "spray", "fertilizer", "weather",
        "price", "mandi", "soybean", "cotton", "onion", "wheat", "maize",
        "tomato", "potato", "blight", "aphid", "borer", "fungus",
        "insecticide", "fungicide", "herbicide", "irrigation", "rain",
    }
    text_lower = text.lower().strip()
    
    if any(sig.lower() in text_lower for sig in _AGRI_SIGNALS):
        return None

    # ── Query normalisation ────────────────────────────────────────────────
    # Strip common abbreviations so "pune mh" → "pune" before appending state.
    _MH_ALIASES = {"maharashtra", "maha", "mh", "महाराष्ट्र", "india", "भारत"}
    # Also strip "india" variants so we control the suffix ourselves.
    tokens = [t for t in text_lower.split() if t not in _MH_ALIASES]
    clean_text = " ".join(tokens).strip(" ,")

    # ── Smart Maharashtra append ───────────────────────────────────────────
    # Only append if the farmer hasn't already included the state in any form.
    # This prevents "Shirur, Pune, Maharashtra, India, Maharashtra, India".
    _MH_PRESENT = {"maharashtra", "maha", "mh", "महाराष्ट्र"}
    already_has_state = any(alias in text_lower for alias in _MH_PRESENT)
    if already_has_state:
        query = f"{clean_text}, India"
    else:
        query = f"{clean_text}, Maharashtra, India"

    logger.info(f"[Geocode] Query: '{query}'")

    try:
        import httpx

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "countrycodes": "in",
            # Bias results toward Maharashtra bounding box so "Shirur" picks
            # Pune district (18.8°N, 74.4°E) not Karnataka (15.1°N, 76.9°E).
            "viewbox": "72.5,22.1,80.9,15.5",   # Maharashtra rough bbox
            "bounded": "0",                        # soft bias, not hard clip
        }
        headers = {
            # Nominatim policy: must identify app + contact email or IP gets banned.
            "User-Agent": "Farmyworth-AgriIntel/1.0 (contact@farmyworth.com)",
        }

        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(url, params=params, headers=headers)

        # ── Explicit 429 handling ──────────────────────────────────────────
        if resp.status_code == 429:
            logger.warning(
                "[Geocode] Nominatim rate-limited (429) — failing open. "
                "Farmer will get location reminder instead."
            )
            return None

        if resp.status_code != 200:
            logger.warning(f"[Geocode] Nominatim returned {resp.status_code}")
            return None

        results = resp.json()
        if not results:
            logger.info(f"[Geocode] No results for '{query}'")
            return None

        top = results[0]
        lat = float(top["lat"])
        lon = float(top["lon"])
        logger.info(
            f"[Geocode] '{text}' → lat={lat}, lon={lon} "
            f"(resolved: {top.get('display_name', '')})"
        )
        return lat, lon

    except Exception as e:
        logger.warning(f"[Geocode] Failed for '{text}': {e}")
        return None


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
            session_id = session.get("session_id", "")

            needs_loc_service = session.get("service_type") in ("mandi", "weather")
            if needs_loc_service and not session.get("payment_link_id"):

                # ── Horizon reply guard ───────────────────────────────────
                if session.get("horizon_asked") and not session.get("harvest_date"):
                    _AGRI_SIGNALS = {
                        # Marathi
                        "पीक", "शेत", "रोग", "कीड", "खत", "फवारणी", "बियाणे", "पाऊस",
                        "हवामान", "मंडी", "भाव", "किंमत", "मावा", "करपा", "भुरी",
                        # English + transliterated
                        "crop", "pest", "disease", "spray", "fertilizer", "weather",
                        "price", "mandi", "soybean", "cotton", "onion", "wheat", "maize",
                        "tomato", "potato", "blight", "aphid", "borer", "fungus",
                        "insecticide", "fungicide", "herbicide", "irrigation", "rain",
                    }
                    text_lower = text.lower().strip()
                    if any(sig.lower() in text_lower for sig in _AGRI_SIGNALS):
                        # New agri question — clear session, fall through to re-orchestrate
                        store.clear_session(session_id)
                        session = None
                    else:
                        # Not a new question — remind to tap horizon button
                        logger.info(
                            f"[Main] Horizon not yet answered for session {session_id}, reminding"
                        )
                        await send_buttons(
                            to        = phone,
                            body_text = (
                                "कृपया खालीलपैकी एक निवडा 👇"
                                if lang == "mr" else
                                "Please choose one below 👇"
                            ),
                            buttons   = [
                                ("horizon_15d", "१५ दिवस"),
                                ("horizon_1m",  "१ महिना"),
                                ("horizon_2m",  "२ महिने"),
                            ],
                        )
                        return

                # ── Geocode fallback (typed location, no horizon pending) ─
                # Guard: if farmer is answering a crop/pest/service question, don't geocode
                if session and not session.get("horizon_asked") and session.get("awaiting") not in ("crop", "pest_confirmation", "service"):
                    coords = await _geocode_text_location(text)
                    if coords:
                        lat, lon = coords
                        success = store.update_location(session_id, lat=lat, lon=lon)
                        if success:
                            logger.info(
                                f"[Main] Geocoded text location '{text}' → "
                                f"lat={lat}, lon={lon} for session {session_id}"
                            )
                            await _send_payment(phone, session_id, lang)
                            return

            # Check if they have a payment link id — if so, remind them
            if session and session.get("payment_link_id"):
                await send_text(phone, _AWAITING_PAYMENT_MSG.get(lang, _AWAITING_PAYMENT_MSG["mr"]))
                return

        elif current_status == "paid":
            store.clear_session(session.get("session_id"))
            session = None
            _repeat_nudge = {
                "mr": "💡 _(वारंवार प्रश्न विचारता? 'topup' लिहा आणि ₹२०/₹३० पॅक घ्या — प्रत्येक वेळी पेमेंट नको)_\n\n",
                "hi": "💡 _(बार-बार पूछते हैं? 'topup' लिखें और ₹20/₹30 पैक लें — हर बार पेमेंट नहीं)_\n\n",
                "en": "💡 _(Ask often? Reply 'topup' for a ₹20/₹30 credit pack — no payment each time)_\n\n",
            }
            await send_text(phone, _repeat_nudge.get(lang, _repeat_nudge["mr"]))

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

    status         = result.get("status")
    reply_message  = result.get("reply_message", "")
    needs_location = result.get("needs_location", False)
    detected_lang  = result.get("detected_language", lang)

    # ── Send orchestrator reply ────────────────────────────────────────────
    if reply_message:
        await send_text(phone, reply_message)

    # ── Topup: show package selection buttons ──────────────────────────────
    if status == "topup_select":
        await send_buttons(
            to=phone,
            body_text="पॅक निवडा 👇" if detected_lang == "mr" else "Choose a pack 👇",
            buttons=[
                ("PACK_20", "₹20 — 5 प्रश्न"),
                ("PACK_30", "₹30 — 10 प्रश्न"),
            ],
        )
        return
    # ── Topup: farmer picked a package → create topup payment link ─────────
    if status == "topup_payment":
        package_id = result.get("package_id")
        link_result = await create_topup_payment_link(
            session_id=session_id,
            phone=phone,
            package_id=package_id,
        )
        if link_result.get("error"):
            logger.error(
                f"[Main] Topup link creation failed: {link_result.get('error_reason')}"
            )
            await send_text(
                phone,
                "⚠️ लिंक तयार करता आली नाही. पुन्हा 'topup' पाठवा. 🙏"
                if detected_lang == "mr" else
                "⚠️ Could not create topup link. Send 'topup' again. 🙏"
            )
        else:
            pkg_labels = {
                "PACK_20": "₹20 — 5 प्रश्न",
                "PACK_30": "₹30 — 10 प्रश्न",
            }
            store.update_session_data(
                session_id,
                payment_link_id=link_result.get("payment_link_id"),
                payment_status="pending",
            )
            await send_payment_link(
                to=phone,
                body_text=(
                    f"💳 *{pkg_labels.get(package_id, package_id)} क्रेडिट पॅक*\n\n"
                    "UPI ने पेमेंट करा — PhonePe, GPay, Paytm."
                    if detected_lang == "mr" else
                    f"💳 *{pkg_labels.get(package_id, package_id)} Credit Pack*\n\nPay via UPI."
                ),
                button_label="💳 UPI ने भरा",
                url=link_result["short_url"],
                header_text="Farmyworth क्रेडिट",
            )
        return

    # ── Not routed — done ──────────────────────────────────────────────────
    if status != "routed":
        return

    # ── Routed with credits — deliver directly, no Razorpay ───────────────
    if result.get("used_credits"):
        await _deliver_with_credits(phone, session_id, detected_lang)
        return

    # ── Routed without credits — normal payment flow ───────────────────────
    if needs_location:
        needs_horizon = result.get("needs_horizon", False)

        if needs_horizon:
            store.update_session_data(session_id, horizon_asked=True)
            logger.info(f"[Main] Weather horizon ask sent for session {session_id}")
            await send_buttons(
                to=phone,
                body_text={
                    "mr": "तुम्हाला किती दिवसांचा हवामान अंदाज हवा आहे?",
                    "hi": "कितने दिनों का मौसम अनुमान चाहिए?",
                    "en": "How many days of weather forecast do you need?",
                }.get(detected_lang, "तुम्हाला किती दिवसांचा हवामान अंदाज हवा आहे?"),
                buttons=[
                    ("horizon_15d", "१५ दिवस"),
                    ("horizon_1m",  "१ महिना"),
                    ("horizon_2m",  "२ महिने"),
                ],
            )
        else:
            dpdpa_consent = (
                "📍 *स्थान माहिती*: चांगल्या सेवेसाठी आम्हाला आपले स्थान आवश्यक आहे.\n"
                "आपले स्थान फक्त या विनंतीसाठी वापरले जाईल."
                if detected_lang == "mr" else
                "📍 *Location*: We need your location for accurate results.\n"
                "It will only be used for this request."
            )
            await send_text(phone, dpdpa_consent)
            await send_location_request(
                to=phone,
                body_text=_LOCATION_REQUEST_TEXT.get(detected_lang, _LOCATION_REQUEST_TEXT["mr"]),
            )
    else:
        await _send_payment(phone, session_id, detected_lang)


async def _transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    if not audio_bytes:
        return None
    try:
        keys = [k for k in [
            os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"), os.getenv("GEMINI_API_KEY_4"),
            os.getenv("GEMINI_API_KEY_5"),
        ] if k]
        client = genai.Client(api_key=random.choice(keys))
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite",  # use existing stack model
            contents=[
                "You are an expert transcriber. Transcribe this audio exactly. Output only the spoken text in the original language (Marathi, Hindi, or English). No extra text.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
            ]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"[Main] Audio transcription failed: {e}")
        return None

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

        # ── Credit mode vs pay-per-query ──────────────────────────────────────
    if session.get("payment_mode") == "credits":
        await _deliver_with_credits(phone, session_id, lang)
    else:
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
    # ── Mandi preflight: check data exists before charging farmer ──────
    if service_type == "mandi":
        lat  = session.get("lat")
        lon  = session.get("lon")
        crop = session.get("crop", "")
        radius_km = session.get("radius_km") or 100

        if lat is not None and lon is not None and crop:
            try:
                async with httpx.AsyncClient(timeout=5.0) as http:
                    resp = await http.get(
                        "https://agriintellect.site/mandi-check",
                        params={
                            "crop": crop,
                            "lat": float(lat),
                            "lon": float(lon),
                            "radius_km": int(radius_km),
                        },
                    )
                preflight = resp.json()

                if not preflight.get("has_data", True):
                    logger.warning(
                        f"[Main] Mandi preflight blocked payment | "
                        f"session={session_id} crop={crop} radius={radius_km}km"
                    )
                    store.clear_session(session_id)
                    await send_text(
                        phone,
                        (
                            f"⚠️ माफ करा, आज *{crop}* साठी {radius_km}km च्या आत "
                            f"कोणताही सक्रिय मंडी आढळला नाही.\n\n"
                            "तुमचे पैसे वाचवले! 🌾\n\n"
                            "वेगळ्या पिकासाठी किंवा उद्या पुन्हा प्रयत्न करा."
                            if lang == "mr" else
                            f"⚠️ Sorry, no active market found for *{crop}* "
                            f"within {radius_km}km today.\n\n"
                            "Your money has been saved! 🌾\n\n"
                            "Try another crop or check again tomorrow."
                        ),
                    )
                    return

            except Exception as e:
                # Fail open — preflight error must never block payment
                logger.warning(f"[Main] Mandi preflight error (fail-open): {e}")

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

async def _deliver_with_credits(phone: str, session_id: str, lang: str) -> None:
    """
    Delivers advisory directly for farmers with credit balance.
    No Razorpay involved — deduct 1 credit on successful delivery.
    Called when orchestrator returns used_credits=True.
    """
    from delivery import deliver
    from marathi_formatter import format_response_for_whatsapp

    session = store.get_session(session_id)
    if not session:
        logger.error(f"[Main] Credit delivery: session {session_id} gone")
        await send_text(phone, "⚠️ तांत्रिक अडचण. पुन्हा प्रयत्न करा.")
        return

    service_type = session.get("service_type", "")
    farmer_msg   = session.get("original_message", "")

    # ── Need location first for mandi/weather ─────────────────────────────
    if service_type in ("mandi", "weather") and not session.get("lat"):
        # Location not yet collected — send location request
        # (credits will be used after location is received)
        dpdpa_consent = (
            "📍 *स्थान माहिती*: चांगल्या सेवेसाठी आम्हाला आपले स्थान आवश्यक आहे."
            if lang == "mr" else
            "📍 *Location*: We need your location for accurate results."
        )
        await send_text(phone, dpdpa_consent)
        await send_location_request(
            to=phone,
            body_text=_LOCATION_REQUEST_TEXT.get(lang, _LOCATION_REQUEST_TEXT["mr"]),
        )
        # Mark session as credit-based so location handler knows not to create
        # a Razorpay link when location arrives
        store.update_session_data(session_id, payment_mode="credits")
        return

    logger.info(
        f"[Main] 💳 Credit delivery | session={session_id} | service={service_type}"
    )

    try:
        result = await deliver(session)
    except Exception as e:
        logger.error(f"[Main] Credit delivery deliver() failed: {e}")
        await send_text(phone, "⚠️ माहिती मिळवताना अडचण. पुन्हा प्रयत्न करा. 🙏")
        return

    if result.get("error"):
        error_type = result.get("error_type", "UNKNOWN")
        logger.error(f"[Main] Credit delivery error: {error_type}")
        await send_text(
            phone,
            "⚠️ माहिती उपलब्ध नाही. क्रेडिट वापरला नाही — पुन्हा प्रयत्न करा. 🙏"
            if lang == "mr" else
            "⚠️ Data unavailable. Credit not used — please try again. 🙏"
        )
        # Don't deduct — delivery failed
        return

           # For fertilizer: deduct here after successful delivery
    # For mandi/weather: already deducted in orchestrator at location-collection step
    session_data = store.get_session(session_id) or {}
    if not session_data.get("credit_deducted", False):
        deducted = wallet_db.deduct_credit(phone)
        if deducted:
            store.update_session_data(session_id, credit_deducted=True)
            logger.info(f"[Main] 💳 Credit deducted post-delivery for {phone[-4:]}")
        else:
            logger.warning(f"[Main] Credit deduction returned False for {phone[-4:]}")
    else:
        logger.info(f"[Main] 💳 Credit already deducted at routing for {phone[-4:]}")

    # ── Format and send ────────────────────────────────────────────────────
    try:
        formatted = await format_response_for_whatsapp(
            service_type=service_type,
            raw_data=result,
            original_user_text=farmer_msg,
        )
    except Exception as e:
        logger.error(f"[Main] Credit delivery formatter failed: {e}")
        await send_text(
            phone,
            "✅ *माहिती मिळाली* — पण फॉर्मेट करताना अडचण आली.\n"
            "कृपया पुन्हा प्रयत्न करा. 🙏"
        )
        return

    await send_text(phone, formatted)

    # Topup nudge — highest receptivity moment, right after value delivery
    _topup_nudge = {
        "mr": "💡 *वारंवार प्रश्न विचारता?*\n'topup' लिहा — ₹२०/₹३० पॅकमध्ये पेमेंट एकदाच करा.",
        "hi": "💡 *बार-बार सवाल पूछते हैं?*\n'topup' लिखें — ₹20/₹30 पैक में एक बार पेमेंट करें।",
        "en": "💡 *Ask often?*\nReply 'topup' — pay once with a ₹20/₹30 credit pack.",
    }
    await send_text(phone, _topup_nudge.get(lang, _topup_nudge["mr"]))

    # ── Mark result delivered ──────────────────────────────────────────────
    store.update_session_data(session_id, result_ready=True, retry_scheduled=False)
    logger.info(f"[WalletMonitor] ✅ Response delivered to {phone[-4:]} for session {session_id}")
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
            wallet_db      = wallet_db,
        )
    )

    return Response(status_code=200)


# ─── Dev runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)