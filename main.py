"""
main.py
────────
FastAPI application — the single entry point for the AgriIntel WhatsApp agent.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Tuple

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from google import genai
from google.genai import types

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
    get_media_url,
    download_media,
)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Singletons ──────────────────────────────────────────────────────────────

store          = SessionStore()
consent_logger = ConsentLogger()

# ─── Strings & Commands ───────────────────────────────────────────────────────

_DELETE_TRIGGERS = {
    "delete my data", "delete data", "data delete",
    "माझा डेटा काढा", "maza data kadha", "data kadha",
    "delete karo", "data hatao", "mera data delete karo",
    "डेटा डिलीट", "data delete kara",
}

_LOCATION_REQUEST_TEXT = {
    "mr": "📍 खालील बटणावर क्लिक करून तुमचे स्थान शेअर करा.\n\nकिंवा तुमचे *गाव, तालुका, जिल्हा* टाइप करा.",
    "hi": "📍 नीचे बटन दबाकर अपना स्थान शेयर करें.\n\nया अपना *गाँव, तालुका, जिला* टाइप करें।",
    "en": "📍 Tap the button below to share your location.\n\nOr type your *village, taluka, district*.",
}

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

_UNSUPPORTED_MSG = {
    "mr": "माफ करा, आम्ही फक्त मजकूर संदेश आणि स्थान स्वीकारतो.\nकृपया तुमचा प्रश्न मजकूरात पाठवा. 🌾",
    "hi": "माफ करें, हम सिर्फ text और location स्वीकार करते हैं.\nकृपया अपना सवाल text में भेजें। 🌾",
    "en": "Sorry, we only accept text messages and location.\nPlease send your question as text. 🌾",
}

_AWAITING_PAYMENT_MSG = {
    "mr": "⏳ तुमचे पेमेंट अपेक्षित आहे.\nपेमेंट पूर्ण करा — वर पाठवलेल्या UPI लिंकवर क्लिक करा.\n\nनवीन प्रश्नासाठी पेमेंट पूर्ण होण्याची वाट पाहा किंवा 'रद्द' टाइप करा.",
    "hi": "⏳ आपका पेमेंट बाकी है.\nऊपर भेजे गए UPI लिंक से पेमेंट करें.\n\nनए सवाल के लिए पेमेंट करें या 'cancel' टाइप करें।",
    "en": "⏳ Your payment is pending.\nPlease complete payment via the UPI link sent above.\n\nFor a new question, complete payment or type 'cancel'.",
}

_CANCEL_TRIGGERS = {
    "cancel", "रद्द", "radd", "band karo", "stop",
    "nako", "नको", "quit",
}

_CANCEL_REPLY = {
    "mr": "✅ रद्द केले. तुमचा नवीन प्रश्न पाठवा. 🌾",
    "hi": "✅ रद्द किया। नया सवाल भेजें। 🌾",
    "en": "✅ Cancelled. Send your new question. 🌾",
}


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌾 AgriIntel WhatsApp agent starting up")
    yield
    consent_logger.close()
    logger.info("🌾 AgriIntel WhatsApp agent shut down cleanly")

app = FastAPI(
    title="AgriIntel WhatsApp Agent",
    description="Agricultural advisory backend — x402 + Razorpay payment gated",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "service": "whatsapp-agent"}


@app.get("/webhook")
async def whatsapp_verify(request: Request):
    params = request.query_params
    challenge = verify_webhook_subscription(
        mode      = params.get("hub.mode"),
        token     = params.get("hub.verify_token"),
        challenge = params.get("hub.challenge"),
    )
    if challenge:
        return PlainTextResponse(challenge, status_code=200)
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    raw_body = await request.body()
    sig_header = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(raw_body, sig_header):
        logger.warning("[Main] WhatsApp signature verification failed — rejected")
        return Response(status_code=200)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.error("[Main] Invalid JSON in WhatsApp webhook body")
        return Response(status_code=200)

    msg = parse_incoming_webhook(payload)
    if not msg:
        return Response(status_code=200)

    message_id = msg.get("message_id")
    phone      = msg.get("phone")
    msg_type   = msg.get("type")

    if not phone:
        return Response(status_code=200)

    if is_duplicate_message(message_id):
        logger.info(f"[Main] Duplicate message {message_id} — skipping")
        return Response(status_code=200)

    asyncio.create_task(
        _process_whatsapp_message(msg, phone, msg_type)
    )
    return Response(status_code=200)


# ─── Speech-to-Text helper (Gemini) ──────────────────────────────────────────

async def _transcribe_audio(audio_bytes: bytes) -> Optional[str]:
    """Uses Gemini to convert audio bytes to text asynchronously."""
    if not audio_bytes:
        return None
        
    try:
        client = genai.Client() 
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "You are an expert transcriber. Transcribe this audio exactly as it is spoken. Do not answer questions, do not add extra text, just output the spoken text in the original language (Marathi, Hindi, or English).",
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/ogg',
                )
            ]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"[Main] Audio transcription failed: {e}")
        return None


# ─── Main Processing ─────────────────────────────────────────────────────────

async def _process_whatsapp_message(msg: dict, phone: str, msg_type: str) -> None:
    message_id = msg.get("message_id")
    await mark_as_read(message_id)

    session = store.get_session_by_phone(phone)
    lang    = session.get("language", "mr") if session else "mr"

    if msg_type == "text":
        text_lower = (msg.get("text") or "").lower().strip()
        if any(trigger in text_lower for trigger in _DELETE_TRIGGERS):
            await _handle_data_deletion(phone, session, lang)
            return

        if any(trigger in text_lower for trigger in _CANCEL_TRIGGERS):
            if session:
                store.clear_session(session.get("session_id", ""))
            await send_text(phone, _CANCEL_REPLY.get(lang, _CANCEL_REPLY["mr"]))
            return

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
            
        logger.info(f"[Main] 🎙️ Transcribed Audio for {phone}: '{transcribed_text}'")
        await _handle_text_message(phone, transcribed_text, session, lang)

    elif msg_type in ("button_reply", "list_reply"):
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
        await send_text(phone, _UNSUPPORTED_MSG.get(lang, _UNSUPPORTED_MSG["mr"]))


# ─── Helpers (Geocoding, Handlers) ───────────────────────────────────────────

async def _geocode_text_location(text: str) -> Optional[Tuple[float, float]]:
    if not text or len(text.strip()) > 80:
        return None

    _AGRI_SIGNALS = {
        "पीक", "शेत", "रोग", "कीड", "खत", "फवारणी", "बियाणे",
        "crop", "pest", "disease", "spray", "fertilizer", "weather",
        "हवामान", "मंडी", "भाव", "किंमत", "price", "mandi",
    }
    text_lower = text.lower().strip()
    if any(sig.lower() in text_lower for sig in _AGRI_SIGNALS):
        return None

    _MH_ALIASES = {"maharashtra", "maha", "mh", "महाराष्ट्र", "india", "भारत"}
    tokens = [t for t in text_lower.split() if t not in _MH_ALIASES]
    clean_text = " ".join(tokens).strip(" ,")

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
            "viewbox": "72.5,22.1,80.9,15.5",
            "bounded": "0",
        }
        headers = {
            "User-Agent": "Farmyworth-AgriIntel/1.0 (contact@farmyworth.com)",
        }

        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(url, params=params, headers=headers)

        if resp.status_code == 429:
            logger.warning("[Geocode] Nominatim rate-limited (429)")
            return None

        if resp.status_code != 200:
            return None

        results = resp.json()
        if not results:
            return None

        top = results[0]
        lat = float(top["lat"])
        lon = float(top["lon"])
        return lat, lon
    except Exception as e:
        logger.warning(f"[Geocode] Failed for '{text}': {e}")
        return None


async def _handle_text_message(
    phone: str,
    text: str,
    session: dict | None,
    lang: str,
) -> None:
    clean_phone = normalize_phone(phone)

    if session:
        current_status = session.get("payment_status")

        if current_status == "pending":
            session_id = session.get("session_id", "")
            needs_loc_service = session.get("service_type") in ("mandi", "weather")
            if needs_loc_service and not session.get("payment_link_id"):
                coords = await _geocode_text_location(text)
                if coords:
                    lat, lon = coords
                    success = store.update_location(session_id, lat=lat, lon=lon)
                    if success:
                        await _send_payment(phone, session_id, lang)
                        return
            
            if session.get("payment_link_id"):
                await send_text(phone, _AWAITING_PAYMENT_MSG.get(lang, _AWAITING_PAYMENT_MSG["mr"]))
                return

        elif current_status == "paid":
            store.clear_session(session.get("session_id"))
            session = None

    if not session:
        session_id = store.create_session(
            phone        = clean_phone,
            crop         = "",
            qty          = "",
            intent       = "",
            service_type = "",
        )
        if not session_id:
            await send_text(phone, "⚠️ तांत्रिक अडचण. कृपया पुन्हा प्रयत्न करा.")
            return
        try:
            consent_logger.log_consent(
                phone        = clean_phone,
                consent_type = "conversation_started",
                granted      = True,
                language     = lang,
                metadata     = {"session_id": session_id},
            )
        except Exception as e:
            pass
    else:
        session_id = session.get("session_id")

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

    if reply_message:
        await send_text(phone, reply_message)

    if status != "routed":
        return

    if needs_location:
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
        await _send_payment(phone, session_id, detected_lang)


async def _handle_location_message(
    phone: str,
    lat: float,
    lon: float,
    session: dict | None,
    lang: str,
) -> None:
    if not session:
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

    if payment_status == "paid" or session.get("payment_link_id"):
        return

    success = store.update_location(session_id, lat=float(lat), lon=float(lon))
    if not success:
        await send_text(phone, "⚠️ स्थान जतन करताना अडचण आली. पुन्हा प्रयत्न करा." if lang == "mr" else "⚠️ Could not save location. Please try again.")
        return

    await _send_payment(phone, session_id, lang)


async def _send_payment(phone: str, session_id: str, lang: str) -> None:
    session = store.get_session(session_id)
    if not session:
        return

    service_type = session.get("service_type", "")
    link_result = await create_payment_link(
        session_id   = session_id,
        phone        = phone,
        service_type = service_type,
    )

    if link_result.get("error"):
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
    logger.info(f"[Main] ✅ Payment link sent | session={session_id}")


async def _handle_data_deletion(
    phone: str,
    session: dict | None,
    lang: str,
) -> None:
    clean_phone = normalize_phone(phone)

    if session:
        session_id = session.get("session_id", "")
        store.clear_session(session_id)

    try:
        consent_logger.log_consent(
            phone        = clean_phone,
            consent_type = "data_deletion_request",
            granted      = True,
            language     = lang,
            metadata     = {"action": "session_cleared", "requested_by": "farmer"},
        )
    except Exception as e:
        pass

    confirmation = {
        "mr": "✅ *तुमचा सर्व डेटा काढला गेला आहे.*\n\nतुमचे स्थान, प्रश्न आणि सत्र माहिती काढली गेली आहे.\nपुन्हा माहिती हवी असल्यास कधीही संपर्क करा. 🌾",
        "hi": "✅ *आपका सभी डेटा हटा दिया गया है।*\n\nआपकी location, सवाल और session जानकारी हटा दी गई है।\nदोबारा जानकारी के लिए कभी भी संपर्क करें। 🌾",
        "en": "✅ *All your data has been deleted.*\n\nYour location, questions and session data have been removed.\nFeel free to contact us again anytime. 🌾",
    }
    await send_text(phone, confirmation.get(lang, confirmation["mr"]))


# ─── Razorpay Webhook ────────────────────────────────────────────────────────

@app.post("/razorpay-webhook")
async def razorpay_webhook(request: Request):
    raw_body = await request.body()
    sig_header = request.headers.get("X-Razorpay-Signature")
    if not verify_webhook_signature(raw_body, sig_header):
        return Response(status_code=200)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return Response(status_code=200)

    event = parse_webhook_event(payload)
    if not event:
        return Response(status_code=200)

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
