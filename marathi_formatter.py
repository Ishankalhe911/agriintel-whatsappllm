"""
marathi_formatter.py
─────────────────────
Formatting layer for AgriIntel WhatsApp backend.

Takes raw JSON from x402 endpoints and produces a warm, clear, WhatsApp-
formatted Marathi narrative using Gemini 3.1 Flash Lite.

FIXES APPLIED vs original version:
    ✅ Fix 1: No hardcoded API keys — _get_client() rotates across all 5 keys
    ✅ Fix 2: Key rotation same as orchestrator.py (random.choice across 5 keys)
    ✅ Fix 3: All calls use client.aio.models.generate_content (async, non-blocking)
    ✅ Fix 4: All public functions are async — main.py can await them correctly
    ✅ Fix 5: service_type routing uses "weather"/"mandi"/"fertilizer" — matches
              session_store.py and delivery.py exactly (no hyphens)
    ✅ Fix 6: System prompt in GenerateContentConfig(system_instruction=...)
              not as first item in contents[] list
    ✅ Fix 7: x402 error detection checks error=True (bool key) not status="error"
    ✅ Fix 8: Client built fresh per call — no module-level client=None risk
"""

import json
import logging
import os
import random
from typing import Any, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── Key rotation (same pattern as orchestrator.py) ──────────────────────────

_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]

MODEL = "gemini-3.1-flash-lite"


def _get_client() -> genai.Client:
    """
    Returns a Gemini client using a randomly selected key from all 5 keys.
    Built fresh per call — no module-level client that breaks if key missing at import.
    Same rotation pattern as orchestrator.py.
    """
    keys = [k for k in _GEMINI_KEYS if k]
    if not keys:
        raise ValueError("No GEMINI_API_KEY_* env vars set — need at least one")
    return genai.Client(api_key=random.choice(keys))


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """तुम्ही एक अनुभवी महाराष्ट्रीयन कृषी तज्ञ आणि WhatsApp सहाय्यक आहात.
तुमचे काम म्हणजे कच्च्या JSON डेटाला शेतकऱ्यांसाठी स्पष्ट, उपयुक्त मराठीत रूपांतरित करणे.

WHATSAPP FORMATTING RULES (कठोरपणे पाळा):
1. फक्त मराठी भाषा वापरा (Devanagari script).
2. Bold साठी फक्त *एकच asterisk* वापरा — **double** नाही, # header नाही.
3. Lists साठी फक्त - किंवा • वापरा.
4. JSON keys, null values, database field names कधीही print करू नका.
5. JSON मध्ये नसलेली माहिती (किंमत, डोस, वेळ) कधीही स्वतःहून जोडू नका.
6. संबंधित emojis वापरा: 🌾 🌧️ 🐛 💰 🚜 💡 🧪 ⚠️ ✅
7. Mobile screen साठी छोटे paragraphs ठेवा.
8. शेवटी नेहमी उत्साहवर्धक sign-off द्या.
"""

# ─── Error response builder ───────────────────────────────────────────────────

def _error_response(reason: str) -> str:
    return (
        f"⚠️ *माहिती उपलब्ध नाही*\n\n{reason}\n\n"
        "कृपया थोड्या वेळाने पुन्हा प्रयत्न करा. 🙏"
    )


# ─── Core async Gemini call ───────────────────────────────────────────────────

async def _call_gemini(user_prompt: str) -> str:
    """
    Async Gemini call — uses client.aio.models.generate_content so it never
    blocks the FastAPI event loop. System prompt goes in system_instruction,
    not in contents[].
    """
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.3,        # low temp for factual precision
                max_output_tokens=2048,
            ),
        )
        return response.text.strip()
    except ValueError as e:
        logger.error(f"[Formatter] Config error: {e}")
        return _error_response("तांत्रिक अडचण: API key उपलब्ध नाही.")
    except Exception as e:
        logger.error(f"[Formatter] Gemini call failed: {e}")
        return _error_response("माहिती तयार करताना अडचण आली.")


# ─── Service-specific formatters ──────────────────────────────────────────────

async def format_weather_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /weather-risk JSON into actionable Marathi weather advisory.
    Highlights: rain forecast, spray windows, irrigation, pest risk windows.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'हवामान माहिती'}"
सेवा: हवामान जोखीम विश्लेषण

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

यातून शेतकऱ्यासाठी व्यावहारिक हवामान माहिती द्या:
- पुढील काही दिवसांत पाऊस येणार का? कधी?
- फवारणी करता येईल का? कोणत्या तारखा सर्वोत्तम?
- सिंचन करायला हवे का?
- कोणत्या रोगांचा/किडींचा धोका आहे?
- उष्णता किंवा जोराचा वारा असेल तर सांगा.
फक्त JSON मध्ये असलेली माहिती द्या."""

    return await _call_gemini(prompt)


async def format_mandi_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /mandi-optimize JSON into Marathi market advisory.
    Highlights: best mandis, prices, transport cost, net profit range.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'मंडी भाव माहिती'}"
सेवा: मंडी भाव आणि नफा विश्लेषण

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

यातून शेतकऱ्यासाठी स्पष्ट मंडी सल्ला द्या:
- कोणत्या मंडीत सर्वाधिक भाव मिळेल? (रु/क्विंटल)
- वाहतूक खर्च किती असेल?
- निव्वळ नफा किती होईल? (conservative ते optimistic range)
- आज विकावे की थांबावे?
- APMC कर/कमिशन किती कापला जाईल?
नक्कल एकच आकडा देऊ नका — नेहमी range द्या (कमी ते जास्त)."""

    return await _call_gemini(prompt)


async def format_fertilizer_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /fertilizer-info JSON into Marathi crop protection guide.
    Highlights: chemical names, brand names, dosage, waiting period.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'कीड/रोग माहिती'}"
सेवा: पीक संरक्षण आणि कीडनाशक सल्ला

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

यातून शेतकऱ्यासाठी step-by-step सल्ला द्या:
- कोणती कीड/रोग आढळली?
- कोणते कीडनाशक/बुरशीनाशक वापरायचे? (active ingredient + brand name)
- 15 लिटर पंपास किती डोस? किंवा एकरी किती?
- काढणीपूर्वी किती दिवस थांबायचे? (waiting period)
- कोणत्या ब्रँडची नावे महाराष्ट्रात मिळतात?
फक्त JSON मध्ये असलेली माहिती द्या — स्वतःहून डोस जोडू नका."""

    return await _call_gemini(prompt)


# ─── Main entry point ─────────────────────────────────────────────────────────

async def format_response_for_whatsapp(
    service_type: str,
    raw_data: dict[str, Any],
    original_user_text: str = "",
) -> str:
    """
    Main entry point — called by main.py after delivery.py returns endpoint data.

    service_type must match session_store.py values exactly:
        "weather"     → weather-risk endpoint response
        "mandi"       → mandi-optimize endpoint response
        "fertilizer"  → fertilizer-info endpoint response

    x402 errors have: {"error": True, "error_type": ..., "error_reason": ...}
    Endpoint no-match: {"status": "no_match", ...}
    Both are caught and return a clean Marathi error message.
    """
    # ── x402 client errors (error=True bool key) ──────────────────────────
    if not raw_data or raw_data.get("error") is True:
        error_type = raw_data.get("error_type", "UNKNOWN") if raw_data else "EMPTY_RESPONSE"
        error_reason = raw_data.get("error_reason", "") if raw_data else ""
        logger.warning(f"[Formatter] x402 error received: {error_type} — {error_reason}")

        if error_type == "DATA_UNAVAILABLE":
            return _error_response("सध्या डेटा उपलब्ध नाही. सर्व्हर काही वेळात पुन्हा चालू होईल.")
        if error_type in ("BAD_REQUEST", "CROP_NOT_FOUND"):
            return _error_response(f"चुकीची माहिती पाठवली गेली: {error_reason}")
        return _error_response("तांत्रिक अडचणीमुळे माहिती मिळवता आली नाही.")

    # ── Endpoint-level no_match / crop_not_found ──────────────────────────
    status = raw_data.get("status")
    if status in ("no_match", "crop_not_found", "missing_info"):
        msg = raw_data.get("message", "माहिती आढळली नाही.")
        return (
            f"⚠️ *माहिती आढळली नाही*\n\n{msg}\n\n"
            "पिकाचे नाव किंवा किडीचे नाव पुन्हा तपासून पाठवा. 🙏"
        )

    # ── Route to correct formatter ────────────────────────────────────────
    # service_type values match session_store.py exactly: "weather"/"mandi"/"fertilizer"
    if service_type == "weather":
        return await format_weather_response(raw_data, original_user_text)

    elif service_type == "mandi":
        return await format_mandi_response(raw_data, original_user_text)

    elif service_type == "fertilizer":
        return await format_fertilizer_response(raw_data, original_user_text)

    else:
        # Unknown service_type — generic fallback, should never happen in production
        logger.warning(f"[Formatter] Unknown service_type: '{service_type}' — using generic formatter")
        prompt = (
            f"शेतकऱ्यासाठी या कृषी माहितीचे मराठीत सोप्या भाषेत स्पष्टीकरण द्या:\n"
            f"{json.dumps(raw_data, ensure_ascii=False, indent=2)}"
        )
        return await _call_gemini(prompt)