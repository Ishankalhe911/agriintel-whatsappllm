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
from datetime import datetime
import zoneinfo

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

MODEL = "gemini-3.5-flash-lite"


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

_SYSTEM_PROMPT = """तुम्ही एक अनुभवी महाराष्ट्रीयन कृषी तज्ञ आहात जे शेतकऱ्यांना WhatsApp वर सल्ला देता.
शेतकऱ्याने पैसे देऊन हा सल्ला विकत घेतला आहे — त्यामुळे उत्तर सखोल, आत्मविश्वासपूर्ण आणि
JSON मधील प्रत्येक उपयुक्त आकडा वापरून द्या. "थोडक्यात" सांगू नका — पूर्ण माहिती द्या.

FORMATTING RULES (काटेकोरपणे पाळा):
1. फक्त मराठी भाषा (Devanagari script).
2. Bold: फक्त *एकच asterisk* — **double** नाही, # header नाही.
3. Lists: फक्त - किंवा • वापरा.
4. JSON field names, null, undefined कधीही print करू नका.
5. JSON मध्ये नसलेली माहिती — किंमत, डोस, तारीख — कधीही स्वतःहून जोडू नका.
6. Emojis: 🌾 🌧️ 🐛 💰 🚜 💡 🧪 ⚠️ ✅ ☀️ 🌱
7. Mobile screen साठी छोटे paragraphs — प्रत्येक मुद्दा नवीन line वर.
8. null किंवा missing field असेल तर तो section पूर्णपणे skip करा — "उपलब्ध नाही" लिहू नका.
9. शेवटी एक उत्साहवर्धक मराठी वाक्य द्या."""

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
                temperature=0.25,       # tight for factual JSON data
                max_output_tokens=3000, # farmer paid — give full detail
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
    Uses today's date as a reference and dynamically shapes the response 
    based on the farmer's specific intent.
    """
    # Grab current IST date and time as a strict calendar reference
    ist_zone = zoneinfo.ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(ist_zone)
    # Keep YYYY-MM-DD format so it matches the JSON date keys exactly —
    # mixing "06 August 2026" (human) with "2026-08-06" (JSON) caused
    # Gemini to misalign dates by one day.
    today_iso  = now_ist.strftime("%Y-%m-%d")          # e.g. 2026-08-06
    today_human = now_ist.strftime("%d %B %Y (%A)")    # e.g. 06 August 2026 (Wednesday)
    current_time = now_ist.strftime("%I:%M %p IST")

    prompt = f"""शेतकऱ्याचा प्रश्न: "{original_text or 'पुढील हवामान कसे राहील?'}"

━━━ DATE ANCHOR — आधी वाचा, मग JSON वाचा ━━━
- आजची तारीख: {today_iso} ({today_human}). वेळ: {current_time}.
- daily_preview मधील प्रत्येक entry FORECAST आहे — {today_iso} सुद्धा. कोणताही entry historical नाही.
- {today_iso} ची entry SKIP करू नका — ती आजचीच आहे.
- Date mapping:
    "आज" → {today_iso}  |  "उद्या" → {today_iso} नंतरची पुढची date  |  "परवा" → त्यानंतरची
- Farmer ला date सांगताना: YYYY-MM-DD → "६ ऑगस्ट" असे नैसर्गिक मराठीत सांगा.

━━━ FIELD MEANINGS (JSON वाचण्यापूर्वी समजून घ्या) ━━━
rain_mm: पाऊस मिमी | t_max_c/t_min_c: कमाल/किमान °C | rh_max_pct: आर्द्रता %
wind_kmh: वाऱ्याचा वेग | wcode: 0-2=स्वच्छ, 51-55=हलका पाऊस, 61-65=मध्यम पाऊस, 95+=गडगडाट
wind_risk_days: जास्त वारा — फवारणी drift होईल | pest_disease_risk_windows: रोग/कीड येण्याची शक्यता
next_rain_date: पुढचा पाऊस कधी | next_dry_spell: कोरडे दिवस (सिंचनाची योग्य वेळ)
optimal_drone_spray_dates: ड्रोन फवारणीसाठी सर्वोत्तम दिवस
net_water_balance_7d: negative = पाण्याची कमतरता → सिंचन करा
irrigation_recommended: true = आत्ता सिंचन करा | crop_stress_risk_level: LOW/MEDIUM/HIGH
rainfall_total_mm: एकूण पाऊस | et0_7d_mm: पाण्याची बाष्पीभवनातून गरज

━━━ HOW TO ANSWER ━━━
शेतकऱ्याचा प्रश्न काळजीपूर्वक वाचा आणि त्यानुसार उत्तर द्या:

• फवारणी बद्दल विचारले (आज/उद्या/परवा) →
  त्या दिवसाचा rain_mm, wind_kmh, rh_max_pct सांगा.
  wind_risk_days मध्ये तो दिवस असेल तर ⚠️ "फवारणी टाळा" सांगा.
  pest_disease_risk_windows असेल तर त्याचा उल्लेख करा.

• पाऊस कधी येईल →
  next_rain_date सांगा. daily_preview मधून प्रत्येक दिवसाचा rain_mm द्या.
  rainfall_total_mm आणि net_water_balance_7d सांगा.

• सामान्य हवामान / पुढील काही दिवस →
  daily_preview मधील प्रत्येक दिवस एक bullet मध्ये: तारीख, rain_mm, t_max_c, wind_kmh.
  नंतर: next_dry_spell (सिंचन करा), wind_risk_days, pest_disease_risk_windows.
  crop_stress_risk_level HIGH असेल तर ⚠️ द्या.
  irrigation_recommended=true असेल तर स्पष्ट सांगा.

━━━ OUTPUT EXAMPLE (हे format वापरा, actual values JSON मधून घ्या) ━━━
✅ *[पिकाचे नाव] हवामान अंदाज — [तारीख range]*

🌧️ *पुढील [N] दिवस:*
- [तारीख]: [rain_mm] मिमी पाऊस, कमाल [t_max_c]°C, वारा [wind_kmh] kmh
- [तारीख]: [rain_mm] मिमी पाऊस, कमाल [t_max_c]°C, वारा [wind_kmh] kmh
[...सर्व दिवस...]

💧 *पाण्याचा ताळेबंद:* [net_water_balance_7d] मिमी — [positive=चांगले/negative=सिंचन करा]
🌱 *एकूण पाऊस (७ दिवस):* [rainfall_total_mm] मिमी

🚜 *फवारणी:*
- ✅ योग्य दिवस: [optimal_drone_spray_dates]
- ⚠️ टाळा: [wind_risk_days] — जास्त वारा आहे

🐛 *रोग/कीड धोका:* [pest_disease_risk_windows] — [crop_stress_risk_level]

[irrigation_recommended=true असेल तर:]
💡 *सिंचन:* होय — [next_dry_spell.start_date] ते [next_dry_spell.end_date] दरम्यान करा

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL RULES ━━━
- JSON मध्ये नसलेले field skip करा — "उपलब्ध नाही" लिहू नका
- Dates नेहमी "६ ऑगस्ट" format मध्ये सांगा
- प्रत्येक relevant आकडा द्या — farmer ने पैसे दिले आहेत, थोडक्यात नको"""

    return await _call_gemini(prompt)

async def format_mandi_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /mandi-optimize JSON into Marathi market advisory.
    Highlights: best mandis, prices, transport cost, net profit range.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'मंडई भाव सांगा'}"

━━━ FIELD MEANINGS ━━━
mandis[]: जवळच्या मंडींची यादी (best first)
mandi_name/district: मंडीचे नाव व जिल्हा | distance_km: अंतर km
modal_price_per_quintal: आजचा मुख्य भाव रु/क्विंटल
min_price/max_price: आजचे किमान-कमाल भाव
transport_cost_est: वाहतूक खर्च (रु, अंदाजे)
apmc_commission_pct: APMC कमिशन % | net_profit_per_quintal_est: निव्वळ नफा रु/क्विंटल
arrival_trend: rising=भाव वाढतोय, falling=कमी होतोय, stable=स्थिर
qty_quintals: शेतकऱ्याचे एकूण क्विंटल | data_freshness_hours: डेटा किती तासांपूर्वीचा

━━━ OUTPUT EXAMPLE (हे format वापरा, actual values JSON मधून घ्या) ━━━
✅ *[पीक] मंडी भाव विश्लेषण*

🏆 *सर्वोत्तम मंडी: [mandi_name], [district]*
- अंतर: [distance_km] km
- आजचा भाव: ₹[modal_price_per_quintal] प्रति क्विंटल
- भाव range: ₹[min_price] ते ₹[max_price]
- वाहतूक खर्च: ₹[transport_cost_est]
- APMC कमिशन: [apmc_commission_pct]%
- *निव्वळ नफा: ₹[net_profit_per_quintal_est] प्रति क्विंटल*

💰 *[qty_quintals] क्विंटलसाठी एकूण नफा:*
- Conservative (₹[min_price] भावाने): ₹[min_price × qty - transport - commission] अंदाजे
- Optimistic (₹[max_price] भावाने): ₹[max_price × qty - transport - commission] अंदाजे

[जर दुसरी मंडी असेल तर:]
📍 *पर्यायी मंडी: [mandi_name 2], [district]*
- भाव: ₹[modal_price] | अंतर: [distance_km] km | नफा: ₹[net_profit_per_quintal_est]

📈 *भाव कल:* [arrival_trend → rising=वाढत आहे, थांबा / falling=कमी होतोय, आज विका / stable=स्थिर]
💡 *सल्ला:* [arrival_trend वरून आज विकायचे की थांबायचे ते स्पष्ट सांगा]

[data_freshness_hours > 24 असेल तर:]
⚠️ हा डेटा [data_freshness_hours] तासांपूर्वीचा आहे — मंडीत जाण्यापूर्वी भाव तपासा.

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL RULES ━━━
- Conservative profit = min_price वापरून calculate करा
- Optimistic profit = max_price वापरून calculate करा
- नेहमी range द्या — एकच आकडा नको
- JSON मध्ये नसलेले field skip करा"""

    return await _call_gemini(prompt)


async def format_fertilizer_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /fertilizer-info JSON into Marathi crop protection guide.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'कीड/रोग माहिती'}"

━━━ FIELD MEANINGS ━━━
pest_identified / disease_identified: आढळलेली कीड किंवा रोग
active_ingredient: रासायनिक घटकाचे नाव | category: कीडनाशक/बुरशीनाशक/जैविक
brand_names[]: बाजारात मिळणाऱ्या औषधांची नावे
dose_per_15L: १५ लिटर पंपासाठी डोस | dose_per_acre: एकरी डोस
waiting_period_days: फवारणीनंतर पीक काढणीसाठी किती दिवस थांबावे
confidence: high=खात्रीशीर, medium=साधारण, low=अंदाजे

━━━ OUTPUT EXAMPLE (हे format वापरा) ━━━
✅ *पीक संरक्षण सल्ला*

🐛 *आढळलेली समस्या:* [pest_identified / disease_identified]

🧪 *शिफारस केलेले औषध:*
- *घटक:* [active_ingredient] ([category])
- *बाजारातील नावे:* [brand_names - २-३ नावे द्या]
- *डोस:* [dose_per_15L] प्रति १५ लिटर पंप / [dose_per_acre] एकरी
- *काढणीपूर्वी थांबा:* [waiting_period_days] दिवस

[जर confidence 'low' असेल तर:]
⚠️ *टीप:* ही माहिती अंदाजे आहे, कृपया कृषी सेवा केंद्रात खात्री करा.

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL RULES ━━━
- JSON मध्ये डोस (dose) दिलेला नसेल, तर स्वतःच्या मनाने डोस सांगू नका. "कृषी सेवा केंद्रात विचारा" असे लिहा.
- OUTPUT EXAMPLE मधील brackets [ ] छापू नका, फक्त त्या जागी JSON मधील माहिती भरा."""

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
