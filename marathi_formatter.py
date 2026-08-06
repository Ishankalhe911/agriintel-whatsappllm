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

PROMPT ARCHITECTURE UPGRADES (ported from GPT-4.1-mini reference version):
    ✅ Fix 9:  System prompt is now rules-first, persona second — Gemini processes
              system_instruction top-down, so rules having priority prevents
              persona tone overriding structural constraints.
    ✅ Fix 10: Rule added — null/missing JSON fields are silently skipped, not
              printed as "not available" or "null".
    ✅ Fix 11: Each service prompt now has a JSON FIELD GUIDE before the data dump
              so Gemini understands what each field means before reading values.
              Weather alone has 50+ fields; without a guide, rare fields get skipped.
    ✅ Fix 12: Weather prompt — spray_window_ok, delta_t_c thresholds, and
              pest_pressure_index scale explained so model applies them correctly.
    ✅ Fix 13: Mandi prompt — field→meaning mapping + ordered response structure +
              data_freshness_hours staleness warning.
    ✅ Fix 14: Fertilizer prompt — layer_used confidence ladder, dosage contradiction
              fixed (only report dose if JSON has it), confidence=low flagging.
    ✅ Fix 15: temperature lowered 0.3 → 0.25 for tighter factual output.
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

_SYSTEM_PROMPT = """WHATSAPP FORMATTING RULES — follow these exactly:
1. Use only Marathi language (Devanagari script) in your response.
2. Bold: use *single asterisk* only — never **double**, never # headers.
3. Lists: use - or • bullet characters only.
4. NEVER print JSON keys, field names, null, undefined, or database identifiers.
5. NEVER invent data not present in the JSON — no guessed prices, doses, or dates.
6. Use relevant emojis: 🌾 🌧️ 🐛 💰 🚜 💡 🧪 ⚠️ ✅ ☀️ 🌱
7. Keep paragraphs short — WhatsApp mobile screen, not a desktop browser.
8. End every response with an encouraging sign-off line.
9. If a JSON field is null or missing, skip it entirely — do not say "not available".

तुम्ही एक अनुभवी महाराष्ट्रीयन कृषी तज्ञ आहात जे WhatsApp वर शेतकऱ्यांना
सोप्या, उपयुक्त भाषेत सल्ला देता. तुमचा सल्ला नेहमी JSON मधील डेटावर आधारित
असतो — तुम्ही कधीही स्वतःहून माहिती जोडत नाही."""

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
                temperature=0.25,       # tighter for factual precision (JSON data)
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

CRITICAL DATE ANCHOR — JSON वाचण्यापूर्वी हे वाचा:
- आजची तारीख: {today_iso} ({today_human}). वेळ: {current_time}.
- daily_preview मधील सर्व entries FORECAST आहेत — {today_iso} सुद्धा.
  कोणताही entry historical नाही. {today_iso} entry skip करू नका.
- Date mapping:
    "आज"   → {today_iso}
    "उद्या" → JSON मधील {today_iso} नंतरची पुढची तारीख
    "परवा" → त्यानंतरची तारीख
- JSON मध्ये YYYY-MM-DD format आहे. शेतकऱ्यासाठी "६ ऑगस्ट" सारख्या नैसर्गिक
  मराठी format मध्ये सांगा.

JSON FIELD GUIDE — हे fields काय सांगतात (JSON वाचण्यापूर्वी समजून घ्या):
- daily_preview[].date: YYYY-MM-DD तारीख
- rain_mm: त्या दिवशी किती पाऊस पडेल (mm)
- t_max_c / t_min_c: कमाल/किमान तापमान (°C)
- rh_max_pct / rh_min_pct: आर्द्रता श्रेणी (%)
- wind_kmh: वाऱ्याचा वेग (kmh)
- wcode: हवामान कोड (95=गडगडाट, 51=हलका पाऊस, 0/1=स्वच्छ)
- spray_window_ok=true → फवारणीसाठी योग्य दिवस; false → फवारणी टाळा
- delta_t_c: 2-8°C = आदर्श फवारणी श्रेणी; <2 = inversion risk; >8 = खूप गरम/कोरडे
- pest_pressure_index: 0-3=कमी धोका, 4-6=मध्यम, 7-10=जास्त धोका ⚠️
- gdd_base10: पीक वाढीचे मोजमाप (Growth Degree Days)
- next_rain_date: पुढचा पाऊस कधी येईल
- next_dry_spell: कोरड्या दिवसांचा कालावधी (सिंचनासाठी उपयुक्त)
- optimal_drone_spray_dates: ड्रोन फवारणीसाठी सर्वोत्तम दिवस
- wind_risk_days: जास्त वारा — फवारणी drift होण्याचा धोका
- pest_disease_risk_windows: रोग/कीड येण्याची शक्यता असलेले दिवस
- irrigation_recommended: सिंचन करायचे का (true/false)
- net_water_balance_7d: पाऊस - ET0 = पाण्याचा ताळेबंद (negative = दुष्काळ)
- subseasonal: 3-4 आठवड्यांचा ECMWF दीर्घकालीन अंदाज
- enso_iod_state: monsoon वर परिणाम करणाऱ्या हवामान शक्ती

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

MARATHI RESPONSE — मराठीत उत्तर द्या:
१. शेतकऱ्याने जो प्रश्न विचारला त्याचे थेट उत्तर द्या.
   - फक्त "परवा" विचारले → फक्त त्या दिवसाची माहिती द्या.
   - सामान्य प्रश्न → पुढील 3-5 दिवसांचा trend.
२. spray_window_ok=true असलेले दिवस स्पष्टपणे सांगा (फवारणी योग्य दिवस).
३. pest_pressure_index ≥7 असेल तर ⚠️ सावधानता द्या.
४. irrigation_recommended=true असेल तर सिंचनाचा सल्ला द्या.
५. फक्त JSON मधील डेटा वापरा — अंदाज किंवा सामान्य ज्ञान नको."""

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

JSON FIELD GUIDE — हे fields काय सांगतात (JSON वाचण्यापूर्वी समजून घ्या):
- mandis[]: जवळच्या मंडींची यादी, best first
- mandi_name / district: मंडीचे नाव आणि जिल्हा
- distance_km: शेतकऱ्याच्या ठिकाणापासून अंतर (km)
- modal_price_per_quintal: आजचा मुख्य बाजार भाव (रु/क्विंटल)
- min_price / max_price: त्या मंडीत आज किती कमी-जास्त मिळाला
- transport_cost_est: घरापासून त्या मंडीपर्यंत वाहतूक खर्च (रु, अंदाजे)
- apmc_commission_pct: APMC कमिशन टक्केवारी
- net_profit_per_quintal_est: भाव - वाहतूक - कमिशन = निव्वळ नफा (रु/क्विंटल)
- arrival_trend: "rising"=भाव वाढत आहेत, "falling"=कमी होत आहेत, "stable"=स्थिर
- qty_quintals: शेतकऱ्याचे एकूण क्विंटल (total profit साठी वापरा)
- data_freshness_hours: हा डेटा किती तासांपूर्वीचा आहे

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

MARATHI RESPONSE — या क्रमाने सांगा:
१. *सर्वोत्तम मंडी* — net_profit_per_quintal_est सर्वाधिक असलेली मंडी.
   नाव, जिल्हा, अंतर, आणि modal_price_per_quintal सांगा.
२. *भाव range* — त्या मंडीत आज min_price ते max_price किती मिळाला?
३. *वाहतूक आणि APMC* — transport_cost_est आणि apmc_commission_pct किती?
४. *निव्वळ नफा* — qty_quintals असेल तर total profit range सांगा.
   (conservative = min_price वापरा; optimistic = max_price वापरा)
५. *भाव कल* — arrival_trend वरून आज विकावे का थांबावे?
६. data_freshness_hours >24 असेल तर ⚠️ "डेटा जुना आहे, मंडीत जाण्यापूर्वी भाव तपासा" सांगा.

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
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'कीड/रोग माहिती सांगा'}"

JSON FIELD GUIDE — हे fields काय सांगतात (JSON वाचण्यापूर्वी समजून घ्या):
- pest_identified / disease_identified: कोणती कीड/रोग आढळली
- recommendations[]: शिफारशींची यादी (एक किंवा अधिक)
- active_ingredient: रासायनिक घटकाचे शास्त्रीय नाव
- category: insecticide=कीडनाशक, fungicide=बुरशीनाशक, herbicide=तणनाशक,
            bio-pesticide=जैविक कीडनाशक, PGR=वाढ नियंत्रक
- dose_per_15L: 15 लिटर पंपास किती ml/g घालायचे
- dose_per_acre: एकरी किती द्यायचे
- waiting_period_days: काढणीपूर्वी किती दिवस थांबायचे (Pre-Harvest Interval)
- brand_names[]: दुकानात कोणत्या नावाने मागायचे
- confidence: high=खात्रीशीर माहिती; medium=साधारण; low=अंदाजे — flag करा
- layer_used: 1=CIBRC सरकारी यादी (सर्वात विश्वासार्ह), 2=कृषी विद्यापीठ,
              3=Search, 4=AI अंदाज (कमी विश्वासार्ह — नेहमी flag करा)
- warning: कोणतीही महत्त्वाची सावधानता

JSON डेटा:
{json.dumps(data, ensure_ascii=False, indent=2)}

MARATHI RESPONSE — या क्रमाने सांगा:
१. *कोणती कीड/रोग* — pest_identified किंवा disease_identified सांगा.
२. *उपाय* — recommendations[] मधून प्रत्येक शिफारशीसाठी:
   - active_ingredient + category (मराठीत: कीडनाशक/बुरशीनाशक/इ.)
   - brand_names[] मधून 2-3 नावे ("दुकानात ___ नावाने मागा")
   - DOSAGE RULE: dose_per_15L किंवा dose_per_acre JSON मध्ये असेल तरच सांगा.
     नसेल तर: "डोससाठी कृषी सेवा केंद्राला विचारा." — स्वतःहून डोस कधीही सांगू नका.
   - waiting_period_days असेल तर: "काढणीपूर्वी ___ दिवस थांबा"
३. confidence=low किंवा layer_used=4 असेल तर ⚠️ सांगा:
   "ही माहिती अंदाजे आहे — कृपया कृषी विभागाकडून खात्री करा."
४. warning field असेल तर ते स्पष्टपणे सांगा.

महत्त्वाचे: JSON मध्ये dose नसेल तर स्वतःहून कधीही डोस सांगू नका."""

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