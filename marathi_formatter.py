"""
marathi_formatter.py
─────────────────────
Formatting layer for AgriIntel WhatsApp backend.

MODEL: gemini-3.1-flash-lite (google-genai SDK)
Kept on Gemini for throughput — lower time-to-first-token than GPT-4.1 mini
at this scale, and Gemini 3.1 Flash-Lite scores higher on general intelligence
benchmarks (AA Index 25 vs 15 for GPT-4.1 mini).

PROMPT ARCHITECTURE (this version):
  - System prompt: rules-first (formatting), then persona — GPT-style ordering
    that also works well for Gemini instruction following
  - Each service prompt has THREE sections before the JSON dump:
      1. STRUCTURE GUIDE  → how the JSON is nested (top-level keys)
      2. FIELD MEANINGS   → what each leaf field actually means
      3. HOW TO ANSWER    → branching logic for different farmer questions
  - Field names in all guides are verified against the actual endpoint source
  - Mandi profit math is dimensionally correct (pct commission handled properly)
  - Fertilizer dosage disclaimer is unit-agnostic (works for dose_per_ha AND dose_per_15L)
  - Weather field guide matches actual /weather-risk output schema exactly

FIXES from previous version:
  ✅ Weather field guide now matches actual nested JSON (horizon_1_forecast, etc.)
  ✅ daily_preview field names corrected (was using wrong guessed names)
  ✅ Mandi profit calculation instruction is dimensionally correct
  ✅ Fertilizer PHI disclaimer is unit-agnostic
  ✅ max_output_tokens = 3000 (consistent — farmer paid, give full detail)
  ✅ temperature = 0.25 (tighter than 0.3 for factual JSON extraction)
  ✅ System prompt is rules-first for stronger instruction compliance
"""

import json
import logging
import os
import random
from typing import Any
from datetime import datetime
import zoneinfo

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── Key rotation ─────────────────────────────────────────────────────────────

_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]

MODEL = "gemini-3.1-flash-lite"


def _get_client() -> genai.Client:
    keys = [k for k in _GEMINI_KEYS if k]
    if not keys:
        raise ValueError("No GEMINI_API_KEY_* env vars set — need at least one")
    return genai.Client(api_key=random.choice(keys))


# ─── System prompt: RULES FIRST, then persona ────────────────────────────────
# Rules-first ordering matters: the model processes beginning of system prompt
# with highest priority. Persona second so it sets tone without overriding rules.

_SYSTEM_PROMPT = """WHATSAPP FORMATTING RULES — follow exactly, no exceptions:
1. फक्त मराठी भाषा (Devanagari script) — no English words in output.
2. Bold: *एकच asterisk* — never **double**, never # headers.
3. Lists: - किंवा • bullet characters only.
4. NEVER print JSON key names, field names, null, undefined, or "N/A".
5. NEVER invent data not in JSON — no guessed prices, doses, or dates.
6. Skip null/missing fields entirely — do not say "उपलब्ध नाही".
7. Emojis: 🌾 🌧️ 🐛 💰 🚜 💡 🧪 ⚠️ ✅ ☀️ 🌱 (use contextually)
8. Mobile paragraphs — short, each point on its own line.
9. End every response with one encouraging Marathi sign-off line.

तुम्ही एक अनुभवी महाराष्ट्रीयन कृषी तज्ञ आहात. शेतकऱ्याने पैसे देऊन हा सल्ला
विकत घेतला आहे — त्यामुळे JSON मधील प्रत्येक उपयुक्त आकडा वापरून पूर्ण उत्तर द्या."""


# ─── Error builder ────────────────────────────────────────────────────────────

def _error_response(reason: str) -> str:
    return (
        f"⚠️ *माहिती उपलब्ध नाही*\n\n{reason}\n\n"
        "कृपया थोड्या वेळाने पुन्हा प्रयत्न करा. 🙏"
    )


# ─── Core async Gemini call ───────────────────────────────────────────────────

async def _call_gemini(user_prompt: str) -> str:
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.25,
                max_output_tokens=3000,
            ),
        )
        return response.text.strip()
    except ValueError as e:
        logger.error(f"[Formatter] Config error: {e}")
        return _error_response("तांत्रिक अडचण: API key उपलब्ध नाही.")
    except Exception as e:
        logger.error(f"[Formatter] Gemini call failed: {e}")
        return _error_response("माहिती तयार करताना अडचण आली.")


# ─── WEATHER formatter ───────────────────────────────────────────────────────

async def format_weather_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    ist_zone = zoneinfo.ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(ist_zone)
    today_iso   = now_ist.strftime("%Y-%m-%d")
    today_human = now_ist.strftime("%d %B %Y (%A)")
    current_time = now_ist.strftime("%I:%M %p IST")

    prompt = f"""शेतकऱ्याचा प्रश्न: "{original_text or 'पुढील हवामान कसे राहील?'}"

━━━ DATE ANCHOR — JSON वाचण्यापूर्वी हे वाचा ━━━
आजची तारीख: {today_iso} ({today_human}). वेळ: {current_time}.
daily_preview मधील प्रत्येक entry FORECAST आहे — {today_iso} सुद्धा.
{today_iso} entry skip करू नका.
Date mapping: "आज"→{today_iso} | "उद्या"→पुढची date | "परवा"→त्यानंतरची
शेतकऱ्याला date सांगताना: YYYY-MM-DD → "७ ऑगस्ट" असे मराठीत.

━━━ JSON STRUCTURE (हे top-level keys आहेत) ━━━
horizon_1_forecast   → मुख्य 0-16 दिवसांचा अंदाज (हे नेहमी असते)
horizon_2_subseasonal → ECMWF 3-4 आठवड्यांचा अंदाज (harvest_date दिल्यास येतो)
horizon_3_seasonal   → NASA POWER दीर्घकालीन (harvest_date दिल्यास येतो)
enso_iod_state       → ENSO/IOD मान्सून स्थिती
days_to_harvest      → काढणीपर्यंत दिवस
partial_data         → true असेल तर horizon_2/3 चा डेटा अपूर्ण आहे

━━━ FIELD MEANINGS — horizon_1_forecast च्या आत ━━━
crop_stress_risk_level      → LOW/MEDIUM/HIGH — पिकावरचा एकूण ताण
crop_stress_factors[]       → कशामुळे ताण आहे (उष्णता, पाण्याची कमतरता, इ.)
operational_risk_level      → LOW/MEDIUM/HIGH — शेत कामांना अडचण
operational_factors[]       → कोणती कामे करता येणार नाहीत
next_rain_date              → पुढचा पाऊस कधी (YYYY-MM-DD)
next_dry_spell              → {{ start_date, end_date, days }} — कोरड्या दिवसांचा काळ
optimal_drone_spray_dates[] → ड्रोन फवारणीसाठी सर्वोत्तम दिवस (Delta-T आधारित)
pest_disease_risk_windows[] → कीड/रोगाचा धोका कधी आहे
wind_risk_days[]            → जास्त वाऱ्याचे दिवस — फवारणी टाळा
heavy_rain_days[]           → जड पावसाचे दिवस (day index, 0=आज)
heat_stress_days[]          → उष्णतेचे दिवस (day index)
gdd_accumulated_forecast_window → एकूण GDD — पीक वाढीचे मोजमाप
growth_stage                → पिकाची अवस्था (sowing_date असल्यास)
irrigation_recommended      → true = सिंचन करा
net_water_balance_7d        → positive=पाणी जास्त, negative=कमतरता (mm)
daily_preview[]             → रोजचा तपशील (date, rain_mm, temp_max_c,
                              temp_min_c, wind_speed_kmh, humidity_pct,
                              spray_window_ok, pest_pressure_index,
                              gdd_base10, delta_t_c)

━━━ FIELD MEANINGS — horizon_2_subseasonal ━━━
weekly_outlook[] → {{ week, dates, projected_rain_mm, trend }}
trend: "above_normal"=जास्त पाऊस | "below_normal"=कमी | "near_normal"=सामान्य

━━━ FIELD MEANINGS — enso_iod_state ━━━
oni_phase: el_nino=कमी पाऊस शक्य | la_nina=जास्त पाऊस शक्य | neutral=सामान्य
dmi_phase: positive_iod=कमी | negative_iod=जास्त | neutral=सामान्य

━━━ KEY THRESHOLDS (interpret करताना वापरा) ━━━
spray_window_ok = true → फवारणीसाठी योग्य दिवस
delta_t_c: 2-8°C = ideal spray | <2 = inversion risk | >8 = खूप गरम
pest_pressure_index: 0-3=कमी | 4-6=मध्यम | 7-10=जास्त धोका ⚠️
net_water_balance_7d negative → irrigation_recommended पहा

━━━ HOW TO ANSWER (शेतकऱ्याचा प्रश्न वाचा, त्यानुसार उत्तर द्या) ━━━

फवारणी बद्दल विचारले (आज/उद्या/परवा) →
  त्या दिवसाचा daily_preview मधून: rain_mm, wind_speed_kmh, spray_window_ok.
  wind_risk_days मध्ये तो दिवस असेल → ⚠️ फवारणी टाळा.
  optimal_drone_spray_dates सांगा.
  pest_disease_risk_windows असेल तर mention करा.

पाऊस बद्दल विचारले →
  next_rain_date सांगा.
  daily_preview मधून प्रत्येक दिवसाचा rain_mm सांगा.
  net_water_balance_7d सांगा.

सामान्य हवामान / पुढील काही दिवस →
  daily_preview प्रत्येक दिवस: date→मराठी, rain_mm, temp_max_c, wind_speed_kmh.
  crop_stress_risk_level + factors सांगा.
  irrigation_recommended=true असेल तर स्पष्ट सांगा.
  next_dry_spell असेल तर सांगा (सिंचनाची/फवारणीची योग्य वेळ).
  horizon_2 असेल तर weekly_outlook trend सांगा.

━━━ OUTPUT FORMAT ━━━
*[पीक] हवामान अंदाज* 🌾

🌧️ *पुढील दिवस:*
- [मराठी date]: [rain_mm] मिमी पाऊस, [temp_max_c]°C, वारा [wind_speed_kmh] km/h [spray_window_ok=true→✅फवारणी योग्य]

💧 *पाण्याचा ताळेबंद (७ दिवस):* [net_water_balance_7d] मिमी
[negative असेल:] 💡 सिंचन करण्याची वेळ आली आहे.

🚜 *फवारणीसाठी योग्य दिवस:* [optimal_drone_spray_dates]
⚠️ *टाळा:* [wind_risk_days] — जास्त वारा

🐛 *कीड/रोग धोका:* [pest_disease_risk_windows] | पीक ताण: [crop_stress_risk_level]
[crop_stress_factors असतील:] कारण: [factors]

[horizon_2 असेल तर:]
📅 *पुढील महिन्याचा अंदाज (ECMWF):*
[weekly_outlook प्रत्येक week: dates + projected_rain_mm + trend मराठीत]

[enso_iod_state असेल तर:]
🌏 *मान्सून स्थिती:* [oni_phase + dmi_phase मराठीत]

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ FINAL RULES ━━━
- JSON key नावे कधीही छापू नका (horizon_1_forecast, daily_preview, etc.)
- Dates नेहमी "७ ऑगस्ट" format — YYYY-MM-DD कधीही नको
- partial_data=true असेल तर सांगा: "काही दीर्घकालीन डेटा उपलब्ध नाही"
- JSON मध्ये नसलेले field/section पूर्णपणे skip करा"""

    return await _call_gemini(prompt)


# ─── MANDI formatter ──────────────────────────────────────────────────────────

async def format_mandi_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    """
    Formats /mandi-optimize JSON into Marathi market advisory.
    Acts as an advanced Ag-Economist LLM router: handles both 'price_only' 
    and 'full_optimization' modes while executing dynamic agent rules.
    """
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'मंडई भाव सांगा'}"

━━━ SYSTEM PERSONA & MISSION ━━━
तुम्ही एक अत्यंत हुशार कृषी-अर्थतज्ञ (Ag-Economist) आहात. 
तुमचे काम शेतकऱ्याला त्याच्या मालासाठी सर्वोत्तम बाजारपेठ शोधून देणे आणि 
'agent_execution_rules' मध्ये दिलेल्या सूचनांचे काटेकोरपणे पालन करणे आहे.

━━━ JSON STRUCTURE (नवीन Schema) ━━━
mode                     → "price_only" किंवा "full_optimization"
qty_quintals             → शेतकऱ्याचा माल (क्विंटलमध्ये - profit calculation साठी)
nearest_mandi            → Baseline (सर्वात जवळची मंडी)
top_mandis[]             → सर्वाधिक फायदेशीर मंडींची यादी (Arbitrage)
agent_execution_rules    → शेतकऱ्यासाठी महत्त्वाच्या सूचना (Checklist आणि Warnings)

━━━ FIELD MEANINGS (Nested Data) ━━━
market                                     → मंडीचे नाव
exact_scraped_data.modal_price_per_quintal → आजचा मुख्य भाव (रु/क्विंटल)
exact_scraped_data.min_price / max_price   → किमान आणि कमाल भाव (असल्यास)
exact_scraped_data.variety                 → वाण (Variety)
driving_distance.value_km                  → अंतर (km)
transport_cost_est                         → वाहतूक खर्च (असल्यास)
apmc_commission_pct                        → APMC कमिशन % (असल्यास)
arrival_trend                              → rising=भाव वाढतोय, falling=कमी होतोय, stable=स्थिर

━━━ PROFIT CALCULATION ENGINE (फक्त जर mode="full_optimization" असेल तरच वापरा) ━━━
जर JSON मध्ये qty_quintals, min_price, max_price आणि transport_cost_est दिलेले असतील, तर हे गणित करा:
- Conservative total profit: (min_price × qty_quintals) - (Gross × apmc_commission_pct/100) - transport_cost_est
- Optimistic total profit: (max_price × qty_quintals) - (Gross × apmc_commission_pct/100) - transport_cost_est
(जर हे आकडे JSON मध्ये नसतील, तर गणिताचा भाग पूर्णपणे सोडून द्या.)

━━━ HOW TO ANSWER (AGENT RULES) ━━━
१. Presentation Rule: नेहमी आधी 'nearest_mandi' ची माहिती द्या (शेतकऱ्याचा विश्वास जिंकण्यासाठी). त्यानंतर 'top_mandis' मधील इतर जास्त फायद्याच्या मंड्या सांगा.
२. जर mode="price_only" असेल: नफ्याचे कोणतेही आकडे किंवा वाहतूक खर्च स्वतःच्या मनाने लिहू नका. फक्त भाव आणि अंतर सांगा.
३. जर mode="full_optimization" असेल: वरील Calculation Engine वापरून एकूण नफ्याची (Profit) Range सांगा.
४. 'pre_dispatch_checklist_to_show_user' मधील मुद्दे बुलेट पॉईंट्स (-) मध्ये उत्तम मराठीत भाषांतरित करा.
५. 'variety_warning_to_show_user' मधील टीप शेवटी ठळक अक्षरात द्या.

━━━ OUTPUT FORMAT (याच साच्यात उत्तर द्या) ━━━
✅ *[crop] मंडी भाव विश्लेषण* 💰

📍 *तुमची जवळची मंडी (Baseline): [nearest_mandi.market]*
- अंतर: [driving_distance.value_km] km
- आजचा मुख्य दर: ₹[modal_price_per_quintal] प्रति क्विंटल
[जर min/max price असेल:] - दर range: ₹[min_price] ते ₹[max_price]
- वाण/प्रकार: [variety]

🏆 *इतर फायदेशीर मंड्या (Arbitrage):*
- *[top_mandis मधील market 1]:* ₹[modal_price_per_quintal] | अंतर: [value_km] km | वाण: [variety]
- *[top_mandis मधील market 2]:* ₹[modal_price_per_quintal] | अंतर: [value_km] km | वाण: [variety]

[जर mode="full_optimization" आणि qty_quintals असेल तरच खालील भाग दाखवा:]
💰 *[qty_quintals] क्विंटलसाठी एकूण नफ्याचे गणित (अंदाजे):*
- वाहतूक खर्च: ₹[transport_cost_est]
- Conservative नफा: ₹[calculated conservative net] 
- Optimistic नफा: ₹[calculated optimistic net] 

[जर arrival_trend असेल:]
📈 *भाव कल:* [arrival_trend मराठीत - उदा. 'भाव वाढत आहेत, थोडा वेळ थांबणे फायदेशीर ठरू शकते.']

📝 *माल पाठवण्यापूर्वीची तयारी:*
- [checklist item 1 - in Marathi]
- [checklist item 2 - in Marathi]
- [checklist item 3 - in Marathi]

💡 *महत्त्वाची टीप:* [variety_warning_to_show_user - in Marathi]

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL ANTI-HALLUCINATION RULES ━━━
- JSON मधील top_mandis मधीलच शहरे घ्या — स्वतःच्या मनाने (उदा. गडचिरोली, गोंदिया, रोहा) कोणतीही इतर शहरे जोडू नका.
- JSON मध्ये min_price, max_price किंवा qty_quintals नसेल, तर नफ्याचे कोणतेही गणित स्वतःहून करू नका.
- JSON key नावे (उदा. exact_scraped_data, driving_distance) कधीही output मध्ये छापू नका.
- OUTPUT FORMAT मधील [ ] brackets output मध्ये छापू नका, त्या जागी माहिती भरा."""

    return await _call_gemini(prompt)

# ─── FERTILIZER formatter ─────────────────────────────────────────────────────

async def format_fertilizer_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'कीड/रोग माहिती'}"

━━━ JSON STRUCTURE ━━━
pest_identified / disease_identified → आढळलेली कीड किंवा रोग
recommendations[]                   → शिफारशींची यादी
warning                             → महत्त्वाची सावधानता (असल्यास)
layer_used                          → माहिती कोठून आली

━━━ FIELD MEANINGS — recommendations[] च्या आत ━━━
active_ingredient   → रासायनिक घटकाचे शास्त्रीय नाव
category            → insecticide=कीडनाशक | fungicide=बुरशीनाशक |
                      herbicide=तणनाशक | bio-pesticide=जैविक | PGR=वाढ नियंत्रक
brand_names[]       → बाजारात मिळणाऱ्या औषधांची नावे — दुकानात हेच नाव सांगा
dose_per_ha         → प्रति हेक्टर डोस (ml किंवा g) — हे standard unit आहे
dose_per_15L        → १५ लिटर पंपासाठी डोस (ml किंवा g) — असल्यास
waiting_period_days → काढणीपूर्वी किती दिवस थांबायचे (PHI — Pre-Harvest Interval)
confidence          → high=खात्रीशीर (CIBRC) | medium=साधारण | low=अंदाजे
layer_used          → 1=CIBRC सरकारी (सर्वात विश्वासार्ह) | 4=AI अंदाज (कमी विश्वासार्ह)

━━━ DOSAGE RULE (महत्त्वाचे) ━━━
- dose_per_ha JSON मध्ये असेल → "प्रति हेक्टर [dose] ml/g" असे सांगा
- dose_per_15L JSON मध्ये असेल → "१५ लिटर पंपासाठी [dose] ml/g" असे सांगा
- दोन्ही नसतील → "डोससाठी कृषी सेवा केंद्रात विचारा" — स्वतःहून कधीही डोस सांगू नका

━━━ HOW TO ANSWER ━━━
१. आढळलेली कीड/रोग कोणती
२. recommendations[] मधून प्रत्येक शिफारशीसाठी:
   - active_ingredient + category मराठीत
   - brand_names[] मधून 2-3 नावे: "दुकानात ___ नावाने मागा"
   - डोस (वरील DOSAGE RULE वापरा)
   - waiting_period_days: "काढणीपूर्वी ___ दिवस थांबा"
३. confidence=low किंवा layer_used=4 → ⚠️ "ही माहिती अंदाजे आहे — कृषी विभागाकडून खात्री करा"
४. warning field असेल → ते स्पष्टपणे सांगा

━━━ OUTPUT FORMAT ━━━
✅ *पीक संरक्षण सल्ला* 🧪

🐛 *आढळलेली समस्या:* [pest_identified / disease_identified]

🧪 *शिफारस:*
- *घटक:* [active_ingredient] ([category मराठीत])
- *बाजारातील नावे:* [brand_names - 2-3]
- *डोस:* [dose_per_ha असेल: "प्रति हेक्टर X ml/g"] [dose_per_15L असेल: "/ १५ लिटरसाठी Y ml/g"]
- *काढणीपूर्वी थांबा:* [waiting_period_days] दिवस (PHI)

[confidence=low किंवा layer_used=4:]
⚠️ *लक्षात घ्या:* ही माहिती अंदाजे आहे — कृपया कृषी सेवा केंद्रात खात्री करा.

[warning असेल:]
⚠️ *सावधानता:* [warning]

⚠️ *महत्त्वाची टीप:* फवारणीपूर्वी औषधाच्या बाटलीवरील लेबल व PHI (काढणीपूर्वी थांबायचा काळ) नक्की तपासा.

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ FINAL RULES ━━━
- dose JSON मध्ये नसेल तर कधीही स्वतःहून सांगू नका
- JSON key नावे output मध्ये कधीही छापू नका
- OUTPUT FORMAT मधील [ ] brackets output मध्ये छापू नका — त्या जागी actual data भरा"""

    return await _call_gemini(prompt)


# ─── Main entry point ─────────────────────────────────────────────────────────

async def format_response_for_whatsapp(
    service_type: str,
    raw_data: dict[str, Any],
    original_user_text: str = "",
) -> str:
    """
    Main entry point — called by main.py after delivery.py returns endpoint data.

    service_type: "weather" | "mandi" | "fertilizer"
    x402 errors:  {"error": True, "error_type": ..., "error_reason": ...}
    No-match:     {"status": "no_match" | "crop_not_found" | "missing_info"}
    """
    # ── x402 / pipeline errors ────────────────────────────────────────────
    if not raw_data or raw_data.get("error") is True:
        error_type   = raw_data.get("error_type", "UNKNOWN") if raw_data else "EMPTY_RESPONSE"
        error_reason = raw_data.get("error_reason", "")      if raw_data else ""
        logger.warning(f"[Formatter] x402 error: {error_type} — {error_reason}")
        if error_type == "DATA_UNAVAILABLE":
            return _error_response("सध्या डेटा उपलब्ध नाही. सर्व्हर काही वेळात पुन्हा चालू होईल.")
        if error_type in ("BAD_REQUEST", "CROP_NOT_FOUND"):
            return _error_response(f"चुकीची माहिती पाठवली गेली: {error_reason}")
        return _error_response("तांत्रिक अडचणीमुळे माहिती मिळवता आली नाही.")

    # ── Endpoint-level no_match ───────────────────────────────────────────
    status = raw_data.get("status")
    if status in ("no_match", "crop_not_found", "missing_info"):
        msg = raw_data.get("message", "माहिती आढळली नाही.")
        return (
            f"⚠️ *माहिती आढळली नाही*\n\n{msg}\n\n"
            "पिकाचे नाव किंवा किडीचे नाव पुन्हा तपासून पाठवा. 🙏"
        )

    # ── Route ─────────────────────────────────────────────────────────────
    if service_type == "weather":
        return await format_weather_response(raw_data, original_user_text)
    elif service_type == "mandi":
        return await format_mandi_response(raw_data, original_user_text)
    elif service_type == "fertilizer":
        return await format_fertilizer_response(raw_data, original_user_text)
    else:
        logger.warning(f"[Formatter] Unknown service_type: '{service_type}'")
        prompt = (
            f"शेतकऱ्यासाठी या कृषी माहितीचे मराठीत सोप्या भाषेत स्पष्टीकरण द्या:\n"
            f"{json.dumps(raw_data, ensure_ascii=False, indent=2)}"
        )
        return await _call_gemini(prompt)