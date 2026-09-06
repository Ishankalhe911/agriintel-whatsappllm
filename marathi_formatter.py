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
        reply = response.text.strip()
        
        # 🚀 META API 4096 CHAR LIMIT FAILSAFE
        if len(reply) > 4000:
            logger.warning(f"[Formatter] Truncating long response: {len(reply)} chars")
            # Find the last clean paragraph break before 3800 characters
            cut_index = reply.rfind('\n\n', 0, 3800)
            if cut_index == -1:
                cut_index = 3800  # Hard cut if no paragraph break is found
            
            # Slice cleanly and add a polite warning
            reply = reply[:cut_index] + "\n\n⚠️ *(संदेश खूप मोठा असल्याने काही पर्याय वगळले आहेत. अधिक माहितीसाठी कृषी सेवा केंद्रात संपर्क करा.)*"
            
        return reply

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

    # ── Query intent detection (Python-side, before Gemini sees anything) ──
    q = (original_text or "").lower()
    # 🚀 ADDED FERTILIZER KEYWORDS HERE:
    spray_keywords   = ["फवारणी", "spray", "ड्रोन", "drone", "औषध", "pesticide", "फवार", "खत", "fertilizer", "युरिया", "dap"]
    rain_keywords    = ["पाऊस", "rain", "पर्जन्य", "वरुण", "बरसात", "पाणी कधी"]
    harvest_keywords = ["काढणी", "harvest", "कापणी", "निघणे"]
    long_keywords    = ["महिना", "हंगाम", "season", "month", "दीर्घ", "long"]
    is_spray_query   = any(k in q for k in spray_keywords)
    is_rain_query    = any(k in q for k in rain_keywords) and not is_spray_query
    is_harvest_query = any(k in q for k in harvest_keywords)
    is_long_query    = any(k in q for k in long_keywords)

    prompt = f"""शेतकऱ्याचा प्रश्न: "{original_text or 'पुढील हवामान कसे राहील?'}"

━━━ DATE ANCHOR — JSON वाचण्यापूर्वी हे वाचा ━━━
आजची तारीख: {today_iso} ({today_human}). वेळ: {current_time} IST.
⚠️ CRITICAL TEMPORAL RULE: जर शेतकऱ्याने "उद्या" (tomorrow) विचारले असेल, तर फक्त {today_iso} च्या पुढच्या तारखेचा (next day) डेटा दाखवा. आजचा डेटा सांगू नका. 
जर शेतकऱ्याने विशिष्ट दिवस विचारले असतील (उदा. "१५ दिवस"), तर JSON मधील सर्व १५ दिवस दाखवा, ७ वर कापू नका.
daily_preview मधील प्रत्येक entry FORECAST आहे — {today_iso} सुद्धा forecast च आहे.
Date display rule: YYYY-MM-DD → "२२ ऑगस्ट" (फक्त दिवस + मराठी महिना).
Date mapping: "आज"→{today_iso} | "उद्या"→पुढची date | "परवा"→त्यानंतरची.
heavy_rain_days / heat_stress_days मधील number म्हणजे day index (1=आज).
Day index → date: daily_preview[index-1]["date"] वरून काढा.

━━━ QUERY INTENT: {"SPRAY_FOCUSED" if is_spray_query else "RAIN_FOCUSED" if is_rain_query else "HARVEST_FOCUSED" if is_harvest_query else "LONG_TERM" if is_long_query else "GENERAL_WEATHER"} ━━━
{"→ शेतकऱ्याने फवारणीबद्दल विचारले आहे. फवारणी section सर्वात आधी आणि सर्वात विस्तारित द्या. बाकी sections संक्षिप्त ठेवा." if is_spray_query else ""}
{"→ शेतकऱ्याने पावसाबद्दल विचारले आहे. पाऊस section विस्तारित द्या, daily_preview सर्व दिवस दाखवा." if is_rain_query else ""}
{"→ शेतकऱ्याने काढणीबद्दल विचारले आहे. horizon_2 आणि horizon_3 sections विस्तारित द्या." if is_harvest_query else ""}
{"→ दीर्घकालीन प्रश्न. horizon_3 + enso_iod_state sections विस्तारित द्या." if is_long_query else ""}

━━━ JSON STRUCTURE (हे top-level keys आहेत) ━━━
season_to_date        → जून १ पासून आत्तापर्यंतचा एकूण पाऊस (harvest_date दिल्यास येतो)
horizon_1_forecast    → मुख्य 0-16 दिवसांचा अंदाज (हे नेहमी असते)
horizon_2_subseasonal → ECMWF 3-4 आठवड्यांचा sub-seasonal अंदाज (harvest_date दिल्यास येतो)
horizon_3_seasonal    → NASA POWER + ENSO दीर्घकालीन मासिक अंदाज (harvest_date दिल्यास येतो)
enso_iod_state        → ENSO/IOD मान्सून स्थिती (ONI + DMI values)
days_to_harvest       → काढणीपर्यंत किती दिवस उरले
partial_data          → true असेल तर horizon_2/3 चा डेटा अपूर्ण आहे

━━━ FIELD MEANINGS — season_to_date ━━━
monsoon_start        → मान्सून हंगाम सुरुवात (सामान्यतः जून १)
accumulated_rain_mm  → जून १ पासून आत्तापर्यंत एकूण पाऊस (मिमी)
agronomic_context    → Horizon 3 च्या normals शी तुलना करा हे string सांगते
⚠️ INTERPRET: accumulated_rain_mm ला horizon_3 च्या त्या महिन्याच्या rainfall_normal_mm शी तुलना करा.
  जास्त पाऊस → पूर/ओला रोग धोका सांगा | कमी पाऊस → सिंचन/दुष्काळ धोका सांगा.

━━━ FIELD MEANINGS — horizon_1_forecast च्या आत ━━━
source                  → "open_meteo"=पूर्ण डेटा | "visual_crossing_fallback"=VC डेटा (Delta-T अनुपलब्ध) | "nasa_power_fallback"=ड्रोन गणना अनुपलब्ध
crop_stress_risk_level  → LOW/MEDIUM/HIGH — पिकावरचा एकूण ताण
crop_stress_factors[]   → कशामुळे ताण: उष्णता, जड पाऊस, पाण्याची कमतरता, कीड/रोग धोका
operational_risk_level  → LOW/MEDIUM/HIGH — शेत कामांना अडचण
operational_factors[]   → कोणती कामे करता येणार नाहीत आणि का
next_rain_date          → पुढचा पाऊस (≥2mm) कधी येईल (YYYY-MM-DD → मराठीत सांगा)
next_dry_spell          → {{ start_date, end_date, days }} — कोरड्या दिवसांचा काळ
                          फवारणी, सिंचन, काढणीसाठी ही खिडकी वापरा
optimal_drone_spray_dates[] → Delta-T (2-8°C) + वारा (<15km/h) + पाऊस (<2mm) या तिन्ही
                          अटी पूर्ण असलेले दिवस — वैज्ञानिकदृष्ट्या सर्वोत्तम फवारणीचे दिवस
pest_disease_risk_windows[] → सलग 3+ दिवस RH>85% + तापमान 25-32°C असेल तर कीड/बुरशी धोका
wind_risk_days[]        → दैनंदिन कमाल वारा >20km/h चे दिवस — हे DAILY MAXIMUM आहे, सरासरी नाही.
                          महाराष्ट्रात मान्सूनमध्ये हे सामान्य आहे.
                          ⚠️ संपूर्ण दिवस टाळू नका — best_spray_window_by_day मध्ये
                          सकाळच्या वेळेत अनुकूल खिडकी असू शकते.
heavy_rain_days[]       → जड पाऊस (पीक-specific threshold ओलांडलेले) day index (1=आज)
heat_stress_days[]      → उष्णता (पीक-specific max temp ओलांडलेले) day index (1=आज)
gdd_accumulated_forecast_window → पुढील 16 दिवसांत पीक किती "उष्णता एकक" मिळवणार
                          (GDD = Growing Degree Days) — जास्त GDD = पीक वेगाने पुढच्या अवस्थेत जाईल
growth_stage            → germination=उगवण | vegetative=वाढ | flowering=फुलोरा ⚠️सर्वात नाजूक
                          pod_fill=शेंगा भरणे | maturity=पक्वता
irrigation_recommended  → हे field null असते — net_water_balance_7d वरूनच सिंचन सल्ला द्या
                          null असेल तर हे field पूर्णपणे skip करा
irrigation_recommendation_status → हे field असेल तर पूर्णपणे skip करा — output मध्ये दाखवू नका
net_water_balance_7d    → ७ दिवसांत पाऊस minus वाफ (ET0) = निव्वळ पाणीसाठा (mm)
                          >0=पाणी पुरेसे | -5 ते 0=किरकोळ कमतरता
                          -5 ते -10=सिंचन करा | <-10=गंभीर कमतरता ⚠️ तातडीने सिंचन
et0_7d_mm               → ७ दिवसांत जमिनीतून वाफ होणारे पाणी (mm) — FAO Penman-Monteith
rainfall_7d_mm          → ७ दिवसांतील एकूण अंदाजित पाऊस (mm)

━━━ FIELD MEANINGS — daily_preview (प्रत्येक दिवसाचे fields) ━━━
date          → YYYY-MM-DD (मराठीत "७ ऑगस्ट" सांगा)
rain_mm       → त्या दिवशी किती पाऊस अपेक्षित
et0_mm        → त्या दिवशी किती पाणी वाफ होईल
t_max_c       → जास्तीत जास्त तापमान (°C)
t_min_c       → किमान तापमान (°C)
rh_max_pct    → जास्तीत जास्त आर्द्रता (%) — >85% + उबदार = बुरशी/कीड धोका
rh_min_pct    → किमान आर्द्रता (%) — Delta-T गणनेसाठी
wind_kmh      → दिवसाचा कमाल sustained वारा km/h (सरासरी नाही)
wind_gust_kmh → दिवसाचा कमाल gust km/h — null असेल तर पूर्णपणे skip करा
wcode         → WMO हवामान कोड:
                0=☀️स्वच्छ | 1-3=🌤️ढगाळ | 45-48=🌫️धुके
                51-67=🌧️पाऊस | 80-82=🌧️⚠️मुसळधार | 95-99=⛈️मेघगर्जना
                ⚠️ wcode 95-99 असेल त्या दिवशी शेतात काम करू नका

━━━ FIELD MEANINGS — best_spray_window_by_day (CRITICAL) ━━━
Structure: {{"YYYY-MM-DD": [ {{window_start, window_end, avg_wind_kmh, max_wind_kmh, max_gust_kmh, rain_mm, rain_probability_max_pct, spray_status}}, ... ]}}
- प्रत्येक date साठी array असते — रिकामी array [] म्हणजे त्या दिवशी कोणतीही सुरक्षित वेळ नाही
- window_start/end: "2026-08-22T06:00" → "सकाळी ६:००" असे मराठीत
- max_wind_kmh → त्या window मधील सर्वाधिक sustained वारा
- max_gust_kmh → त्या window मधील सर्वाधिक gust — null असेल तर skip
- rain_probability_max_pct → पाऊस येण्याची शक्यता % — null असेल तर skip
- spray_status: "favorable"=✅ आदर्श | "caution"=⚠️ सावधगिरीने | array रिकामी=❌ सुरक्षित वेळ नाही

ड्रोन vs Manual distinction (CRITICAL — प्रत्येक window साठी अनिवार्य):
- max_gust_kmh < 20    → ✅ ड्रोन + manual दोन्ही
- max_gust_kmh 20-35   → ✅ Manual knapsack ठीक | ❌ ड्रोन DGCA नियमानुसार नाही
- max_gust_kmh > 35    → ❌ दोन्ही नाही — फवारणी रद्द करा
- max_gust_kmh null    → max_wind_kmh वरून: <15=दोन्ही ठीक | 15-25=manual फक्त | >25=दोन्ही नाही

━━━ FIELD MEANINGS — danger_spray_window_by_day (नवीन field) ━━━
Structure: {{"YYYY-MM-DD": {{window_start, window_end, max_wind_kmh, max_gust_kmh, rain_mm, spray_status}}}}
- हे त्या दिवसातील सर्वात वाईट/धोकादायक वेळ आहे
- फक्त ❌ block दाखवण्यासाठी वापरा: "दुपारी X ते Y टाळा — वारा Z km/h"
- null/missing असेल → त्या दिवशी विशेष धोकादायक वेळ नाही, हा section skip करा

━━━ FIELD MEANINGS — horizon_2_subseasonal ━━━
valid_window     → हा अंदाज कोणत्या कालावधीसाठी आहे
weekly_outlook[] → {{ week, dates, projected_rain_mm, trend }}
trend values:
  "normal_or_wet" → सामान्य किंवा जास्त पाऊस अपेक्षित
  "dry_anomaly"   → ⚠️ कमी पाऊस — या काळात सिंचन तयार ठेवा

━━━ FIELD MEANINGS — horizon_3_seasonal (मासिक दीर्घकालीन अंदाज) ━━━
monthly_outlook[] प्रत्येक महिन्यासाठी:
  rainfall_normal_mm     → ३०-वर्षांचा सरासरी पाऊस (NASA POWER)
  rainfall_adjusted_mm   → ENSO/IOD ग्राह्य धरून समायोजित अंदाज
  rainfall_pct_of_normal → >110%=खूप जास्त पाऊस 🌊 | 90-110%=सामान्य ✅
                           75-90%=कमी पाऊस ⚠️ | <75%=टंचाई धोका 🔴
  t_max_normal_c         → सामान्य कमाल तापमान (°C)
  adjustment_basis       → "ENSO=..., IOD=..."=मान्सून काळात ENSO/IOD वापरला
                           "climatology only"=मान्सूनबाहेर — फक्त ऐतिहासिक सरासरी

━━━ FIELD MEANINGS — enso_iod_state ━━━
oni_phase  → el_nino=मान्सून कमकुवत (कमी पाऊस) | la_nina=मान्सून जोरदार (जास्त पाऊस) | neutral=सामान्य
oni_value  → ONI निर्देशांक: >0.5=El Niño, <-0.5=La Niña
dmi_phase  → positive_iod=El Niño परिणाम कमी करतो | negative_iod=La Niña परिणाम कमी करतो | neutral=सामान्य
dmi_value  → DMI निर्देशांक (ताकद दर्शवतो)
COMBINED EFFECT:
  La Niña + positive_iod  → ⛈️ खूप जास्त पाऊस — पूर/ओला रोग सावधानता
  La Niña + negative_iod  → थोडा जास्त पाऊस, mixed
  El Niño + positive_iod  → ≈ balanced — सामान्य पाऊस शक्य
  El Niño + negative_iod  → 🔴 दुष्काळ धोका — सिंचन योजना आखा
━━━ KEY THRESHOLDS ━━━
━net_water_balance_7d:
  > 0        → जमिनीत पाणी पुरेसे आहे
  -5 ते 0    → किरकोळ कमतरता — पाण्यावर लक्ष ठेवा
  -5 ते -10  → सिंचन करा
  < -10      → गंभीर कमतरता — तातडीने सिंचन करा ⚠️

Delta-T (optimal_drone_spray_dates मधून अप्रत्यक्षपणे):
  2-8°C  → आदर्श फवारणी — रसायन पिकापर्यंत नक्की पोहोचते
  <2°C   → धुके/inversion — फवारणी टाळा
  >8°C   → थेंब वाफ होतात — सकाळी लवकर/संध्याकाळी करा

wind_kmh (daily_preview) — दिवसाचा कमाल वेग, सरासरी नाही:
  < 15 km/h  → दिवसभर फवारणीसाठी योग्य
  15-25 km/h → सकाळी/संध्याकाळी वेळ निवडा — best_spray_window_by_day पहा
  > 25 km/h  → best_spray_window_by_day मध्ये योग्य वेळ असेल तरच फवारणी करा
  > 35 km/h  → ❌ फवारणी टाळा — रसायन शेजारच्या शेतात जाते

wind_gust_kmh — sustained wind_kmh पेक्षा अधिक महत्वाचा:
  < 20 km/h  → ड्रोन + manual दोन्ही ठीक
  20-35 km/h → manual knapsack ठीक — ड्रोन DGCA नियमानुसार टाळा
  > 35 km/h  → ❌ कोणतीही फवारणी टाळा
  null       → wind_kmh वरूनच निर्णय घ्या

growth_stage + net_water_balance:
  flowering + net_water_balance_7d < -10 → ⚠️ तातडीने सिंचन — फुलोरा अवस्थेत पाण्याची कमतरता उत्पादनावर थेट परिणाम करते
  pod_fill + heavy_rain → बुरशीजन्य रोगांसाठी prophylactic फवारणी विचार करा
━━━ SPRAY WINDOW OUTPUT RULES ━━━

नियम १ — प्रत्येक window साठी हे सांगा:
  वेळ: window_start→window_end मराठीत ("सकाळी ६:०० ते ८:००")
  वारा: max_wind_kmh km/h[max_gust_kmh असेल: (जोर max_gust_kmh km/h)]
  पाऊस: rain_mm mm[rain_probability_max_pct असेल: , शक्यता rain_probability_max_pct%]
  ड्रोन/Manual: वरील distinction नुसार स्पष्टपणे सांगा
  spray_status="caution" असेल → 💡 वारा थोडा जास्त आहे — नोझल पिकाच्या अगदी जवळ (६-८ इंच) ठेवा, जेणेकरून औषध drift होणार नाही.

नियम २ — danger_spray_window_by_day असेल त्या दिवशी:
  ❌ *टाळा — [window_start→वेळ] ते [window_end→वेळ]:* वारा [max_wind_kmh] km/h[gust: (जोर [max_gust_kmh])] — औषध drift होईल

नियम ३ — best_spray_window_by_day रिकामी array [] असेल:
  ❌ संपूर्ण दिवस फवारणीयोग्य नाही — [danger window असेल: कारण: वारा [max_wind_kmh] km/h]
  💡 पर्यायी उपाय: [danger window मध्ये rain_mm < 2 असेल म्हणजे फक्त वाऱ्यामुळे block:]
     वाऱ्यामुळे Foliar Spray टाळा — पण Broadcasting (जमिनीवर पसरवणे) किंवा Drip/Fertigation द्वारे खते देणे शक्य आहे.
  [danger window मध्ये rain_mm ≥ 2 असेल: आज पाऊसही आहे — Broadcasting आणि Drip दोन्ही टाळा, खत वाहून जाईल.]

नियम ४ — wind_risk_days मध्ये date असेल पण best_spray_window_by_day मध्ये favorable/caution window असेल:
  "दिवसाचा कमाल वारा जास्त आहे — पण [window वेळ] मध्ये वारा [max_wind_kmh] km/h आहे, फवारणी शक्य आहे"
  → "संपूर्ण दिवस टाळा" कधीही म्हणू नका जोपर्यंत array रिकामी नाही

नियम ५ — optimal_drone_spray_dates मध्ये date असेल:
  त्या window जवळ: "✅ ड्रोन फवारणीसाठी आदर्श — Delta-T आणि वारा दोन्ही योग्य"

नियम ६ — pest_disease_risk_windows असेल:
  "🐛 कीड/बुरशी धोका [dates] — वातावरण कीडीसाठी अनुकूल, उपलब्ध window मध्ये तातडीने फवारणी करा"

━━━ HOW TO ANSWER — intent नुसार ━━━

SPRAY_FOCUSED (शेतकऱ्याने फवारणीबद्दल विचारले):
  Step 0 → शेतकऱ्याच्या प्रश्नाचे थेट उत्तर १-२ ओळींत सर्वात आधी द्या.
           उदा. "होय, उद्या सकाळी ६ ते ८ फवारणी करता येईल." किंवा "नाही, उद्या वारा जास्त आहे — परवा सकाळी संधी आहे."
           Generic header नको — थेट उत्तर द्या.
  Step 1 → best_spray_window_by_day: आज + उद्या + परवा तिन्ही दिवसांचे windows विस्तारित द्या (नियम १-५ नुसार)
  Step 2 → danger_spray_window_by_day: धोकादायक वेळ सांगा (नियम २ नुसार)
  Step 3 → pest_disease_risk_windows असेल → तातडी सांगा (नियम ६)
  Step 4 → optimal_drone_spray_dates → ड्रोन availability सांगा
  Step 5 → daily_preview: फक्त आज + उद्या + परवाचे rain_mm + wcode (संक्षिप्त)
  Step 6 → net_water_balance_7d: एक ओळ फक्त
  Step 7 → wcode 95-99 असेल → ⚠️ मेघगर्जना — शेतात जाऊ नका — हे spray window आधी येते
  बाकी sections (horizon_2, horizon_3, enso) → skip करा जोपर्यंत harvest_date नसेल

RAIN_FOCUSED (पावसाबद्दल विचारले):
  Step 1 → next_rain_date सांगा (मराठी date)
  Step 2 → daily_preview: सर्व उपलब्ध दिवस, rain_mm + wcode emoji + t_max_c
  Step 3 → next_dry_spell असेल → "या काळात फवारणी/काढणी योग्य संधी"
  Step 4 → season_to_date accumulated_rain_mm सांगा — हंगाम surplus/deficit
  Step 5 → horizon_2 weekly_outlook: dry_anomaly असेल → सिंचन तयार ठेवा
  Step 6 → net_water_balance_7d एक ओळ
  Spray windows → skip

GENERAL_WEATHER (default):
  Step 1 → daily_preview: JSON मधील सर्व उपलब्ध दिवस दाखवा (date→मराठी, rain_mm, t_max_c, wind_kmh, wcode emoji). कधीही ७ दिवसांवर थांबवू नका, सर्व डेटा वापरा.
  Step 2 → net_water_balance_7d → सिंचन सल्ला (threshold प्रमाणे)
  Step 3 → growth_stage + GDD असेल → पीक अवस्था सल्ला
  Step 4 → best_spray_window_by_day: फक्त आज + उद्याचे windows (संक्षिप्त, नियम १ नुसार)
  Step 5 → crop_stress_risk_level + factors
  Step 6 → next_dry_spell असेल → संधी सांगा
  Step 7 → horizon_2 + horizon_3 + enso: harvest_date दिल्यास फक्त

HARVEST_FOCUSED / LONG_TERM:
  Step 1 → horizon_3 monthly_outlook: प्रत्येक महिना rainfall_pct_of_normal सहित
  Step 2 → enso_iod_state: combined effect सांगा
  Step 3 → horizon_2 weekly_outlook
  Step 4 → season_to_date surplus/deficit
  Step 5 → daily_preview: फक्त पहिले ५ दिवस संक्षिप्त
  Spray windows → skip

पाऊस / सामान्य / दीर्घकालीन HOW TO ANSWER (existing logic — preserve):
  फवारणी बद्दल → Step 1-7 SPRAY_FOCUSED नुसार
  पाऊस बद्दल → RAIN_FOCUSED नुसार
  सामान्य → GENERAL_WEATHER नुसार
  दीर्घकालीन → HARVEST_FOCUSED/LONG_TERM नुसार

━━━ OUTPUT FORMAT ━━━

[SPRAY_FOCUSED असेल तर — थेट उत्तर सर्वात आधी, generic header नाही:]
[शेतकऱ्याच्या प्रश्नाचे थेट १-२ ओळींत उत्तर — उदा. "होय, उद्या सकाळी ६ ते ८ फवारणी करता येईल."]

[GENERAL / RAIN / HARVEST असेल तर:]
*[पीक] हवामान अंदाज* 🌾

[growth_stage असेल तर:]
🌱 *पीक अवस्था:* [growth_stage मराठीत] [days_to_harvest असेल: | काढणी [N] दिवसांत]
[GDD असेल: पुढील 16 दिवसांत [gdd] उष्णता एकके मिळणार]
[growth_stage विशेष सल्ला: flowering+negative balance→तातडीने सिंचन | pod_fill+heavy_rain→prophylactic फवारणी]

[SPRAY_FOCUSED — हे section FIRST, विस्तारित:]
🚜 *फवारणीसाठी सुरक्षित वेळ*

[आज/उद्या/परवा प्रत्येक दिवसासाठी:]
📅 *[मराठी date] ([wcode emoji]):*
[wcode 95-99 असेल: ⚠️ मेघगर्जना — आज शेतात जाऊ नका]
[best_spray_window_by_day मध्ये windows असतील:]
✅ *सुरक्षित वेळ:*
- *[window_start→मराठी वेळ] ते [window_end→मराठी वेळ]:* वारा [max_wind_kmh] km/h[gust null नसेल: (जोर [max_gust_kmh] km/h)][rain_mm>0: , पाऊस [rain_mm] mm][rain_prob null नसेल: , शक्यता [rain_probability_max_pct]%]
  → [ड्रोन/manual distinction — नियम १ नुसार]
  [spray_status="caution": 💡 नोझल पिकाच्या अगदी जवळ (६-८ इंच) ठेवा — वारा थोडा जास्त आहे.]
[दुसरी window असेल:]
- *[वेळ] ते [वेळ]:* [तेच fields + distinction + caution टीप]
[danger_spray_window_by_day त्या date साठी असेल:]
❌ *टाळा — [danger window_start→वेळ] ते [window_end→वेळ]:* वारा [max_wind_kmh] km/h[gust: (जोर [max_gust_kmh])] — औषध उडून जाईल
[best array रिकामी असेल:]
❌ *संपूर्ण दिवस फवारणीयोग्य नाही*[danger window असेल: — वारा [max_wind_kmh] km/h]
💡 *पर्यायी उपाय:* [danger rain_mm < 2: वाऱ्यामुळे Foliar Spray टाळा — Broadcasting किंवा Drip/Fertigation द्वारे खते देणे शक्य आहे.]
[danger rain_mm ≥ 2: आज पाऊसही आहे — Broadcasting आणि Drip दोन्ही टाळा, खत वाहून जाईल.]

[optimal_drone_spray_dates असेल:]
🛸 *ड्रोन फवारणी आदर्श दिवस:* [dates → मराठीत] — Delta-T + वारा दोन्ही नियंत्रणात

[pest_disease_risk_windows असेल:]
🐛 *⚠️ कीड/बुरशी धोका:* [dates] — वातावरण कीडीसाठी अनुकूल, उपलब्ध window मध्ये तातडीने फवारणी करा

[RAIN_FOCUSED / GENERAL — daily_preview section:]
☀️ *पुढील दिवसांचा अंदाज:*
[प्रत्येक दिवस:]
- *[मराठी date]* [wcode emoji]: पाऊस [rain_mm] मिमी, कमाल [t_max_c]°C, वारा [wind_kmh] km/h[wind_gust_kmh null नसेल: (जोर [wind_gust_kmh])]

[net_water_balance_7d — SPRAY_FOCUSED मध्ये एक ओळ, बाकी सर्वांत पूर्ण:]
[SPRAY_FOCUSED:] 💧 *पाण्याचे संतुलन:* [net_water_balance_7d] मिमी — [एक ओळ interpretation]
[बाकी:] 💧 *पाण्याचा ताळेबंद (७ दिवस):* [net_water_balance_7d] मिमी
(पाऊस [rainfall_7d_mm] मिमी − वाफ [et0_7d_mm] मिमी)
[threshold नुसार: किरकोळ कमतरता / सिंचन करा ⚠️ / तातडीने सिंचन 🔴]

[GENERAL मध्ये spray windows — संक्षिप्त, फक्त आज + उद्या:]
🚜 *फवारणी सल्ला:*
- [मराठी date]: [spray_status emoji] [window_start→वेळ] ते [window_end→वेळ] | वारा [max_wind_kmh] km/h[gust: (जोर [max_gust_kmh])]
  → [ड्रोन/manual distinction]
  [caution: 💡 नोझल जवळ ठेवा]
[null/रिकामी entry: - [मराठी date]: ❌ फवारणीसाठी चांगली वेळ नाही]
[wind_risk_days असेल: ⏰ [dates] — दिवसाचा कमाल वारा जास्त, पण सकाळच्या खिडकीत फवारणी शक्य — वर पहा]

[पीक ताण — GENERAL/RAIN मध्ये:]
🌡️ *पीक ताण:* [LOW→कमी 🟢 | MEDIUM→मध्यम 🟡 | HIGH→जास्त 🔴]
[factors: कारण: [factors मराठीत]]
[operational_factors: ⚠️ [factors]]

[horizon_2 असेल — HARVEST/LONG_TERM/RAIN मध्ये:]
📅 *पुढील महिन्याचा अंदाज (ECMWF):*
- आठवडा [N] ([dates]): [projected_rain_mm] मिमी — [dry_anomaly→⚠️ कमी पाऊस, सिंचन तयार ठेवा | normal_or_wet→सामान्य/जास्त पाऊस]

[horizon_3 असेल — HARVEST/LONG_TERM मध्ये:]
📆 *हंगाम अंदाज (NASA):*
[प्रत्येक month: *[महिना मराठीत]:* [rainfall_pct_of_normal]% सामान्य — [>110%→जास्त पाऊस ⛈️ | 90-110%→सामान्य ✅ | <90%→कमी पाऊस ⚠️ सिंचन तयार]]

[enso_iod_state असेल — HARVEST/LONG_TERM मध्ये:]
🌏 *मान्सून स्थिती:* [oni_phase मराठीत][oni_value: (ONI [value])] | [dmi_phase मराठीत][dmi_value: (DMI [value])]
[combined effect एक ओळ — वरील COMBINED EFFECT table नुसार]

[season_to_date accumulated_rain_mm असेल:]
🌧️ *हंगाम पाऊस (जून १ पासून):* [accumulated_rain_mm] मिमी — [surplus/deficit interpretation]

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ FINAL RULES ━━━
- JSON key नावे कधीही output मध्ये येऊ नयेत (horizon_1_forecast, daily_preview, wcode, etc.)
- null fields पूर्णपणे skip करा — "उपलब्ध नाही" पण लिहू नका
- Dates नेहमी "७ ऑगस्ट" format — YYYY-MM-DD कधीही नको
- heavy_rain_days/heat_stress_days → daily_preview मधून actual date काढा
- partial_data=true असेल तर: "काही दीर्घकालीन डेटा उपलब्ध नाही"
- source="nasa_power_fallback" → "ड्रोन फवारणी गणना सध्या उपलब्ध नाही" एक ओळ
- source="visual_crossing_fallback" → Delta-T mention करू नका
- irrigation_recommended null → output मध्ये कधीही येऊ नका
- irrigation_recommendation_status → output मध्ये कधीही येऊ नका
- spray_windows field → output मध्ये कधीही येऊ नका — फक्त best_spray_window_by_day आणि danger_spray_window_by_day वापरा
- wind_gust_kmh null असेल → त्या दिवशी gust mention करू नका
- best_spray_window_by_day मध्ये key missing (not empty array) → त्या date साठी windows नाहीत असे समजा
- wind_risk_days = blanket block नाही — best_spray_window_by_day तपासा, वेळेचा सल्ला द्या
- महाराष्ट्र मान्सून context: 15-25 km/h वारा सामान्य आहे — संपूर्ण दिवस block करू नका
- wind_gust_kmh हा sustained wind_kmh पेक्षा अधिक महत्वाचा फवारणी निर्णयासाठी
- ड्रोन: DGCA नियमानुसार gusts <20 km/h — हे optimal_drone_spray_dates मध्ये आधीच तपासले आहे
- rainfall_pct_of_normal शिवाय horizon_3 पाऊस सांगू नका
- adjustment_basis "climatology only" → ENSO/IOD mention करू नका त्या महिन्यासाठी
- SPRAY_FOCUSED → horizon_2, horizon_3, enso sections skip (harvest_date असेल तरच एक ओळ summary)
- danger_spray_window_by_day null/missing → danger block पूर्णपणे skip करा
- wcode 95-99 असलेल्या दिवशी → spray window आधी मेघगर्जना warning द्या
- प्रत्येक window साठी ड्रोन/manual distinction अनिवार्य आहे"""

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
तुमचे काम शेतकऱ्याला त्याच्या मालासाठी सर्वोत्तम बाजारपेठ शोधून देणे,
खरे नफा-तोटा सांगणे, आणि 'agent_execution_rules' मधील सूचनांचे काटेकोरपणे पालन करणे.

━━━ JSON STRUCTURE (top-level keys) ━━━
crop                     → पिकाचे नाव (output header साठी वापरा)
mode                     → "price_only" किंवा "full_optimization"
qty_quintals             → शेतकऱ्याचा माल (क्विंटलमध्ये — profit calculation साठी)
radius_km                → शोधाची त्रिज्या (km) — distance warnings साठी वापरा
is_within_requested_radius → true/false — ⚠️ हे CRITICAL आहे (खाली HOW TO ANSWER पहा)
presentation_rule        → endpoint ने सांगितलेला सादरीकरणाचा नियम (खाली पहा)
nearest_mandi            → Baseline — सर्वात जवळची मंडी
top_mandis[]             → सर्वाधिक फायदेशीर मंडींची यादी (Arbitrage संधी)
agent_execution_rules    → Checklist, Warnings, आणि विशेष सूचना

━━━ FIELD MEANINGS — nearest_mandi आणि top_mandis[] च्या आत ━━━
market                                     → मंडीचे नाव
driving_distance.value_km                  → रस्त्याने अंतर (km) — हवाई अंतर नाही
driving_distance.duration_min              → प्रवासाचा अंदाजित वेळ (मिनिटे) — असल्यास सांगा
exact_scraped_data.modal_price_per_quintal → आजचा मुख्य भाव (₹/क्विंटल) — सर्वाधिक व्यवहार या भावाने
exact_scraped_data.min_price               → किमान भाव (₹/क्विंटल) — Conservative estimate
exact_scraped_data.max_price               → कमाल भाव (₹/क्विंटल) — Optimistic estimate
exact_scraped_data.variety                 → वाण/प्रकार — ⚠️ Variety mismatch असल्यास variety_warning_to_show_user पहा
exact_scraped_data.arrivals_qty            → आज मंडीत आलेला माल (क्विंटल/मेट्रिक टन) — असल्यास:
                                             जास्त आवक = भाव दबावात | कमी आवक = भाव टिकतात
exact_scraped_data.price_date              → भावाची तारीख — जुना असेल तर ⚠️ सांगा
price_spread                               → max_price - min_price — मोठा spread = negotiation संधी
transport_cost_est                         → वाहतूक खर्च अंदाज (₹) — qty_quintals साठी एकूण
apmc_commission_pct                        → APMC + Arhtiya एकत्रित कपात % (हा % Gross वर लावा)
hamali_est                                 → हमाली + तुलाई खर्च (₹) — असल्यास calculation मध्ये घ्या
price_advantage_over_nearest               → या मंडीचा nearest_mandi पेक्षा प्रति क्विंटल फायदा (₹) — असल्यास
arrival_trend                              → भावाचा कल:
                                             "rising"  = भाव वाढत आहेत → ⚡ थोडा वेळ थांबणे शक्य असल्यास विचार करा
                                             "falling" = भाव घसरत आहेत → 🔴 आजच विका — उशीर नको
                                             "stable"  = भाव स्थिर → कोणत्याही दिवशी सारखेच
data_source                                → "agmarknet_api" | "msamb_scraping" | "cached" — असल्यास:
                                             "cached" = भाव थोडा जुना असू शकतो, प्रत्यक्ष मंडीत confirm करा

━━━ FIELD MEANINGS — agent_execution_rules च्या आत ━━━
pre_dispatch_checklist_to_show_user[]  → माल पाठवण्यापूर्वी करायच्या गोष्टी (उत्तम मराठीत द्या)
variety_warning_to_show_user           → वाण संबंधी इशारा — ठळक अक्षरात शेवटी द्या
market_holiday_warning                 → मंडी आज बंद असल्यास इशारा — ⚠️ प्राधान्याने सांगा
is_price_stale_warning                 → भाव जुना असल्यास इशारा — असल्यास output मध्ये द्या
arbitrage_net_gain                     → nearest vs best mandi मधील एकूण अंदाजित फायदा (₹)
                                         हा pre-computed असेल तर स्वतः calculate करू नका

━━━ presentation_rule VALUES आणि त्यांचे अर्थ ━━━
"show_nearest_first"     → नेहमीसारखे: आधी nearest_mandi, नंतर top_mandis
"highlight_arbitrage"    → Arbitrage फायदा ठळकपणे दाखवा, nearest_mandi दुय्यम
"price_only_no_calc"     → फक्त भाव सांगा, कोणताही calculation नाही
"distance_warning_only"  → is_within_requested_radius=false असताना येतो (खाली पहा)
(हा field नसल्यास default: "show_nearest_first")

━━━ ⚠️ CRITICAL — is_within_requested_radius LOGIC ━━━
जर is_within_requested_radius = false असेल:
→ शेतकऱ्याला स्पष्टपणे सांगा: "तुमच्या [radius_km] km परिसरात आज [crop] साठी
  कोणतीही मंडी उपलब्ध नाही किंवा भाव उपलब्ध नाही."
→ nearest_mandi आणि top_mandis चे भाव "संदर्भ म्हणून" द्या परंतु अंतर स्पष्ट सांगा.
→ driving_distance.value_km आणि duration_min (असल्यास) अवश्य द्या.
→ शेतकऱ्याला निर्णय घेऊ द्या — भाव लपवू नका, पण अंतर लपवू नका.
जर is_within_requested_radius = true असेल → सामान्यपणे सांगा.

━━━ PROFIT CALCULATION ENGINE ━━━
फक्त mode="full_optimization" असेल तरच वापरा.
JSON मध्ये qty_quintals, min_price/max_price, transport_cost_est असणे आवश्यक.

चरण १ — Gross Revenue:
  Conservative Gross = min_price × qty_quintals
  Optimistic Gross   = max_price × qty_quintals

चरण २ — APMC + Arhtiya कपात (apmc_commission_pct असल्यास):
  Conservative Deduction = Conservative Gross × (apmc_commission_pct / 100)
  Optimistic Deduction   = Optimistic Gross   × (apmc_commission_pct / 100)

चरण ३ — हमाली कपात (hamali_est JSON मध्ये असल्यासच — स्वतःहून लावू नका):
  Hamali = hamali_est (already total, not per-quintal)

चरण ४ — वाहतूक खर्च:
  Transport = transport_cost_est

चरण ५ — Net:
  Conservative Net = Conservative Gross - Conservative Deduction - Hamali (if any) - Transport
  Optimistic Net   = Optimistic Gross   - Optimistic Deduction   - Hamali (if any) - Transport

चरण ६ — Net Per Quintal (शेतकऱ्यासाठी सर्वात उपयुक्त आकडा):
  Net Per Quintal (Conservative) = Conservative Net / qty_quintals

जर arbitrage_net_gain JSON मध्ये pre-computed असेल → तो वापरा, स्वतः calculate करू नका.
वरीलपैकी कोणताही आकडा JSON मध्ये नसेल → त्या calculation चा भाग पूर्णपणे सोडा.

━━━ HOW TO ANSWER (AGENT RULES) ━━━

नियम १ — DISTANCE WARNING प्रथम:
  is_within_requested_radius = false → RESPONSE सुरुवातीलाच ⚠️ distance warning द्या.
  market_holiday_warning असेल → RESPONSE सुरुवातीलाच द्या.
  is_price_stale_warning असेल → भावाजवळ "(भाव जुना)" असे लिहा.

नियम २ — Presentation Order:
  presentation_rule = "show_nearest_first" (default) → nearest_mandi आधी, नंतर top_mandis.
  presentation_rule = "highlight_arbitrage" → Arbitrage section आधी, nearest नंतर.
  presentation_rule = "price_only_no_calc" → mode="price_only" प्रमाणे वागा.

नियम ३ — Mode नुसार content:
  mode = "price_only" → भाव आणि अंतर फक्त. Profit calculation नाही. वाहतूक खर्च नाही.
  mode = "full_optimization" → Calculation Engine (वर दिलेले) वापरा.

नियम ४ — top_mandis[] रिकामा असेल:
  → "तुमच्या परिसरात आज इतर कोणतीही फायदेशीर मंडी आढळली नाही.
     [nearest_mandi] हाच आजचा सर्वोत्तम पर्याय आहे."

नियम ५ — Arrival Trend आधारित सल्ला:
  "rising"  → "भाव वाढत असल्याने, माल साठवण्याची सोय असल्यास १-२ दिवस थांबणे विचार करा."
               ⚠️ पण: माल जास्त काळ ठेवणे शक्य नसेल तर हे सल्ला देऊ नका.
  "falling" → "भाव घसरत आहेत — आजच माल न्या, उशीर केल्यास अधिक नुकसान."
  "stable"  → "भाव स्थिर आहेत, कोणत्याही दिवशी विकणे सारखेच."

नियम ६ — arrivals_qty असेल:
  जास्त आवक (local context नुसार high) → "आज मंडीत जास्त माल आल्याने भावावर दबाव असू शकतो."
  कमी आवक → "आज माल कमी आल्याने भाव टिकण्याची शक्यता."

नियम ७ — pre_dispatch_checklist_to_show_user मधील मुद्दे:
  बुलेट पॉईंट्स (-) मध्ये उत्तम मराठीत भाषांतरित करा — एकही मुद्दा सोडू नका.

नियम ८ — variety_warning_to_show_user:
  शेवटी ठळक अक्षरात द्या. JSON मध्ये नसेल तर हा section पूर्णपणे skip करा.

━━━ OUTPUT FORMAT (याच साच्यात उत्तर द्या) ━━━

[is_within_requested_radius=false असल्यास येथे:]
⚠️ *[radius_km] km परिसरात आज [crop] ची मंडी उपलब्ध नाही*
खाली जवळच्या मंडींचे संदर्भ भाव दिले आहेत — अंतर लक्षात घेऊन निर्णय घ्या.

[market_holiday_warning असल्यास येथे:]
⚠️ *[market_holiday_warning मराठीत]*

✅ *[crop] मंडी भाव विश्लेषण* 💰

📍 *तुमची जवळची मंडी: [nearest_mandi.market]*
- अंतर: [value_km] km[duration_min असल्यास: — प्रवास सुमारे [duration_min] मिनिटे]
- आजचा मुख्य दर: ₹[modal_price_per_quintal] प्रति क्विंटल[is_price_stale_warning असल्यास: (भाव जुना — confirm करा)]
[min/max असल्यास:] - दर range: ₹[min_price] ते ₹[max_price][price_spread मोठा असल्यास: — मोठा spread, negotiation करा]
- वाण/प्रकार: [variety]
[arrivals_qty असल्यास:] - आजची आवक: [arrivals_qty] — [interpretation]

[top_mandis रिकामा नसल्यास:]
🏆 *इतर फायदेशीर मंड्या (Arbitrage):*
[प्रत्येक top_mandi साठी:]
- *[market]:* ₹[modal_price_per_quintal] प्रति क्विंटल | अंतर: [value_km] km[duration_min: — ~[N] मिनिटे] | वाण: [variety]
  [price_advantage_over_nearest असल्यास:] → जवळच्या मंडीपेक्षा ₹[price_advantage_over_nearest]/क्विंटल जास्त

[mode="full_optimization" आणि qty_quintals असल्यासच:]
💰 *[qty_quintals] क्विंटलसाठी नफा अंदाज ([best top_mandi.market]):*
- Gross Revenue: ₹[min_price × qty] ते ₹[max_price × qty]
- APMC/Arhtiya कपात ([apmc_commission_pct]%): − ₹[deduction conservative] ते − ₹[deduction optimistic]
[hamali_est असल्यास:] - हमाली: − ₹[hamali_est]
- वाहतूक खर्च: − ₹[transport_cost_est]
- *Net मिळकत: ₹[conservative net] ते ₹[optimistic net]*
- *प्रति क्विंटल: ₹[net per quintal conservative] ते ₹[net per quintal optimistic]*
[arbitrage_net_gain असल्यास:] - जवळच्या मंडीपेक्षा एकूण फायदा: *₹[arbitrage_net_gain] जास्त*

[arrival_trend असल्यास:]
📈 *भाव कल:* [arrival_trend नुसार सल्ला — नियम ५ प्रमाणे]

📝 *माल पाठवण्यापूर्वीची तयारी:*
- [checklist item 1 मराठीत]
- [checklist item 2 मराठीत]
- [checklist item 3 मराठीत]
[सर्व items — एकही सोडू नका]

[variety_warning_to_show_user असल्यास:]
💡 *महत्त्वाची टीप:* [variety_warning_to_show_user मराठीत — ठळक]

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL ANTI-HALLUCINATION RULES ━━━
- top_mandis[] मधीलच मंड्या घ्या — स्वतःहून कोणतीही मंडी (गडचिरोली, गोंदिया, रोहा इ.) जोडू नका
- JSON मध्ये hamali_est नसेल → हमाली calculation करू नका — फक्त apmc_commission_pct वापरा
- JSON मध्ये duration_min नसेल → प्रवासाचा वेळ स्वतःहून लिहू नका
- JSON मध्ये min_price, max_price, qty_quintals यापैकी कोणतेही नसेल → profit calculation करू नका
- arbitrage_net_gain JSON मध्ये असेल → तोच वापरा, स्वतः calculate करून override करू नका
- price_advantage_over_nearest JSON मध्ये असेल → तोच वापरा, स्वतः calculate करू नका
- data_source="cached" असेल → "(भाव जुना असू शकतो)" असे भावाजवळ नोंद करा
- JSON key नावे (exact_scraped_data, driving_distance, agent_execution_rules इ.) output मध्ये छापू नका
- OUTPUT FORMAT मधील [ ] brackets output मध्ये छापू नका — त्या जागी माहिती भरा
- JSON मध्ये नसलेला कोणताही section पूर्णपणे skip करा"""

    return await _call_gemini(prompt)
# ─── Cross-service recommendation engine (deterministic, non-LLM) ────────────
import re as _re

def _cross_service_suggestions(data: dict[str, Any]) -> str:
    resolved = data.get("resolved_parameters", {}) or {}
    recs = data.get("recommendations", {}) or {}
    is_seed = bool(resolved.get("is_seed_treatment_query"))

    cats_present = set(recs.keys()) - {"overlap_best_matches"}
    cats_present = {c for c in cats_present if recs.get(c)}

    has_herbicide = "herbicide" in cats_present
    has_spray_category = bool(cats_present & {"insecticide", "fungicide", "bio_pesticide", "pgr"})

    has_short_phi = False
    for cat, entries in recs.items():
        for entry in entries or []:
            wp = ((entry.get("dosage") or {}).get("waiting_period")) or ""
            if wp and "not applicable" not in wp.lower():
                m = _re.search(r"(\d+)", wp)
                if m and int(m.group(1)) <= 20:
                    has_short_phi = True
                    break
        if has_short_phi:
            break

    lines = []

    if is_seed:
        lines.append("🌦️ *पुढची पायरी:* पेरणीनंतर बियाण्याला योग्य ओलावा मिळण्यासाठी पावसाचा अंदाज आमच्या *हवामान* सेवेतून आधी तपासा.")
    else:
        if has_herbicide:
            lines.append("🌦️ *पुढची पायरी:* तणनाशक फवारल्यानंतर किमान ६-८ तास पाऊस नसावा (नाहीतर औषध वाया जाते) — *हवामान* सेवेतून पावसाचा अंदाज आधी तपासा.")
        elif has_spray_category:
            lines.append("🌦️ *पुढची पायरी:* फवारणीपूर्वी वारा आणि पावसाचा अंदाज आमच्या *हवामान* सेवेतून तपासा — योग्य वेळ निवडल्यास औषध वाया जाणार नाही.")

        if has_short_phi:
            lines.append("💰 *अजून एक सल्ला:* काढणी जवळ येत आहे — आमच्या *मंडी भाव* सेवेतून आधीच जवळच्या मंडीतले भाव तपासून ठेवा, ऐनवेळी धावपळ नको.")

    if not lines:
        return ""

    return "\n\n" + "\n".join(lines)
# ─── FERTILIZER formatter ─────────────────────────────────────────────────────

async def format_fertilizer_response(
    data: dict[str, Any],
    original_text: str = "",
) -> str:
    prompt = f"""शेतकऱ्याचा संदेश: "{original_text or 'कीड/रोग माहिती'}"

━━━ ⚠️ SAFETY PRIORITY — हे सर्वात आधी वाचा ━━━
हा सल्ला रासायनिक औषधांबद्दल आहे. चुकीचा डोस → पीक नुकसान किंवा मानवी विषबाधा.
चुकीचे waiting period → अन्नात कीडनाशक अवशेष → FSSAI उल्लंघन.
JSON मध्ये नसलेले कोणतेही डोस, brand names, किंवा रासायनिक नावे स्वतःहून कधीही लिहू नका.

━━━ JSON STRUCTURE (top-level keys — actual endpoint schema) ━━━
status                              → "success"
resolved_parameters.crop            → normalized crop key
resolved_parameters.crop_display    → शेतकऱ्याने लिहिलेले मूळ पिकाचे नाव — output header साठी वापरा
resolved_parameters.targets_resolved → कोणत्या कीड/रोगांसाठी शोध घेतला (list)
resolved_parameters.mapped_from_symptom → true असेल तर लक्षणावरून कीड ओळखली गेली
resolved_parameters.is_pgr_query    → true असेल तर हे ग्रोथ बूस्टर उत्तर आहे
resolved_parameters.is_seed_treatment_query → true असेल तर हे पेरणीच्या वेळचे बीजप्रक्रिया उत्तर आहे
  ⚠️ is_pgr_query आणि is_seed_treatment_query कधीही एकत्र true नसतात — दोन्ही वेगळे आहेत.
recommendations                     → हे एक OBJECT आहे, LIST नाही! खालील keys असू शकतात:
  .overlap_best_matches[]           → एकापेक्षा जास्त कीड/रोग असतील — सर्वांवर चालणारी औषधे
  .insecticide[]                    → कीटकनाशके
  .bio_pesticide[]                  → जैविक/सेंद्रिय उपाय
  .fungicide[]                      → बुरशीनाशके
  .herbicide[]                      → तणनाशके
  .pgr[]                            → वाढ नियंत्रक (is_pgr_query=true असताना)
  .seed_treatment[]                 → बीजप्रक्रिया / पेरणी-वेळचे औषध (is_seed_treatment_query=true असताना)
summary.total_options               → एकूण किती पर्याय सापडले
summary.has_bio_options             → जैविक पर्याय उपलब्ध आहे का
summary.has_branded_options         → बाजारातील ब्रँड नावे उपलब्ध आहेत का

━━━ FIELD MEANINGS — प्रत्येक recommendation entry च्या आत ━━━
chemical_name            → रासायनिक घटकाचे शास्त्रीय/CIBRC नाव (हेच खरे नाव — बाटलीवर तपासा)
category                 → insecticide=कीडनाशक | fungicide=बुरशीनाशक |
                            herbicide=तणनाशक | bio_pesticide=जैविक | pgr=वाढ नियंत्रक |
                            seed_treatment=बीजप्रक्रिया (पेरणीच्या वेळी बियाण्यास लावायचे)
is_combination_product   → true असेल तर हे दोन घटकांचे मिश्रण आहे — तसे नमूद करा
covers_all_pests         → true असेल तर सर्व कीड/रोगांवर हे एकच औषध चालते
pests_covered[]          → हे औषध नक्की कोणत्या कीड/रोगासाठी आहे (seed_treatment साठी सहसा रिकामे — skip करा)
dosage.ai_dose           → active ingredient प्रमाणे डोस (असल्यास)
dosage.formulation_dose  → प्रत्यक्ष बाटली/पाकिटावरील फॉर्म्युलेशन डोस — शेतकऱ्यासाठी हाच सर्वात उपयोगी
dosage.dose_unit         → डोसचे खरे एकक: "kg"/"g"/"ml"/"l" — याच शब्दाला मराठीत भाषांतर करून वापरा
                            (kg→किलो, g→ग्राम, ml→मिली, l→लिटर). null असेल तर एकक अजिबात लिहू नका.
dosage.dose_unit_confidence → "high" म्हणजे एकक थेट DB मधून खात्रीशीर आहे,
                            "low" म्हणजे अंदाज आहे — तेव्हा सोबत caution द्यावी लागते
dosage.water_dilution    → किती लिटर पाण्यात मिसळायचे
dosage.waiting_period    → PHI — काढणीपूर्वी किती दिवस थांबायचे — नेहमी सांगा, असल्यास
                            seed_treatment साठी हे सहसा "Not applicable (seed treatment)" येते — तसेच सांगा
dosage.application_method → फवारणी/मातीत/बियाण्यावर — कशा प्रकारे वापरायचे
brands[]                 → बाजारात मिळणाऱ्या औषधांची नावे — यादीतीलच नावे सांगा
companies[]              → या ब्रँड्स बनवणाऱ्या कंपन्या
has_brand_info           → false असेल तर brands/companies section पूर्णपणे skip करा
diy_homemade_options[]   → घरगुती/DIY उपाय (फक्त bio_pesticide साठी)

━━━ DOSAGE & UNIT RULE (CRITICAL) ━━━
१. एकर (Acre) ला प्राधान्य: JSON मध्ये 'formulation_dose_per_acre' असल्यास तोच आकडा वापरा आणि पुढे "प्रति एकर" लिहा. ते नसल्यासच 'formulation_dose' वापरून "प्रति हेक्टर" लिहा.
२. योग्य एकक (Unit): एकक कधीही स्वतःहून guess करू नका — फक्त dosage.dose_unit मधलाच शब्द वापरा:
   - dosage.dose_unit == "kg" → "किलो" म्हणा (कधीही "ग्राम" म्हणू नका)
   - dosage.dose_unit == "g"  → "ग्राम" म्हणा
   - dosage.dose_unit == "ml" → "मिली" म्हणा
   - dosage.dose_unit == "l"  → "लिटर" म्हणा
   - dosage.dose_unit == null → संख्येसोबत कोणतेही एकक न लिहिता वाक्याच्या शेवटी
     "(अचूक प्रमाणासाठी औषधाच्या पाकिटावरील लेबल पहा)" असे स्पष्ट सांगा
   - dosage.dose_unit_confidence == "low" असेल → वाक्यात "(अंदाजे — पक्के प्रमाण लेबलवर तपासा)" ही caution जोडा
३. पाणी (Water): 'water_dilution_per_acre' असल्यास "[X] लिटर पाणी प्रति एकर" सांगा. नसल्यास 'water_dilution' वापरून "[X] लिटर पाणी प्रति हेक्टर" सांगा.
४. पंप (Pump): JSON मध्ये 'formulation_dose_per_15L_pump' असल्यास "*(१५ लिटर पंपासाठी: [value] मिली/ग्राम)*" असे ठळकपणे सांगा. हे नसेल तरच "*(१५ किंवा २० लिटरच्या पंपासाठी योग्य प्रमाण काढा)*" ही टीप जोडा.
५. AI Dose: 'ai_dose' कंसात "(सक्रिय घटक: [X])" असे लिहा.
६. Waiting Period: 'waiting_period' मधील "days" चे भाषांतर "दिवस" करा (उदा. '55 days' -> '५५ दिवस').

━━━ TRANSLATION RULE (CRITICAL) ━━━
JSON मधील pests_covered (उदा. "cynodon_dactylon" -> "हरळी / दुर्वा", "cyperus_rotundus" -> "लव्हाळा"), आणि symptoms हे सर्व १००% मराठीत भाषांतरित करूनच सांगा. इंग्रजी लॅटिन शब्द output मध्ये अजिबात नकोत.

━━━ BRANDS RULE ━━━
has_brand_info = false → brands आणि companies section पूर्णपणे skip करा
has_brand_info = true → brands[] मधून जास्तीत जास्त ३ नावे + companies[] मधून जास्तीत जास्त २ नावे

━━━ HEADER LOGIC (resolved_parameters वरून ठरवा — याच क्रमाने तपासा) ━━━
is_seed_treatment_query = true          → ✅ *[crop_display] बीजप्रक्रिया सल्ला* 🌱
is_pgr_query = true                     → ✅ *[crop_display] ग्रोथ बूस्टर सल्ला* 🌱
mapped_from_symptom = true              → ✅ *[crop_display] लक्षणावरून ओळखलेली समस्या* 🔍
recommendations मध्ये फक्त fungicide[]  → ✅ *[crop_display] रोग व्यवस्थापन सल्ला* 🍃
recommendations मध्ये फक्त herbicide[] → ✅ *[crop_display] तण व्यवस्थापन सल्ला* 🌿
इतर सर्व (कीड/मिश्र)                   → ✅ *[crop_display] पीक संरक्षण सल्ला* 🧪

━━━ HOW TO ANSWER ━━━
नियम ० — is_seed_treatment_query = true असेल (SPECIAL — बाकी सर्व नियम इथे थांबतात):
  हे पेरणीच्या वेळचे बीजप्रक्रिया उत्तर आहे — शेतात सध्या कोणतीही कीड/रोग नाही.
  🎯 सर्व समस्यांसाठी उपयुक्त / IPM जैविक-आधी सल्ला / pests_covered — यातले काहीही दाखवू नका.
  थेट खाली दिलेला विशेष header + recommendations.seed_treatment[] दाखवा.
नियम १ — overlap_best_matches[] असेल आणि रिकामे नसेल: हे सर्वात आधी दाखवा — "🎯 सर्व समस्यांसाठी उपयुक्त:" असे header देऊन.
नियम २ — IPM hierarchy: summary.has_bio_options = true असेल तर "🌿 जैविक उपाय आधी वापरून पहा" असा सल्ला द्या.
नियम ३ — Multiple recommendations: प्रत्येकासाठी स्वतंत्र *पर्याय १*, *पर्याय २* block द्या. स्वतः "हा best आहे" म्हणू नका.
नियम ४ — is_combination_product = true: "(हे दोन घटकांचे संयुक्त औषध आहे)" असे नमूद करा.
नियम ५ — Bifurcation (वर्गीकरण): JSON मध्ये ज्या categories present आहेत, त्यांचे स्वतंत्र headers द्या (खालील साच्यात दिल्याप्रमाणे).

━━━ OUTPUT FORMAT — SEED TREATMENT (is_seed_treatment_query = true असेल तेव्हा हाच वापरा) ━━━

✅ *[crop_display] बीजप्रक्रिया सल्ला* 🌱

🌱 *पेरणीपूर्वी बियाण्यास लावण्यासाठी:*

[प्रत्येक recommendations.seed_treatment[] entry साठी — 2+ असतील तर *पर्याय १*, *पर्याय २*:]
🧪 *[पर्याय १ / पर्याय २ ...]:*
- *घटक:* [chemical_name][is_combination_product true: " (संयुक्त औषध)"]
[has_brand_info true:] - *बाजारातील नावे:* [brands[] max ३][companies[] असतील: | [companies max २]]
[dosage.application_method:] - *वापर पद्धत:* [मराठीत — उदा. बियाण्यास चोळून लावा]
- *डोस:* [DOSAGE RULE नुसार]
[dosage.water_dilution:] - *पाणी:* [value] लिटरमध्ये मिसळा

⚠️ *महत्त्वाची टीप:* बीजप्रक्रिया केल्यानंतर बियाणे सावलीत सुकवा, लगेच पेरणी करा.
हातमोजे वापरा — औषध हाताला थेट लागू देऊ नका.

━━━ OUTPUT FORMAT — NORMAL PEST/PGR (is_seed_treatment_query = false असेल तेव्हा हाच वापरा) ━━━

[HEADER LOGIC नुसार:]
✅ *[crop_display] [योग्य title]* [emoji]

🐛 *आढळलेली समस्या:* [targets_resolved — पूर्ण मराठीत, स्वल्पविरामाने]

[mapped_from_symptom = true असेल:]
🔍 *(लक्षणांवरून ओळखले — प्रत्यक्ष पाहून खात्री करा)*

[summary.has_bio_options = true असेल:]
🌿 *IPM सल्ला:* जैविक उपाय आधी वापरून पहा — रासायनिक उपाय शेवटचा पर्याय.

[overlap_best_matches[] रिकामे नसेल:]
🎯 *सर्व समस्यांसाठी उपयुक्त (All-in-One):*
[त्यातील पर्याय खालील साच्यानुसार द्या]

[पुढील प्रत्येक Category जर JSON मध्ये PRESENT असेल तरच त्याचे Header आणि त्याखालील पर्याय द्या:]

[जर recommendations.bio_pesticide असेल:]
🌿 *जैविक उपाय (Bio-Pesticides):*
[त्यातील पर्याय १, पर्याय २...]

[जर recommendations.herbicide असेल:]
🌿 *तणनाशक (Weedicides):*
[त्यातील पर्याय १, पर्याय २...]

[जर recommendations.fungicide असेल:]
🍃 *बुरशीनाशक (Fungicides):*
[त्यातील पर्याय १, पर्याय २...]

[जर recommendations.insecticide असेल:]
🐛 *कीटकनाशक (Insecticides):*
[त्यातील पर्याय १, पर्याय २...]

[जर recommendations.pgr असेल:]
🌱 *वाढ नियंत्रक / टॉनिक (PGR):*
[त्यातील पर्याय १, पर्याय २...]

🧪 *[पर्याय १ / पर्याय २ ...] साचा:*
- *घटक:* [chemical_name][is_combination_product true: " (संयुक्त औषध)"]
[has_brand_info true:] - *बाजारातील नावे:* [brands[] max ३][companies[] असतील: | [companies max २]]
[dosage.application_method:] - *वापर पद्धत:* [मराठीत]
- *डोस:* [formulation_dose_per_acre किंवा formulation_dose] [dosage.dose_unit वरून एकक — किलो/ग्राम/मिली/लिटर, null असेल तर एकक न लिहिता लेबल-पहा टीप] [प्रति एकर / प्रति हेक्टर] [dose_unit_confidence == "low": (अंदाजे — पक्के प्रमाण लेबलवर तपासा)] [formulation_dose_per_15L_pump असेल: *(१५ लिटर पंपासाठी: [formulation_dose_per_15L_pump] [तेच एकक])*] [ai_dose असेल: (सक्रिय घटक: [ai_dose])]
[water_dilution_per_acre किंवा water_dilution:] - *पाणी:* [value] लिटर पाणी [प्रति एकर / प्रति हेक्टर]
[dosage.waiting_period:] - *काढणीपूर्वी थांबा (PHI):* [value मराठीत (उदा. ५५ दिवस)]
[pests_covered[] रिकामे नसेल:] - *लागू:* [pests_covered — पूर्ण मराठीत भाषांतरित करून]
[diy_homemade_options[] — bio_pesticide साठी:]
  🏡 *घरगुती पर्याय:* [name] — [ingredients] | कृती: [method]

⚠️ *महत्त्वाची टीप:* फवारणीपूर्वी औषधाच्या बाटलीवरील लेबल आणि PPE (हातमोजे, मास्क, डोळ्यांचे रक्षण) नक्की तपासा.

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL ANTI-HALLUCINATION RULES (safety-critical — एकही तोडू नका) ━━━
- DOSE: dosage मधील values null/रिकामे → "कृषी सेवा केंद्रात विचारा" — स्वतःहून कधीही सांगू नका.
- NO ROUNDING: dosage.formulation_dose, dosage.ai_dose, किंवा कोणतेही डोस फक्त JSON मधील EXACT values वापरा — round off करू नका.
- NO MATH: dosage.formulation_dose_per_acre किंवा water_dilution_per_acre असल्यास फक्त JSON मधील exact number वापरा — स्वतः ÷2.47 calculate करण्याची चूक करू नका.
- UNITS: प्रति हेक्टर किंवा प्रति एकर आकडा एकक न सांगता कधीही देऊ नका. ⚠️ एकक फक्त dosage.dose_unit मधूनच घ्या (kg/g/ml/l → किलो/ग्राम/मिली/लिटर) — formulation type (WP/WG/EC/SC इ.) पाहून किंवा स्वतःच्या अंदाजाने एकक कधीही ठरवू नका. dose_unit_confidence == "low" असेल तर "(अंदाजे — पक्के प्रमाण लेबलवर तपासा)" ही caution जोडा.
- BRANDS: has_brand_info = false → brands/companies section पूर्णपणे skip करा. नवे नाव कधीही जोडू नका.
- BIFURCATION: JSON मध्ये नसलेली कोणतीही category (उदा. fungicide नसेल तर) output मध्ये दाखवू नका.
- LENGTH LIMIT (CRITICAL): WhatsApp च्या 4000 अक्षरांच्या मर्यादेमुळे, प्रत्येक Category मध्ये (उदा. कीटकनाशक) जास्तीत जास्त ३ च पर्याय (Options) दाखवा. JSON मध्ये ५ असले तरी फक्त पहिले ३ द्या आणि उर्वरित वगळा.
- JSON key नावे output मध्ये छापू नका. OUTPUT FORMAT मधील [ ] brackets output मध्ये छापू नका.
- is_seed_treatment_query = true असेल तर वरची SEED TREATMENT OUTPUT FORMAT template वापरा.
- शेवटी कोणतेही "पुढची पायरी" / हवामान / मंडी सुचवण्याचे वाक्य स्वतःहून लिहू नका.
"""

    formatted = await _call_gemini(prompt)

    # Deterministic cross-service nudge (weather/mandi) -- computed from the
    # raw JSON in Python, appended after Gemini's output rather than left to
    # Gemini to invent. See _cross_service_suggestions() docstring for the
    # exact relations used.
    suggestion = _cross_service_suggestions(data)

    return formatted + suggestion

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
        if error_type in ("NO_MARKETS_IN_RADIUS",):
            return _error_response(f"⚠️ {error_reason}")
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
