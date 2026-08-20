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
heavy_rain_days / heat_stress_days मधील संख्या म्हणजे day index (1=आज).
day index → date: daily_preview[index-1]["date"] वरून घ्या.

━━━ JSON STRUCTURE (हे top-level keys आहेत) ━━━
season_to_date       → जून १ पासून आत्तापर्यंतचा एकूण पाऊस (harvest_date दिल्यास येतो)
horizon_1_forecast   → मुख्य 0-16 दिवसांचा अंदाज (हे नेहमी असते)
horizon_2_subseasonal → ECMWF 3-4 आठवड्यांचा sub-seasonal अंदाज (harvest_date दिल्यास येतो)
horizon_3_seasonal   → NASA POWER + ENSO दीर्घकालीन मासिक अंदाज (harvest_date दिल्यास येतो)
enso_iod_state       → ENSO/IOD मान्सून स्थिती (ONI + DMI values)
days_to_harvest      → काढणीपर्यंत किती दिवस उरले
partial_data         → true असेल तर horizon_2/3 चा डेटा अपूर्ण आहे

━━━ FIELD MEANINGS — season_to_date ━━━
monsoon_start           → मान्सून हंगाम सुरुवात (सामान्यतः जून १)
accumulated_rain_mm     → जून १ पासून आत्तापर्यंत एकूण पाऊस (मिमी)
agronomic_context       → Horizon 3 च्या normals शी तुलना करा हे string सांगते
⚠️ INTERPRET: accumulated_rain_mm ला horizon_3 च्या त्या महिन्याच्या rainfall_normal_mm शी तुलना करा.
  जास्त पाऊस → पूर/ओला रोग धोका सांगा | कमी पाऊस → सिंचन/दुष्काळ धोका सांगा.

━━━ FIELD MEANINGS — horizon_1_forecast च्या आत ━━━
source                  → "open_meteo" = पूर्ण डेटा | "nasa_power_fallback" = ड्रोन
                          गणना उपलब्ध नाही (हे operational_factors मध्ये सांगितले असेल)
crop_stress_risk_level  → LOW/MEDIUM/HIGH — पिकावरचा एकूण ताण (multiple factors असल्यास HIGH)
crop_stress_factors[]   → कशामुळे ताण: उष्णता, जड पाऊस, पाण्याची कमतरता, कीड/रोग धोका
operational_risk_level  → LOW/MEDIUM/HIGH — शेत कामांना अडचण
operational_factors[]   → कोणती कामे करता येणार नाहीत आणि का
next_rain_date          → पुढचा पाऊस (≥2mm) कधी येईल (YYYY-MM-DD → मराठीत सांगा)
next_dry_spell          → {{ start_date, end_date, days }} — कोरड्या दिवसांचा काळ
                          फवारणी, सिंचन, काढणीसाठी ही खिडकी वापरा
optimal_drone_spray_dates[] → Delta-T (2-8°C) + वारा (<15km/h) + पाऊस (<2mm) या तिन्ही
                          अटी पूर्ण असलेले दिवस — हे वैज्ञानिकदृष्ट्या सर्वोत्तम फवारणीचे दिवस
pest_disease_risk_windows[] → सलग 3+ दिवस RH>85% + तापमान 25-32°C असेल तर कीड/बुरशी धोका
wind_risk_days[]        → दैनंदिन कमाल वारा >20km/h चे दिवस — हे DAILY MAXIMUM आहे, सरासरी नाही.
                          महाराष्ट्रात मान्सूनमध्ये हे सामान्य आहे.
                          ⚠️ संपूर्ण दिवस टाळू नका — best_spray_window_by_day मध्ये
                          सकाळच्या वेळेत अनुकूल खिडकी असू शकते.
                          फक्त दुपारी/जोरदार वाऱ्याच्या वेळी टाळा.
heavy_rain_days[]       → जड पाऊस (पीक-specific threshold ओलांडलेले) day index (1=आज)
heat_stress_days[]      → उष्णता (पीक-specific max temp ओलांडलेले) day index (1=आज)
gdd_accumulated_forecast_window → पुढील 16 दिवसांत पीक किती "उष्णता एकक" मिळवणार
                          (GDD = Growing Degree Days — पिकाच्या वाढीचे शास्त्रीय मोजमाप)
                          जास्त GDD = पीक वेगाने पुढच्या अवस्थेत जाईल
growth_stage            → पिकाची सध्याची अवस्था (sowing_date असल्यास):
                          germination=उगवण | vegetative=वाढ | flowering=फुलोरा ⚠️सर्वात नाजूक
                          pod_fill=शेंगा भरणे | maturity=पक्वता
irrigation_recommended  → हे field null असते — net_water_balance_7d वरूनच सिंचन सल्ला द्या
                          null असेल तर हे field पूर्णपणे skip करा, output मध्ये कधीही null छापू नका
irrigation_recommendation_status → हे field असेल तर पूर्णपणे skip करा — output मध्ये दाखवू नका
net_water_balance_7d    → ७ दिवसांत पाऊस minus वाफ (ET0) = निव्वळ पाणीसाठा (mm)
                          >0 = पाणी जास्त (ओलसर) | -5 ते 0 = किरकोळ कमतरता
                          -5 ते -10 = सिंचन करा | < -10 = गंभीर कमतरता ⚠️ तातडीने सिंचन
et0_7d_mm               → ७ दिवसांत जमिनीतून वाफ होणारे पाणी (mm) — FAO Penman-Monteith
rainfall_7d_mm          → ७ दिवसांतील एकूण अंदाजित पाऊस (mm)

━━━ FIELD MEANINGS — daily_preview (प्रत्येक दिवसाचे fields) ━━━
date        → YYYY-MM-DD (मराठीत "७ ऑगस्ट" सांगा)
rain_mm     → त्या दिवशी किती पाऊस अपेक्षित
et0_mm      → त्या दिवशी किती पाणी वाफ होईल (सिंचन गरज मोजण्यासाठी)
t_max_c     → जास्तीत जास्त तापमान (°C)
t_min_c     → किमान तापमान (°C)
rh_max_pct  → जास्तीत जास्त आर्द्रता (%) — >85% + उबदार = बुरशी/कीड धोका
rh_min_pct  → किमान आर्द्रता (%) — Delta-T गणनेसाठी वापरली जाते
wind_kmh      → वाऱ्याचा वेग km/h — >20 = फवारणी टाळा | 3-15 = आदर्श
wind_gust_kmh → जोरदार वाऱ्याचा वेग km/h — wind_kmh पेक्षा जास्त असेल तर ⚠️ फवारणी धोकादायक
               null असेल तर हे field पूर्णपणे skip करा
wcode         → WMO हवामान कोड (खाली अर्थ पहा):
              0=स्वच्छ ☀️ | 1-3=ढगाळ 🌤️ | 45-48=धुके 🌫️
              51-67=पाऊस 🌧️ | 80-82=मुसळधार 🌧️⚠️ | 95-99=मेघगर्जना ⛈️
              ⚠️ wcode 95-99 असेल त्या दिवशी शेतात काम करू नका

━━━ FIELD MEANINGS — horizon_2_subseasonal ━━━
valid_window    → हा अंदाज कोणत्या कालावधीसाठी आहे
weekly_outlook[] → {{ week, dates, projected_rain_mm, trend }}
trend values:
  "normal_or_wet" → सामान्य किंवा जास्त पाऊस अपेक्षित
  "dry_anomaly"   → ⚠️ कमी पाऊस — या काळात सिंचन तयार ठेवा

━━━ FIELD MEANINGS — horizon_3_seasonal (मासिक दीर्घकालीन अंदाज) ━━━
monthly_outlook[] प्रत्येक महिन्यासाठी:
  rainfall_normal_mm      → त्या महिन्याचा ३०-वर्षांचा सरासरी पाऊस (NASA POWER)
  rainfall_adjusted_mm    → ENSO/IOD ग्राह्य धरून समायोजित अंदाज
  rainfall_pct_of_normal  → सामान्याच्या किती टक्के:
                            >110% = खूप जास्त पाऊस 🌊 | 90-110% = सामान्य ✅
                            75-90% = कमी पाऊस ⚠️ | <75% = टंचाई धोका 🔴
  t_max_normal_c          → सामान्य कमाल तापमान (°C)
  adjustment_basis        → "ENSO=..., IOD=..." = मान्सून काळात ENSO/IOD वापरला
                            "climatology only" = मान्सूनबाहेर — फक्त ऐतिहासिक सरासरी

━━━ FIELD MEANINGS — enso_iod_state ━━━
oni_phase   → el_nino = मान्सून कमकुवत होण्याची शक्यता (कमी पाऊस)
              la_nina = मान्सून जोरदार राहण्याची शक्यता (जास्त पाऊस)
              neutral = सामान्य
oni_value   → ONI निर्देशांक संख्या: >0.5=El Niño, <-0.5=La Niña (ताकद दर्शवतो)
dmi_phase   → positive_iod = El Niño चा परिणाम कमी करतो (भारतात पाऊस वाढतो)
              negative_iod = La Niña चा परिणाम कमी करतो (पाऊस घटतो)
              neutral = सामान्य
dmi_value   → DMI निर्देशांक संख्या (ताकद दर्शवतो)
COMBINED EFFECT (हे महत्वाचे):
  La Niña + positive_iod  → ⛈️ खूप जास्त पाऊस — पूर/ओला रोग सावधानता
  La Niña + negative_iod  → ≈ mixed — थोडा जास्त पाऊस
  El Niño + positive_iod  → ≈ balanced — सामान्य पाऊस शक्य
  El Niño + negative_iod  → 🔴 दुष्काळ धोका — सिंचन योजना आखा

━━━ FIELD MEANINGS — best_spray_window_by_day (नवीन) ━━━
हे field असेल तर → प्रत्येक दिवसासाठी सर्वोत्तम फवारणी वेळ दिला आहे.
structure: {{ "YYYY-MM-DD": {{ window_start, window_end, avg_wind_kmh, max_wind_kmh,
             max_gust_kmh, rain_mm, rain_probability_max_pct, spray_status }} }}
spray_status values:
  "favorable"   → ✅ फवारणीसाठी उत्तम वेळ
  "caution"     → ⚠️ सावधगिरीने — वारा थोडा जास्त, लक्ष ठेवा
  "unfavorable" → ❌ फवारणी टाळा
window_start/end → "2026-08-20T06:00" format — "सकाळी ६ ते ९" असे मराठीत सांगा
max_gust_kmh → null असेल तर skip करा
rain_probability_max_pct → null असेल तर skip करा
⚠️ एखाद्या दिवसाचा value null असेल (best window नाही) → त्या दिवशी "फवारणीसाठी चांगली वेळ नाही" सांगा

spray_windows → हे full hourly breakdown आहे — output मध्ये दाखवू नका, फक्त best_spray_window_by_day वापरा

━━━ KEY THRESHOLDS (interpret करताना वापरा) ━━━
net_water_balance_7d:
  > 0        → जमिनीत पाणी पुरेसे आहे
  -5 ते 0    → किरकोळ कमतरता — पाण्यावर लक्ष ठेवा
  -5 ते -10  → irrigation_recommended=true असेल — सिंचन करा
  < -10      → गंभीर कमतरता — तातडीने सिंचन करा ⚠️
Delta-T (optimal_drone_spray_dates मधून अप्रत्यक्षपणे):
  2-8°C = आदर्श फवारणी (रसायन पिकापर्यंत नक्की पोहोचते)
  <2°C = धुके/inversion — फवारणी टाळा (drift होते)
  >8°C = खूप उष्ण/कोरडे — थेंब वाफ होतात, रसायन वाया (सकाळी लवकर/संध्याकाळी करा)
  ⚠️ optimal_drone_spray_dates रिकामा असेल तर पुढील काही दिवस फवारणीसाठी चांगले नाहीत
wind_kmh (daily_preview) — हा दिवसाचा कमाल वेग आहे, सरासरी नाही:
  < 15 km/h  → दिवसभर फवारणीसाठी योग्य
  15-25 km/h → सकाळी लवकर किंवा संध्याकाळी वेळ निवडा — best_spray_window_by_day पहा
  > 25 km/h  → ⚠️ जोरदार वारा — best_spray_window_by_day मध्ये योग्य वेळ असेल तरच फवारणी करा
  > 35 km/h  → ❌ फवारणी टाळा — रसायन शेजारच्या शेतात जाते, गंभीर नुकसान
wind_gust_kmh (daily_preview) — हा अधिक महत्वाचा आकडा आहे:
  < 20 km/h  → ड्रोन + manual sprayer दोन्ही ठीक
  20-30 km/h → manual knapsack ठीक — ड्रोन DGCA नियमानुसार टाळा
  > 30 km/h  → ⚠️ कोणतीही फवारणी टाळा — रसायन वाया जाते
  null असेल → wind_kmh वरूनच निर्णय घ्या
growth_stage + net_water_balance सल्ला:
  flowering (फुलोरा) + negative balance → ⚠️ तातडीने सिंचन — या अवस्थेत पाण्याची कमतरता उत्पादनावर थेट परिणाम करते
  pod_fill + heavy_rain → बुरशीजन्य रोगांसाठी prophylactic फवारणी विचार करा

━━━ HOW TO ANSWER (शेतकऱ्याचा प्रश्न वाचा, त्यानुसार उत्तर द्या) ━━━

फवारणी बद्दल विचारले (आज/उद्या/परवा) →
  Step 1 — त्या दिवसाचा daily_preview मधून: rain_mm, wind_kmh, wind_gust_kmh (null नसेल तर), wcode.
  Step 2 — best_spray_window_by_day मध्ये त्या दिवसाची entry पहा:
    spray_status = "favorable" → ✅ "[window_start→मराठी वेळ] ते [window_end→मराठी वेळ]" ही फवारणीसाठी सर्वोत्तम वेळ
    spray_status = "caution"   → ⚠️ "[वेळ]" ला फवारणी शक्य — वारा [max_wind_kmh] km/h, सावधगिरीने
    spray_status = "unfavorable" किंवा null → त्या दिवशी फवारणीसाठी चांगली वेळ नाही
  Step 3 — wind_risk_days मध्ये तो दिवस असेल:
    → best_spray_window_by_day मध्ये "favorable" किंवा "caution" असेल → त्या वेळेत फवारणी शक्य आहे असे सांगा
    → best_spray_window_by_day मध्ये सर्व "unfavorable" असेल → ⚠️ आज वारा जास्त, उद्या पहा
    → "दुपारी वारा जास्त असतो — सकाळी लवकर फवारणी करा" हा सल्ला नेहमी द्या
  Step 4 — wcode 95-99 असेल → ⚠️ मेघगर्जना — शेतात जाऊ नका
  Step 5 — pest_disease_risk_windows असेल → फवारणीची तातडी आहे असे सांगा
  Step 6 — optimal_drone_spray_dates मध्ये तो दिवस असेल → ✅ ड्रोन फवारणीसाठी आदर्श — Delta-T योग्य आहे
  best_spray_window_by_day रिकामा किंवा null असेल → "पुढील काही दिवस फवारणीसाठी हवामान अनुकूल नाही"

पाऊस बद्दल विचारले →
  next_rain_date सांगा (मराठी date).
  daily_preview मधून प्रत्येक दिवसाचा rain_mm + wcode emoji सांगा.
  next_dry_spell असेल तर — "या काळात फवारणी/काढणी योग्य संधी"
  season_to_date accumulated_rain_mm सांगा (असल्यास) — हंगाम कसा चालू आहे.

सामान्य हवामान / पुढील काही दिवस →
  daily_preview प्रत्येक दिवस: date→मराठी, rain_mm, t_max_c, wind_kmh, wcode→emoji.
  crop_stress_risk_level + factors सांगा.
  net_water_balance_7d वरून: सिंचन सल्ला (threshold प्रमाणे).
  irrigation_recommended=true असेल → स्पष्ट सिंचन सल्ला द्या.
  growth_stage असेल → त्याला अनुसरून विशेष सल्ला द्या.
  gdd_accumulated_forecast_window असेल → "पुढील 16 दिवसांत पीक [GDD] उष्णता एकके मिळवेल"
  next_dry_spell असेल तर सांगा.
  horizon_2 असेल तर weekly_outlook: "dry_anomaly" = सिंचन तयार ठेवा सांगा.

दीर्घकालीन / हंगाम बद्दल विचारले →
  horizon_3_seasonal: rainfall_pct_of_normal प्रत्येक महिन्यासाठी सांगा.
  >110% → जास्त पाऊस, बुरशी/पूर सावधानता
  <90% → कमी पाऊस, सिंचन योजना आखा
  enso_iod_state: combined effect (वर दिलेले) वापरून मान्सून outlook सांगा.
  season_to_date असेल → हंगाम आत्तापर्यंत surplus/deficit कसा आहे ते सांगा.

━━━ OUTPUT FORMAT ━━━
*[पीक] हवामान अंदाज* 🌾

[season_to_date असेल तर:]
🌧️ *हंगाम पाऊस (जून १ पासून):* [accumulated_rain_mm] मिमी
[surplus/deficit interpretation येथे]

☀️ *पुढील [N] दिवस:*
- [मराठी date] [wcode emoji]: [rain_mm] मिमी पाऊस, [t_max_c]°C, वारा [wind_kmh] km/h[wind_gust_kmh null नसेल: (जोर [wind_gust_kmh])]
[daily_preview मधील पहिले ७ दिवस — बाकी शेतकऱ्याने विचारल्यासच]

💧 *पाण्याचा ताळेबंद (७ दिवस):* [net_water_balance_7d] मिमी
(पाऊस [rainfall_7d_mm] मिमी − वाफ [et0_7d_mm] मिमी)
[negative threshold नुसार: किरकोळ / सिंचन करा / तातडीने सिंचन ⚠️]

[growth_stage असेल तर:]
🌱 *पीक अवस्था:* [growth_stage मराठीत] | [days_to_harvest असेल: काढणी [N] दिवसात]
[GDD असेल: पुढील 16 दिवसांत [gdd] उष्णता एकके मिळणार — पीक वेगाने पुढे जाईल/हळू]
[growth_stage विशेष सल्ला येथे]

🚜 *फवारणी सल्ला:*
[best_spray_window_by_day असेल तर पहिल्या ३ दिवसांसाठी:]
- [मराठी date]: [spray_status emoji] [window_start→मराठी वेळ] ते [window_end→मराठी वेळ] | वारा [max_wind_kmh] km/h[max_gust_kmh null नसेल: , जोर [max_gust_kmh] km/h]
[null entry असेल त्या दिवशी:] - [मराठी date]: ❌ फवारणीसाठी चांगली वेळ नाही
[optimal_drone_spray_dates रिकामे नसेल:]
✅ *ड्रोन फवारणी:* [dates → मराठीत] — Delta-T + वारा आदर्श[wind_risk_days रिकामे नसेल:]
⏰ *वेळ महत्वाची:* [wind_risk_days मराठीत] — दिवसभर कमाल वारा जास्त आहे, पण सकाळी लवकर (६-९) खिडकी असू शकते — वर पहा
[pest_disease_risk_windows असेल:] 🐛 *कीड/बुरशी धोका:* [dates → मराठीत] — आत्ताच फवारणी विचार करा

🌡️ *पीक ताण:* [crop_stress_risk_level मराठीत]
[crop_stress_factors असतील:] कारण: [factors]
[operational_factors असतील:] ⚠️ [factors]

[horizon_2 असेल तर:]
📅 *पुढील महिन्याचा अंदाज (ECMWF):*
- [week 3 dates]: [projected_rain_mm] मिमी — [trend: dry_anomaly→कमी पाऊस⚠️ सिंचन तयार ठेवा | normal_or_wet→सामान्य/जास्त पाऊस]
- [week 4 dates]: [projected_rain_mm] मिमी — [trend मराठीत]

[horizon_3 असेल तर:]
📆 *दीर्घकालीन अंदाज (NASA):*
[प्रत्येक month: month_name + rainfall_pct_of_normal interpretation]

[enso_iod_state असेल तर:]
🌏 *मान्सून स्थिती:* [oni_phase मराठीत] ([oni_value असेल: ONI [value]])
[dmi_phase मराठीत] ([dmi_value असेल: DMI [value]])
[combined effect मराठीत — वर दिलेल्या combination logic नुसार]

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ FINAL RULES ━━━
- JSON key नावे कधीही छापू नका (horizon_1_forecast, daily_preview, wcode, etc.)
- Dates नेहमी "७ ऑगस्ट" format — YYYY-MM-DD कधीही नको
- heavy_rain_days/heat_stress_days मधील संख्या → daily_preview मधून actual date काढा
- partial_data=true असेल तर: "काही दीर्घकालीन डेटा उपलब्ध नाही"
- source="nasa_power_fallback" असेल: "ड्रोन फवारणी गणना सध्या उपलब्ध नाही"
- irrigation_recommended null असेल → हे field output मध्ये कधीही छापू नका
- irrigation_recommendation_status field → output मध्ये कधीही छापू नका
- spray_windows (full hourly data) → output मध्ये कधीही छापू नका — फक्त best_spray_window_by_day वापरा
- wind_gust_kmh null असेल → त्या दिवशी गस्ट mention करू नका
- best_spray_window_by_day मधील value null असेल → "चांगली वेळ नाही" सांगा, null छापू नका
- daily_preview पहिले ७ दिवसच सामान्य उत्तरात — शेतकऱ्याने "पुढील X दिवस" विचारल्यास अधिक दाखवा
- JSON मध्ये नसलेले field/section पूर्णपणे skip करा
- adjustment_basis "climatology only" असेल तर ENSO/IOD mention करू नका त्या महिन्यासाठी
- rainfall_pct_of_normal शिवाय दीर्घकालीन पाऊस सांगू नका — हा आकडा अनिवार्य आहे
- wind_risk_days असेल → "फवारणी टाळा" असे सरळ सांगू नका — best_spray_window_by_day तपासा आणि वेळेचा सल्ला द्या
- महाराष्ट्र मान्सून context: 15-25 km/h वारा सामान्य आहे — हे शेतकऱ्याला माहीत आहे, फक्त योग्य वेळ सांगा
- दिवसाचा कमाल वारा (wind_kmh) वापरून संपूर्ण दिवस block करू नका — time-specific सल्ला द्या
- wind_gust_kmh हा sustained wind_kmh पेक्षा अधिक महत्वाचा आहे फवारणी निर्णयासाठी
- ड्रोन फवारणी: DGCA नियमानुसार gusts <20 km/h असणे आवश्यक — हे optimal_drone_spray_dates मध्ये आधीच तपासले आहे"""

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
recommendations                     → हे एक OBJECT आहे, LIST नाही! खालील keys असू शकतात:
  .overlap_best_matches[]           → एकापेक्षा जास्त कीड/रोग असतील — सर्वांवर चालणारी औषधे
  .insecticide[]                    → कीटकनाशके
  .bio_pesticide[]                  → जैविक/सेंद्रिय उपाय
  .fungicide[]                      → बुरशीनाशके
  .herbicide[]                      → तणनाशके
  .pgr[]                            → वाढ नियंत्रक (is_pgr_query=true असताना)
  (JSON मध्ये जी key PRESENT आहे तीच सांगा — रिकामी key पूर्णपणे skip करा)
summary.total_options               → एकूण किती पर्याय सापडले
summary.has_bio_options             → जैविक पर्याय उपलब्ध आहे का
summary.has_branded_options         → बाजारातील ब्रँड नावे उपलब्ध आहेत का

━━━ FIELD MEANINGS — प्रत्येक recommendation entry च्या आत ━━━
chemical_name            → रासायनिक घटकाचे शास्त्रीय/CIBRC नाव (हेच खरे नाव — बाटलीवर तपासा)
category                 → insecticide=कीडनाशक | fungicide=बुरशीनाशक |
                            herbicide=तणनाशक | bio_pesticide=जैविक | pgr=वाढ नियंत्रक
is_combination_product   → true असेल तर हे दोन घटकांचे मिश्रण आहे — तसे नमूद करा
covers_all_pests         → true असेल तर सर्व कीड/रोगांवर हे एकच औषध चालते
pests_covered[]          → हे औषध नक्की कोणत्या कीड/रोगासाठी आहे
dosage.ai_dose           → active ingredient प्रमाणे डोस (असल्यास)
dosage.formulation_dose  → प्रत्यक्ष बाटली/पाकिटावरील फॉर्म्युलेशन डोस — शेतकऱ्यासाठी हाच सर्वात उपयोगी
dosage.water_dilution    → किती लिटर पाण्यात मिसळायचे
dosage.waiting_period    → PHI — काढणीपूर्वी किती दिवस थांबायचे — नेहमी सांगा, असल्यास
dosage.application_method → फवारणी/मातीत/बियाण्यावर — कशा प्रकारे वापरायचे
brands[]                 → बाजारात मिळणाऱ्या औषधांची नावे — यादीतीलच नावे सांगा
companies[]              → या ब्रँड्स बनवणाऱ्या कंपन्या
has_brand_info           → false असेल तर brands/companies section पूर्णपणे skip करा
diy_homemade_options[]   → घरगुती/DIY उपाय (फक्त bio_pesticide साठी) — प्रत्येकात name, ingredients, method

━━━ DOSAGE RULE (safety-critical) ━━━
dosage.formulation_dose असेल → तेच exact value वापरा (आधी दाखवा — सर्वात प्रॅक्टिकल)
dosage.ai_dose सुद्धा असेल → "(AI dose: [value])" असे कंसात नंतर दाखवा
dosage.water_dilution असेल → "[X] लिटर पाण्यात मिसळा"
dosage.waiting_period असेल → "काढणीपूर्वी [N] थांबा" — नेहमी सांगा
dosage.application_method असेल → वापर पद्धत ओळ द्या
सर्व dosage values null/रिकामे असतील → "डोससाठी कृषी सेवा केंद्रात विचारा" — स्वतःहून कधीही सांगू नका

━━━ BRANDS RULE ━━━
has_brand_info = false → brands आणि companies section पूर्णपणे skip करा
has_brand_info = true → brands[] मधून जास्तीत जास्त ३ नावे + companies[] मधून जास्तीत जास्त २ नावे
brands[] रिकामे पण companies[] असतील → फक्त companies दाखवा

━━━ HEADER LOGIC (resolved_parameters वरून ठरवा) ━━━
is_pgr_query = true                     → ✅ *[crop_display] ग्रोथ बूस्टर सल्ला* 🌱
mapped_from_symptom = true              → ✅ *[crop_display] लक्षणावरून ओळखलेली समस्या* 🔍
recommendations मध्ये फक्त fungicide[]  → ✅ *[crop_display] रोग व्यवस्थापन सल्ला* 🍃
recommendations मध्ये फक्त herbicide[] → ✅ *[crop_display] तण व्यवस्थापन सल्ला* 🌿
इतर सर्व (कीड/मिश्र)                   → ✅ *[crop_display] पीक संरक्षण सल्ला* 🧪

━━━ HOW TO ANSWER ━━━

नियम १ — overlap_best_matches[] असेल आणि रिकामे नसेल:
  हे सर्वात आधी दाखवा — "🎯 सर्व समस्यांसाठी उपयुक्त:" असे header देऊन.
  नंतर बाकी categories त्यांच्या स्वतःच्या headers खाली दाखवा.
  overlap मध्ये आधीच दाखवलेले chemical_name पुन्हा खालच्या यादीत छापू नका.

नियम २ — IPM hierarchy:
  summary.has_bio_options = true असेल → bio_pesticide section आधी दाखवा.
  "🌿 जैविक उपाय आधी वापरून पहा — रासायनिक उपाय शेवटचा पर्याय" असा सल्ला द्या.

नियम ३ — Multiple recommendations (कोणत्याही category मध्ये 2+ items):
  प्रत्येकासाठी स्वतंत्र *पर्याय १*, *पर्याय २* block द्या.
  स्वतः "हा best आहे" असे कधीही म्हणू नका — सर्व options शेतकऱ्याला द्या.

नियम ४ — diy_homemade_options[] असेल (bio_pesticide entries मध्ये):
  त्या recommendation च्या खाली "🏡 घरगुती पर्याय:" असे sub-section द्या.
  name, ingredients/साहित्य, method/कृती थोडक्यात सांगा.

नियम ५ — is_combination_product = true:
  "(हे दोन घटकांचे संयुक्त औषध आहे)" असे नमूद करा.

नियम ६ — targets_resolved यादी:
  🐛 *आढळलेली समस्या:* [targets_resolved स्वल्पविरामाने] असे सांगा.

नियम ७ — mapped_from_symptom = true:
  🔍 "(लक्षणांवरून ओळखले — प्रत्यक्ष पाहून खात्री करा)" असे सांगा.

━━━ OUTPUT FORMAT ━━━

[HEADER LOGIC नुसार:]
✅ *[crop_display] [योग्य title]* [emoji]

🐛 *आढळलेली समस्या:* [targets_resolved — स्वल्पविरामाने]

[mapped_from_symptom = true असेल:]
🔍 *(लक्षणांवरून ओळखले — प्रत्यक्ष पाहून खात्री करा)*

[summary.has_bio_options = true असेल:]
🌿 *IPM सल्ला:* जैविक उपाय आधी वापरून पहा — रासायनिक उपाय शेवटचा पर्याय.

[overlap_best_matches[] रिकामे नसेल:]
🎯 *सर्व समस्यांसाठी उपयुक्त:*
[प्रत्येक overlap entry साठी recommendation block — खाली पहा]

[प्रत्येक category साठी (जे present आहे — bio_pesticide आधी, नंतर insecticide/fungicide/herbicide/pgr):]
🧪 *[पर्याय १ / पर्याय २ ...] — [category मराठीत]:*
- *घटक:* [chemical_name][is_combination_product true: " (संयुक्त औषध)"]
[has_brand_info true:] - *बाजारातील नावे:* [brands[] max ३][companies[] असतील: | [companies max २]]
[dosage.application_method:] - *वापर पद्धत:* [मराठीत]
- *डोस:* [DOSAGE RULE नुसार — formulation_dose आधी, ai_dose कंसात]
[dosage.water_dilution:] - *पाणी:* [value] लिटरमध्ये मिसळा
[dosage.waiting_period:] - *काढणीपूर्वी थांबा (PHI):* [value]
[pests_covered[] रिकामे नसेल:] - *लागू:* [pests_covered — स्वल्पविरामाने]
[diy_homemade_options[] — bio_pesticide साठी:]
  🏡 *घरगुती पर्याय:* [name] — [ingredients] | कृती: [method]

⚠️ *महत्त्वाची टीप:* फवारणीपूर्वी औषधाच्या बाटलीवरील लेबल आणि PPE
(हातमोजे, मास्क, डोळ्यांचे रक्षण) नक्की तपासा. लेबलवरील सूचना कायद्याने बंधनकारक आहेत.

━━━ JSON DATA ━━━
{json.dumps(data, ensure_ascii=False, indent=2)}

━━━ CRITICAL ANTI-HALLUCINATION RULES (safety-critical — एकही तोडू नका) ━━━
DOSE:
- dosage मधील values null/रिकामे → "कृषी सेवा केंद्रात विचारा" — स्वतःहून कधीही सांगू नका
- dosage.formulation_dose / dosage.ai_dose फक्त JSON मधील exact values — round करू नका
- dosage.formulation_dose_per_acre / dosage.water_dilution_per_acre असल्यास फक्त JSON मधील exact number वापरा — स्वतः ÷2.47 calculate करू नका
- प्रति हेक्टर किंवा प्रति एकर आकडा एकक न सांगता कधीही देऊ नका

BRANDS:
- has_brand_info = false → brands/companies section पूर्णपणे skip करा
- brands[] JSON मधील यादीतीलच — नवे नाव कधीही जोडू नका
- companies[] JSON मधील यादीतीलच — max २ दाखवा

STRUCTURE:
- recommendations OBJECT आहे — overlap_best_matches, insecticide, bio_pesticide इत्यादी sub-keys आहेत
- JSON मध्ये नसलेली कोणतीही category output मध्ये दाखवू नका
- JSON key नावे (recommendations, dosage, chemical_name इ.) output मध्ये छापू नका
- OUTPUT FORMAT मधील [ ] brackets output मध्ये छापू नका — त्या जागी actual data भरा
- overlap मध्ये दाखवलेले chemical पुन्हा category section मध्ये छापू नका
- JSON मध्ये नसलेला कोणताही section पूर्णपणे skip करा"""

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