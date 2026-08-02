"""
orchestrator.py
───────────────
Three-stage pipeline for farmer message understanding.

Stage 1 — PREFLIGHT (gemini-2.0-flash, cheapest)
Stage 2 — EXTRACTION (gemini-2.5-flash, accurate)  
Stage 3 — ROUTING (gemini-2.5-flash, function calling)

Fixes applied:
  ✅ Fix 1: Coming soon endpoints (fertilizer, top3_crops)
  ✅ Fix 2: response_mime_type="application/json" for stages 1+2
  ✅ Fix 3: client.aio.models.generate_content (native async, no threads)
  ⚠️  Fix 3 NOT applied to stage 3 — function calling incompatible with mime type
"""

import asyncio
import json
import logging
import os
from typing import Optional
from google import genai
from google.genai import types
import random

logger = logging.getLogger(__name__)

# ─── API Keys (3 keys, rotated to avoid rate limits) ─────────────────────────

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]

# Replace entire _get_client function with this:
def _get_client() -> genai.Client:
    """
    Returns a Gemini client using a randomly selected API key.
    Random selection on every call = true load balancing across all keys.
    Prevents any single key from hitting RPM limits under concurrent load.
    """
    keys = [k for k in GEMINI_KEYS if k]
    if not keys:
        raise ValueError("No GEMINI_API_KEY set. Need at least GEMINI_API_KEY_1.")
    return genai.Client(api_key=random.choice(keys))


# ─── Models ───────────────────────────────────────────────────────────────────

MODEL_PREFLIGHT  = "gemini-3.1-flash-lite"
MODEL_EXTRACTION = "gemini-3.1-flash-lite"
MODEL_ROUTING    = "gemini-3.1-flash-lite"


# ─── Supported Crops ──────────────────────────────────────────────────────────

SUPPORTED_CROPS = """
Grains: jowar/ज्वारी, wheat/गहू, maize/मका/corn, bajra/बाजरी, rice/भात/paddy, ragi/नाचणी
Pulses: tur/तूर/pigeon pea/arhar, chana/हरभरा/chickpea, moong/मूग, urad/उडीद, masoor/मसूर,
        matki/मठ, peas/वाटाणा, cowpea/लोबिया
Oilseeds: soybean/सोयाबीन, cotton/कापूस, sunflower/सूर्यफूल, groundnut/भुईमूग/peanut,
          safflower/करडई, sesame/तीळ, castor seed/एरंडी
Vegetables: onion/कांदा, potato/बटाटा, tomato/टोमॅटो, brinjal/वांगी, cabbage/कोबी,
            cauliflower/फ्लॉवर, okra/भेंडी, bottle gourd/दुधी, bitter gourd/कारले,
            cucumber/काकडी, capsicum/ढोबळी मिरची, drumstick/शेवगा, spinach/पालक,
            fenugreek/मेथी, radish/मुळा, carrot/गाजर, beetroot/बीटरूट
Spices: chilli/मिरची, garlic/लसूण, turmeric/हळद, ginger/आले, cumin/जिरे,
        coriander seed/धने, black pepper/काळी मिरी, coconut/नारळ, jaggery/गूळ
Fruits: pomegranate/डाळिंब, orange/संत्रा, sweet lime/मोसंबी, mango/आंबा,
        banana/केळी, grapes/द्राक्षे, papaya/पपई, guava/पेरू, watermelon/कलिंगड,
        sapota/चिकू, apple/सफरचंद, lemon/लिंबू, jackfruit/फणस
"""

# ─── Redirects & Coming Soon ──────────────────────────────────────────────────

THINGS_WE_DONT_HANDLE = {
    "loan":      "कर्जासाठी PM Kisan helpline: 155261 वर कॉल करा",
    "seeds":     "बियाण्यांसाठी तुमच्या जवळच्या कृषी केंद्रात जा",
    "insurance": "पीक विम्यासाठी PMFBY helpline: 1800-180-1551",
    "scheme":    "सरकारी योजनांसाठी: agri.maharashtra.gov.in",
    "pest_id":   "किडीची ओळख करण्यासाठी: KVK helpline 1800-180-1551",
    "general":   "माफ करा, हे आम्ही सध्या हाताळत नाही. आम्ही मंडी भाव आणि हवामान माहिती देतो."
}

COMING_SOON_MESSAGES = {
    "fertilizer": {
        "mr": "🌱 खत व्यवस्थापन सेवा लवकरच सुरू होत आहे!\nसध्या आम्ही फक्त *मंडी भाव* आणि *हवामान* माहिती देतो.",
        "hi": "🌱 उर्वरक सिफारिश सेवा जल्द शुरू हो रही है!\nअभी हम सिर्फ *मंडी भाव* और *मौसम* जानकारी देते हैं।",
        "en": "🌱 Fertilizer Recommendation service is coming soon!\nCurrently we provide *mandi prices* and *weather* only.",
    },
    "top_crops": {
        "mr": "📊 'कोणते पीक घ्यावे' ही सेवा लवकरच सुरू होत आहे!\nसध्या आम्ही *मंडी भाव* आणि *हवामान* माहिती देतो.",
        "hi": "📊 'कौन सी फसल उगाएं' सेवा जल्द शुरू हो रही है!\nअभी हम *मंडी भाव* और *मौसम* जानकारी देते हैं।",
        "en": "📊 'Top 3 Crops to Grow' service is coming soon!\nCurrently we provide *mandi prices* and *weather* only.",
    }
}

COMING_SOON_KEYS = set(COMING_SOON_MESSAGES.keys())


# ─── Stage 1: Preflight ───────────────────────────────────────────────────────

async def _preflight(message: str) -> dict:
    client = _get_client()

    prompt = f"""You are a preflight filter for an Indian agriculture WhatsApp bot.

The bot handles ONLY:
1. Mandi prices — where to sell crop, APMC rates, market prices, logistics, transport
2. Weather risk — rain forecast, spray safety, irrigation, crop stress, pest risk windows

Coming soon (not yet available):
- Fertilizer recommendation / खत व्यवस्थापन → redirect_key: "fertilizer"
- Top 3 crops to grow / कोणते पीक घ्यावे → redirect_key: "top_crops"

Does NOT handle at all:
- Crop loans / bank / karz → redirect_key: "loan"
- Seeds / beej / biyane → redirect_key: "seeds"
- Insurance / vima → redirect_key: "insurance"
- Government schemes / yojana → redirect_key: "scheme"
- Pest identification (we give risk windows but not diagnosis) → redirect_key: "pest_id"
- Anything non-agricultural → redirect_key: "general"

Farmer message: "{message}"

Return JSON only:
{{
  "is_agri": true/false,
  "is_handled": true/false,
  "redirect_key": null or "loan"/"seeds"/"insurance"/"scheme"/"pest_id"/"general"/"fertilizer"/"top_crops",
  "reason": "one short English sentence"
}}

Examples:
"soybean bhav kiti" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"mandi price query"}}
"karz hava" → {{"is_agri":true,"is_handled":false,"redirect_key":"loan","reason":"loan request"}}
"khad kuthun milel" → {{"is_agri":true,"is_handled":false,"redirect_key":"fertilizer","reason":"fertilizer coming soon"}}
"konthe pik ghyave" → {{"is_agri":true,"is_handled":false,"redirect_key":"top_crops","reason":"crop selection coming soon"}}
"paus yeil ka" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"weather query"}}
"cricket score" → {{"is_agri":false,"is_handled":false,"redirect_key":"general","reason":"not agriculture"}}"""

    try:
        # Fix 3: native async. Fix 2: response_mime_type for clean JSON
        response = await client.aio.models.generate_content(
            model=MODEL_PREFLIGHT,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"[Orchestrator] Preflight failed: {e}")
        return {
            "is_agri": True, "is_handled": True,
            "redirect_key": None, "reason": "preflight_error_assume_handled"
        }


# ─── Stage 2: Extraction ──────────────────────────────────────────────────────

async def _extract_intent(message: str) -> dict:
    client = _get_client()

    prompt = f"""You are an expert agricultural assistant fluent in Marathi, Hindi, and English.
Extract structured information from this farmer message.
Primary language is Marathi. Hindi and English also supported.

Farmer message: "{message}"

Supported crops (normalize to English name):
{SUPPORTED_CROPS}

Rules:
1. crop: English lowercase only. सोयाबीन/soya/soyabean → soybean. कापूस/kapus → cotton. कांदा → onion
2. qty: number only as string. "100 quintal" → qty="100", qty_unit="quintal"
3. variety: Keep in ORIGINAL SCRIPT. शरबती → variety="शरबती". lokwan → variety="lokwan"
4. time_horizon: "now" unless farmer says pudhe/future/N days → "30_days" format
5. language: "mr" for Marathi/Marathi-in-English-script, "hi" for Hindi, "en" for English
6. needs_clarification: true ONLY if crop completely missing AND cannot be inferred
7. Weather queries can have null crop — weather works without crop

Return JSON only:
{{
  "crop": null,
  "qty": null,
  "qty_unit": null,
  "variety": null,
  "radius_km": null,
  "time_horizon": "now",
  "harvest_date": null,
  "sowing_date": null,
  "forecast_days": 7,
  "language": "mr",
  "raw_intent": "one line summary in English",
  "needs_clarification": false,
  "clarification_aspect": null
}}

Examples:
"सोयाबीनचे भाव काय आहेत?" → crop="soybean", language="mr"
"kanda vikaycha ahe 200 quintal" → crop="onion", qty="200", qty_unit="quintal"
"paus yeil ka pudhe 3 diwas cotton la" → crop="cotton", forecast_days=3
"aaj spray karu ka" → crop=null, raw_intent="spray safety check today"
"mera gehun 50 bag bechna hai" → crop="wheat", qty="50", qty_unit="bag", language="hi"
"lokwan variety sathi bhav" → crop="wheat", variety="lokwan"
"hi" → needs_clarification=true, clarification_aspect="service" """

    try:
        # Fix 3: native async. Fix 2: response_mime_type
        response = await client.aio.models.generate_content(
            model=MODEL_EXTRACTION,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"[Orchestrator] Extraction failed: {e}")
        return {
            "crop": None, "qty": None, "qty_unit": None,
            "variety": None, "radius_km": None, "time_horizon": "now",
            "harvest_date": None, "sowing_date": None, "forecast_days": 7,
            "language": "mr", "raw_intent": "unknown",
            "needs_clarification": True, "clarification_aspect": "service"
        }


# ─── Stage 3: Routing (Function Calling) ─────────────────────────────────────
# NOTE: response_mime_type NOT used here — incompatible with function calling
# Fix 3 (native async) still applied via client.aio

MANDI_TOOL = types.FunctionDeclaration(
    name="get_mandi_prices",
    description="""
Use when farmer wants ANY of:
SELLING / PRICE:
- Current APMC mandi modal price (from Maharashtra govt MSAMB database, 450+ records daily)
- Which mandi to sell at for maximum profit
- Price comparison between multiple mandis
- Net profit after APMC deductions (cess 1.05% + commission 3-8% + hamali)
- Transport cost, vehicle recommendation (Tata Ace/Bolero/14ft truck/10-wheeler by qty)
- Future price estimate using seasonal heuristics (N_days format)
- Nearest active mandis discovery (even without specifying crop)

MARATHI: भाव, दर, किंमत, मोल, बाजार, मंडी, विकणे, नफा, ट्रक, वाहतूक, कमाई
HINDI: भाव, दाम, कीमत, बेचना, मंडी, बाजार, फायदा, मुनाफा, ट्रक
ENGLISH: price, rate, sell, market, mandi, profit, transport, vehicle, apmc

EXAMPLES:
"सोयाबीनचे भाव काय आहेत?" → MANDI
"kanda 200 quintal kuth vikaycha?" → MANDI
"cotton rate Amravati la kiti?" → MANDI
"50 din baad soybean ka rate?" → MANDI (future)
"majhya javalyacha mandi konta?" → MANDI (discovery)
"truck kiti lagel 100 quintal sathi?" → MANDI
""",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "crop": types.Schema(type=types.Type.STRING,
                description="Crop in English lowercase e.g. soybean, onion, cotton"),
            "qty": types.Schema(type=types.Type.STRING,
                description="Quantity as string number e.g. '100'"),
            "time_horizon": types.Schema(type=types.Type.STRING,
                description="'now' or '<N>_days' e.g. '30_days'"),
            "variety": types.Schema(type=types.Type.STRING,
                description="Specific variety in original script e.g. 'शरबती'"),
            "radius_km": types.Schema(type=types.Type.INTEGER,
                description="Search radius in km, default 100"),
        },
        required=["crop"]
    )
)

WEATHER_TOOL = types.FunctionDeclaration(
    name="get_weather_risk",
    description="""
Use when farmer wants ANY of:
WEATHER / RAIN:
- Will it rain? Rain forecast next N days (Open-Meteo ECMWF/NOAA precision)
- Drought risk, water stress, season-to-date rainfall vs normal

FARMING OPERATIONS:
- Spray safety today? (Delta-T calculation: 2-8°C optimal, wind < 15kmh, rain < 2mm)
- Drone spray window safety check
- Irrigation advice based on net water balance (rain - ET0)
- Wind risk days for spray drift

CROP HEALTH:
- Crop stress risk level (LOW/MEDIUM/HIGH) with specific factors
- Pest and disease risk windows (RH > 85% + temp 25-32°C = fungal risk)
- Heat stress days (above crop threshold temperature)
- Heavy rain days that could lodge or damage crop
- Growth stage based on sowing date
- GDD (Growing Degree Days) accumulated in forecast window

SEASONAL (requires harvest_date):
- ECMWF sub-seasonal weeks 3-4 outlook (days 17-35)
- NASA POWER 30yr climatology adjusted by ENSO/IOD phase
- Monthly rainfall outlook to harvest date

CRITICAL RULE — SPRAY IS ALWAYS WEATHER:
फवारणी/spray/drone spray → ALWAYS WEATHER even if crop mentioned

MARATHI: पाऊस, हवामान, फवारणी, सिंचन, दुष्काळ, कीड, रोग, वारा, उष्णता
HINDI: बारिश, मौसम, सिंचाई, कीट, रोग, सूखा, हवा
ENGLISH: rain, weather, spray, irrigation, pest, disease, drought, wind, harvest

EXAMPLES:
"आज फवारणी करू का सोयाबीनवर?" → WEATHER (spray = weather always)
"paus yeil ka pudhe 7 diwas?" → WEATHER
"majhya cotton la rog aahe" → WEATHER (disease risk)
"drone udvu ka aaj?" → WEATHER (drone = spray safety)
"soybean la heat stress ahe ka?" → WEATHER
"irrigation karu ka?" → WEATHER
"September madhe paus kaasa rahil?" → WEATHER (subseasonal)
""",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "crop": types.Schema(type=types.Type.STRING,
                description="Crop in English lowercase. Can be null for pure weather queries."),
            "forecast_days": types.Schema(type=types.Type.INTEGER,
                description="Days of forecast 1-16, default 7"),
            "sowing_date": types.Schema(type=types.Type.STRING,
                description="Sowing date YYYY-MM-DD if mentioned"),
            "harvest_date": types.Schema(type=types.Type.STRING,
                description="Harvest date YYYY-MM-DD — triggers subseasonal + seasonal horizons"),
        },
        required=[]
    )
)

TOOLS = [types.Tool(function_declarations=[MANDI_TOOL, WEATHER_TOOL])]


async def _route_to_endpoint(extraction: dict, message: str) -> dict:
    client = _get_client()

    context = f"""Farmer message: "{message}"

Extracted:
- Crop: {extraction.get('crop', 'not mentioned')}
- Quantity: {extraction.get('qty', 'not mentioned')} {extraction.get('qty_unit', '')}
- Variety: {extraction.get('variety', 'not mentioned')}
- Time horizon: {extraction.get('time_horizon', 'now')}
- Forecast days: {extraction.get('forecast_days', 7)}
- Sowing date: {extraction.get('sowing_date', 'not mentioned')}
- Harvest date: {extraction.get('harvest_date', 'not mentioned')}
- Intent: {extraction.get('raw_intent', 'unknown')}

Call the correct tool. Remember: spray/फवारणी/drone = WEATHER always."""

    try:
        # Fix 3: native async
        # NOTE: response_mime_type NOT used — incompatible with function calling
        response = await client.aio.models.generate_content(
            model=MODEL_ROUTING,
            contents=context,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY"
                    )
                ),
                temperature=0.0,
            )
        )

        function_call = None
        for part in response.candidates[0].content.parts:
          if part.function_call:  # truthiness check, not existence check
           function_call = part.function_call
           break

        if not function_call:
            logger.error("[Orchestrator] No function call returned despite mode=ANY")
            return {"service_type": None, "params": {}, "confidence": "low"}

        fn_name = function_call.name
        fn_args = dict(function_call.args)
        service_type = "mandi" if fn_name == "get_mandi_prices" else "weather"

        return {"service_type": service_type, "params": fn_args, "confidence": "high"}

    except Exception as e:
        logger.error(f"[Orchestrator] Routing failed: {e}")
        return {"service_type": None, "params": {}, "confidence": "low"}


# ─── Clarification & Response Messages ───────────────────────────────────────

CLARIFICATION_MESSAGES = {
    "service": {
        "mr": "🌾 नमस्कार! आम्ही दोन सेवा देतो:\n\n1️⃣ *मंडी भाव* — कुठे विकायचे, किती नफा\n2️⃣ *हवामान* — पाऊस, फवारणी, सिंचन\n\nतुम्हाला काय हवे आहे?",
        "hi": "🌾 नमस्ते! हम दो सेवाएं देते हैं:\n\n1️⃣ *मंडी भाव* — कहाँ बेचें, कितना मुनाफा\n2️⃣ *मौसम* — बारिश, छिड़काव, सिंचाई\n\nआपको क्या चाहिए?",
        "en": "🌾 Hello! We offer two services:\n\n1️⃣ *Mandi Prices* — where to sell, profit\n2️⃣ *Weather* — rain, spray safety, irrigation\n\nWhat would you like?",
    },
    "crop": {
        "mr": "कोणत्या पिकाची माहिती हवी आहे?\n\nउदाहरण: सोयाबीन, कापूस, कांदा, तूर, गहू",
        "hi": "किस फसल की जानकारी चाहिए?\n\nउदाहरण: सोयाबीन, कपास, प्याज, तुअर, गेहूँ",
        "en": "Which crop?\n\nExample: soybean, cotton, onion, tur, wheat",
    },
    "qty": {
        "mr": "किती क्विंटल माल विकायचा?\n\nउदाहरण: 100 क्विंटल, 50 पोते",
        "hi": "कितना माल बेचना है?\n\nउदाहरण: 100 क्विंटल, 50 बोरी",
        "en": "How much quantity?\n\nExample: 100 quintals, 50 bags",
    }
}

NOT_HANDLED_MESSAGES = {
    "mr": "माफ करा, हे आम्ही करत नाही.\n{redirect}\n\nआम्ही फक्त *मंडी भाव* आणि *हवामान* माहिती देतो.",
    "hi": "माफ करें, यह हम नहीं करते।\n{redirect}\n\nहम सिर्फ *मंडी भाव* और *मौसम* जानकारी देते हैं।",
    "en": "Sorry, we don't handle this.\n{redirect}\n\nWe only provide *mandi prices* and *weather* information.",
}

NOT_AGRI_MESSAGES = {
    "mr": "माफ करा, आम्ही फक्त शेती विषयक मदत करतो — मंडी भाव आणि हवामान माहिती.",
    "hi": "माफ करें, हम सिर्फ खेती से जुड़ी मदद करते हैं।",
    "en": "Sorry, we only help with farming topics — mandi prices and weather.",
}


# ─── Main Orchestrator ────────────────────────────────────────────────────────

async def orchestrate(
    message: str,
    session_store,
    session_id: str,
) -> dict:
    """
    Main entry point. Called by main.py on every farmer WhatsApp message.

    Returns:
        {
            "status": "routed"/"needs_clarification"/"not_handled"/
                      "not_agri"/"coming_soon"/"error",
            "service_type": "mandi"/"weather"/None,
            "reply_message": str,
            "session_updated": bool,
            "detected_language": str,
            "crop": str or None,
            "qty": str or None,
        }
    """
    logger.info(f"[Orchestrator] Processing: '{message[:60]}'")

    # ── Stage 1: Preflight ──────────────────────────────────────────────────
    preflight = await _preflight(message)
    logger.info(f"[Orchestrator] Preflight: {preflight}")

    lang = "mr"  # Default Marathi until extraction detects language

    if not preflight.get("is_agri"):
        return {
            "status": "not_agri",
            "service_type": None,
            "reply_message": NOT_AGRI_MESSAGES["mr"],
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
        }

    if not preflight.get("is_handled"):
        redirect_key = preflight.get("redirect_key", "general")

        # Fix 1: Coming soon intercept before hard rejection
        if redirect_key in COMING_SOON_KEYS:
            reply = COMING_SOON_MESSAGES[redirect_key].get(lang,
                    COMING_SOON_MESSAGES[redirect_key]["mr"])
            return {
                "status": "coming_soon",
                "service_type": None,
                "reply_message": reply,
                "session_updated": False,
                "detected_language": lang,
                "crop": None, "qty": None,
            }

        # Standard rejection with helpful redirect
        redirect_text = THINGS_WE_DONT_HANDLE.get(
            redirect_key, THINGS_WE_DONT_HANDLE["general"]
        )
        reply = NOT_HANDLED_MESSAGES["mr"].format(redirect=redirect_text)
        return {
            "status": "not_handled",
            "service_type": None,
            "reply_message": reply,
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
        }

    # ── Stage 2: Extraction ─────────────────────────────────────────────────
    extraction = await _extract_intent(message)
    lang = extraction.get("language", "mr")
    logger.info(f"[Orchestrator] Extraction: crop={extraction.get('crop')}, "
                f"qty={extraction.get('qty')}, lang={lang}, "
                f"intent={extraction.get('raw_intent')}")

    if extraction.get("needs_clarification"):
        aspect = extraction.get("clarification_aspect", "service")
        clarification_msgs = CLARIFICATION_MESSAGES.get(
            aspect, CLARIFICATION_MESSAGES["service"]
        )
        reply = clarification_msgs.get(lang, clarification_msgs["mr"])
        return {
            "status": "needs_clarification",
            "service_type": None,
            "reply_message": reply,
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
        }

    # ── Stage 3: Routing ────────────────────────────────────────────────────
    routing = await _route_to_endpoint(extraction, message)
    service_type = routing.get("service_type")
    logger.info(f"[Orchestrator] Routing → {service_type} "
                f"(confidence: {routing.get('confidence')})")

    if not service_type:
        reply = CLARIFICATION_MESSAGES["service"].get(
            lang, CLARIFICATION_MESSAGES["service"]["mr"]
        )
        return {
            "status": "needs_clarification",
            "service_type": None,
            "reply_message": reply,
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
        }

    # ── Save to session ─────────────────────────────────────────────────────
    session_store.update_session_data(
        session_id,
        service_type=service_type,
        crop=extraction.get("crop"),
        qty=extraction.get("qty"),
        variety=extraction.get("variety"),
        radius_km=extraction.get("radius_km") or 100,
        time_horizon=extraction.get("time_horizon", "now"),
        forecast_days=extraction.get("forecast_days", 7),
        sowing_date=extraction.get("sowing_date"),
        harvest_date=extraction.get("harvest_date"),
        language=lang,
        raw_intent=extraction.get("raw_intent", ""),
    )

    # ── Acknowledgment message ──────────────────────────────────────────────
    crop = extraction.get("crop", "")

    if service_type == "mandi":
        ack = {
            "mr": f"✅ *{crop.title() if crop else 'पीक'} मंडी भाव*\n\n📍 आता तुमचे स्थान शेअर करा.",
            "hi": f"✅ *{crop.title() if crop else 'फसल'} मंडी भाव*\n\n📍 अब अपना स्थान शेयर करें।",
            "en": f"✅ *{crop.title() if crop else 'Crop'} Mandi Prices*\n\n📍 Please share your location.",
        }
    else:
        ack = {
            "mr": f"✅ *{crop.title() + ' ' if crop else ''}हवामान माहिती*\n\n📍 आता तुमचे स्थान शेअर करा.",
            "hi": f"✅ *{crop.title() + ' ' if crop else ''}मौसम जानकारी*\n\n📍 अब अपना स्थान शेयर करें।",
            "en": f"✅ *{crop.title() + ' ' if crop else ''}Weather Info*\n\n📍 Please share your location.",
        }

    return {
        "status": "routed",
        "service_type": service_type,
        "reply_message": ack.get(lang, ack["mr"]),
        "session_updated": True,
        "detected_language": lang,
        "crop": crop,
        "qty": extraction.get("qty"),
    }