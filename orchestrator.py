"""
orchestrator.py
───────────────
Three-stage pipeline for farmer message understanding.

Stage 1 — PREFLIGHT (gemini-3.1-flash-lite)
    Gate: is it agri? is it a service we handle? or coming soon?
Stage 2 — EXTRACTION (gemini-3.1-flash-lite)
    Extract: crop, qty, pest, symptom, location hints, language
Stage 3 — ROUTING (gemini-3.1-flash-lite, function calling)
    Route: which endpoint — mandi / weather / fertilizer

Fixes applied:
  ✅ Fix 1: fertilizer removed from coming_soon — it is LIVE
  ✅ Fix 2: FERTILIZER_TOOL added to Stage 3 TOOLS
  ✅ Fix 3: Preflight prompt updated — fertilizer is handled service
  ✅ Fix 4: Extraction schema adds pest/symptom/category_intent fields
  ✅ Fix 5: Session save includes pest/symptom/category_intent
  ✅ Fix 6: Routing else→ explicit if/elif — no silent wrong routing
  ✅ Fix 7: Fertilizer ack message — no location request (correct)
  ✅ Fix 8: All user-facing strings updated to mention 3 services
  ✅ Fix 9: gemini-3.1-flash-lite across all 3 stages
  ✅ Fix 10: response_mime_type stages 1+2, omitted stage 3 (function calling)
  ✅ Fix 11: client.aio.models.generate_content (native async, no threads)
  ✅ Fix 12: Random key rotation across all 5 Gemini keys
  ✅ Fix 13 (NEW): Fertilizer route now confirms crop+pest BEFORE payment.
     If pest/symptom missing, farmer is asked to either type the pest OR
     explicitly say "no" to get PGR/growth-booster advice instead. No more
     silent PGR fallback, no more lost pest data across messages.
"""

import json
import logging
import os
import random
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── API Keys — rotated randomly to avoid RPM limits ─────────────────────────

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]


def _get_client() -> genai.Client:
    """
    Random key selection on every call = true load balancing.
    Prevents any single key hitting RPM limits under concurrent load.
    """
    keys = [k for k in GEMINI_KEYS if k]
    if not keys:
        raise ValueError("No GEMINI_API_KEY_* set. Need at least GEMINI_API_KEY_1.")
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

# ─── Redirects ────────────────────────────────────────────────────────────────
# Fix 8: updated to mention 3 services in general redirect

THINGS_WE_DONT_HANDLE = {
    "loan":      "कर्जासाठी PM Kisan helpline: 155261 वर कॉल करा",
    "seeds":     "बियाण्यांसाठी तुमच्या जवळच्या कृषी केंद्रात जा",
    "insurance": "पीक विम्यासाठी PMFBY helpline: 1800-180-1551",
    "scheme":    "सरकारी योजनांसाठी: agri.maharashtra.gov.in",
    "pest_id":   "किडीची ओळख करण्यासाठी: KVK helpline 1800-180-1551",
    "general":   "माफ करा, हे आम्ही सध्या हाताळत नाही. आम्ही मंडी भाव, हवामान आणि पीक संरक्षण माहिती देतो.",
}

# Fix 1: fertilizer REMOVED — it is a live service now
# Only top_crops remains as coming soon
COMING_SOON_MESSAGES = {
    "top_crops": {
        "mr": "📊 'कोणते पीक घ्यावे' ही सेवा लवकरच सुरू होत आहे!\nसध्या आम्ही *मंडी भाव*, *हवामान* आणि *पीक संरक्षण* माहिती देतो.",
        "hi": "📊 'कौन सी फसल उगाएं' सेवा जल्द शुरू हो रही है!\nअभी हम *मंडी भाव*, *मौसम* और *फसल सुरक्षा* जानकारी देते हैं।",
        "en": "📊 'Top 3 Crops to Grow' service is coming soon!\nCurrently we provide *mandi prices*, *weather* and *crop protection* advice.",
    }
}

# Auto-updates since it's derived from the dict above
COMING_SOON_KEYS = set(COMING_SOON_MESSAGES.keys())


# ─── Stage 1: Preflight ───────────────────────────────────────────────────────

async def _preflight(message: str) -> dict:
    """
    Cheapest gate. Answers three questions:
      1. Is this agri-related at all?
      2. Do we handle this service?
      3. If not handled, which redirect key?
    Returns early with helpful redirect to avoid paying for extraction+routing.
    """
    client = _get_client()

    # Fix 3: fertilizer is now a HANDLED service in the prompt
    # Fix 5: preflight prompt examples updated
    prompt = f"""You are a preflight filter for an Indian agriculture WhatsApp bot.

The bot handles THREE services:
1. Mandi prices — where to sell crop, APMC rates, market prices, profit, transport, logistics
2. Weather risk — rain forecast, spray safety, irrigation, crop stress, pest risk windows, drone safety
3. Crop protection — pest identification, disease diagnosis, chemical/pesticide recommendations,
   fertilizer advice, dosage, waiting periods, brand names (खत, कीटकनाशक, बुरशीनाशक, तणनाशक)

Coming soon (not yet available):
- Top 3 crops to grow / कोणते पीक घ्यावे → redirect_key: "top_crops"

Does NOT handle at all:
- Crop loans / bank / karz → redirect_key: "loan"
- Seeds / beej / biyane → redirect_key: "seeds"
- Insurance / vima → redirect_key: "insurance"
- Government schemes / yojana → redirect_key: "scheme"
- Anything non-agricultural → redirect_key: "general"

Farmer message: "{message}"

Return JSON only:
{{
  "is_agri": true/false,
  "is_handled": true/false,
  "redirect_key": null or "loan"/"seeds"/"insurance"/"scheme"/"general"/"top_crops",
  "reason": "one short English sentence"
}}

Examples:
"soybean bhav kiti" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"mandi price query"}}
"karz hava" → {{"is_agri":true,"is_handled":false,"redirect_key":"loan","reason":"loan request"}}
"konthe pik ghyave" → {{"is_agri":true,"is_handled":false,"redirect_key":"top_crops","reason":"crop selection coming soon"}}
"paus yeil ka" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"weather query"}}
"soybean la stem borer zala" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"pest advisory query"}}
"khad kuthle vapravu" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"crop protection query"}}
"cotton la rog aahe, chemical sanga" → {{"is_agri":true,"is_handled":true,"redirect_key":null,"reason":"disease/chemical query"}}
"cricket score" → {{"is_agri":false,"is_handled":false,"redirect_key":"general","reason":"not agriculture"}}"""

    try:
        response = await client.aio.models.generate_content(
            model=MODEL_PREFLIGHT,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"[Orchestrator] Preflight failed: {e}")
        # Fail open — assume handled so farmer gets a response
        return {
            "is_agri": True, "is_handled": True,
            "redirect_key": None, "reason": "preflight_error_assume_handled",
        }


# ─── Stage 2: Extraction ──────────────────────────────────────────────────────

async def _extract_intent(message: str) -> dict:
    """
    Extracts ALL structured fields from the farmer message.
    Fix 4: Added pest, symptom, category_intent for fertilizer endpoint.
    """
    client = _get_client()

    prompt = f"""You are an expert agricultural assistant fluent in Marathi, Hindi, and English.
Extract structured information from this farmer message.
Primary language is Marathi. Hindi and English also supported.

Farmer message: "{message}"

Supported crops (normalize to English name):
{SUPPORTED_CROPS}

Rules:
1. crop: English lowercase only. 
   - ⚠️ CRITICAL: Generic words like "pik", "पीक", "sheti", "शेती", "crop", or "fasal" are NOT crop names. If no specific crop (e.g. soybean, rice, cotton) is named, set crop=null.
2. qty: number only as string. "100 quintal"→qty="100", qty_unit="quintal"
3. variety: Keep in ORIGINAL SCRIPT. शरबती→variety="शरबती". lokwan→variety="lokwan"
4. time_horizon: "now" unless farmer says pudhe/future/N days → "30_days" format
5. language: "mr" for Marathi/Marathi-in-English-script, "hi" for Hindi, "en" for English
6. pest: extract pest/disease names as a list. मावा→["aphid"], stem borer→["stem_borer"], करपा→["blight"], भुरी→["powdery_mildew"]. Multiple pests → list all.
7. symptom: 
   - ⚠️ CRITICAL: Must be a specific PHYSICAL/VISUAL symptom (e.g. "leaves turning yellow", "holes in leaves", "white spots"). 
   - Generic complaint words like "kharab zhala", "खराब झाले", "rog aala", "रोग आला", "nuksan", "problem aahe" are NOT symptoms. Set symptom=null for generic complaints.
8. category_intent: if farmer asks for a specific chemical category. "बुरशीनाशक सांगा"→"fungicide", "तणनाशक"→"herbicide", "खत"→"fertilizer", "growth booster"→"PGR"


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
  "pest": null,
  "symptom": null,
  "category_intent": null,
  "language": "mr",
  "raw_intent": "one line summary in English",

}}

Examples:
"pik kharab zhalay" → crop=null, pest=null, symptom=null, needs_clarification=true, clarification_aspect="crop"
"rice kharab zhala aahe" → crop="rice", pest=null, symptom=null, needs_clarification=false
"bhav kiti aahet" → crop=null, needs_clarification=true, clarification_aspect="crop"
"soya la bhav kiti" → crop="soybean", needs_clarification=false
"paus yeil ka" → crop=null, raw_intent="weather forecast", needs_clarification=false
"cotton la pane pivali padtat" → crop="cotton", symptom="leaves turning yellow", needs_clarification=false
"tomato la blight zala" → crop="tomato", pest=["blight"], needs_clarification=false
"hi" → needs_clarification=true, clarification_aspect="service" """

    try:
        response = await client.aio.models.generate_content(
            model=MODEL_EXTRACTION,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=400,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"[Orchestrator] Extraction failed: {e}")
        return {
            "crop": None, "qty": None, "qty_unit": None,
            "variety": None, "radius_km": None, "time_horizon": "now",
            "harvest_date": None, "sowing_date": None, "forecast_days": 7,
            "pest": None, "symptom": None, "category_intent": None,
            "language": "mr", "raw_intent": "unknown",
            
        }


# ─── Stage 3: Routing (Function Calling) ─────────────────────────────────────
# response_mime_type NOT used — incompatible with function calling
# Native async (client.aio) still applied

MANDI_TOOL = types.FunctionDeclaration(
    name="get_mandi_prices",
    description="""
Use when farmer wants ANY of:
SELLING / PRICE:
- Current APMC mandi modal price (Maharashtra govt MSAMB database, 450+ records daily)
- Which mandi to sell at for maximum profit
- Price comparison between multiple mandis
- Net profit after APMC deductions (cess 1.05% + commission 3-8% + hamali)
- Transport cost, vehicle recommendation (Tata Ace/Bolero/14ft/10-wheeler by qty)
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
        required=["crop"],
    ),
)

WEATHER_TOOL = types.FunctionDeclaration(
    name="get_weather_risk",
    description="""
Use when farmer wants ANY of:
WEATHER / RAIN:
- Will it rain? Rain forecast next N days
- Drought risk, water stress, season-to-date rainfall vs normal

FARMING OPERATIONS (TIMING):
- WHEN to spray chemicals or apply fertilizer (खत/युरिया कधी देऊ?)
- Spray safety today? (Delta-T: 2-8°C optimal, wind <15kmh, rain <2mm)
- Drone spray window safety check
- Irrigation advice based on net water balance (rain - ET0)
- Wind risk days for spray drift

CROP HEALTH:
- Crop stress risk level (LOW/MEDIUM/HIGH)
- Pest and disease risk WINDOWS (RH>85% + temp 25-32°C = fungal risk)
- Heat stress days, heavy rain days, GDD accumulated
- Growth stage based on sowing date

SEASONAL (requires harvest_date):
- ECMWF sub-seasonal weeks 3-4 outlook
- NASA POWER 30yr climatology adjusted by ENSO/IOD phase

CRITICAL RULE — TIMING IS ALWAYS WEATHER:
फवारणी/spray/drone spray/खत देण्याची वेळ → ALWAYS WEATHER even if crop mentioned

DISAMBIGUATION — WEATHER vs CROP PROTECTION:
- "rog aahe, chemical sanga" → CROP PROTECTION (specific chemical needed)
- "rog yeण्याची शक्यता आहे ka" → WEATHER (disease risk window)
- "stem borer zala, kay maru" → CROP PROTECTION (treatment needed)
- "keed yeण्याचा धोका ahe ka" → WEATHER (pest risk window)
- "खत कधी देऊ?" (When to fertilize) → WEATHER (Timing)
- "कोणते खत वापरू?" (Which fertilizer to buy) → CROP PROTECTION (Product)

MARATHI: पाऊस, हवामान, फवारणी, सिंचन, दुष्काळ, वारा, उष्णता, धोका, कधी, वेळ
HINDI: बारिश, मौसम, सिंचाई, सूखा, हवा, कब, समय
ENGLISH: rain, weather, spray, irrigation, drought, wind, forecast, when, timing

EXAMPLES:
"आज फवारणी करू का?" → WEATHER (spray = weather always)
"युरिया कधी टाकू?" → WEATHER (timing of fertilizer = weather)
"paus yeil ka pudhe 7 diwas?" → WEATHER
"drone udvu ka aaj?" → WEATHER (drone = spray safety)
"soybean la heat stress ahe ka?" → WEATHER
"September madhe paus kaasa rahil?" → WEATHER (subseasonal)
""",
# ... keep the parameters exactly the same ...
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
        required=[],
    ),
)

# Fix 2: FERTILIZER_TOOL added
FERTILIZER_TOOL = types.FunctionDeclaration(
    name="get_crop_protection",
    description="""
Use when farmer wants ANY of:
PEST / DISEASE TREATMENT:
- Specific pest identified and needs chemical recommendation
- Disease on crop and needs fungicide/treatment
- Wants to know what to spray for a specific problem
- Wants dosage, waiting period, brand names for a chemical

CHEMICAL / FERTILIZER ADVICE:
- Which insecticide/fungicide/herbicide to use
- Brand names available in Maharashtra market
- Safe dosage per pump or per acre
- Waiting period before harvest (काढणीपूर्वीचा कालावधी)
- Bio-pesticide options
- CIBRC-approved chemicals for a crop-pest combination

SYMPTOM-BASED DIAGNOSIS:
- Farmer describes visible symptoms (yellow leaves, wilting, holes in leaves)
  and needs to know what pest/disease it is AND what to do

CRITICAL RULES:
- NO location/lat/lon needed — never ask for or use location
- crop is REQUIRED — always extract crop before routing here
- Use for TREATMENT queries, not risk window queries
  (risk windows = WEATHER tool)

DISAMBIGUATION — CROP PROTECTION vs WEATHER:
- "stem borer zala, kay maru" → CROP PROTECTION ✅ (treatment)
- "keed yeण्याचा धोका ahe ka" → WEATHER (risk window, not treatment)
- "chemical sanga" → CROP PROTECTION ✅
- "rog aahe" → CROP PROTECTION ✅ (disease present, needs treatment)
- "rog येण्याची शक्यता" → WEATHER (disease risk forecast)
- "कोणते खत टाकू?" (Which fertilizer) → CROP PROTECTION ✅
- "खत कधी देऊ?" (When to fertilize) → WEATHER ❌ (Timing belongs to Weather)

MARATHI: कीड, रोग, बुरशी, मावा, खोडकिडा, करपा, भुरी, तुडतुडे, फुलकिडे,
         कीटकनाशक, बुरशीनाशक, तणनाशक, खत, औषध, फवारणी (for treatment ONLY)
HINDI: कीट, रोग, फफूंद, माहू, कीटनाशक, फफूंदनाशक, दवाई
ENGLISH: pest, disease, fungus, aphid, stem borer, blight, chemical,
         insecticide, fungicide, herbicide, dosage, brand, spray (for treatment ONLY)

EXAMPLES:
"soybean la stem borer zala, kay maru?" → CROP PROTECTION
"tomato la early blight aahe" → CROP PROTECTION
"cotton la mava aahe, konte chemical" → CROP PROTECTION
"pane pivali padtat, kay hote?" → CROP PROTECTION (symptom diagnosis)
"burshinashak sanga soybean sathi" → CROP PROTECTION
"onion la purple blotch, dose kiti?" → CROP PROTECTION
"grape la powdery mildew, Amistar chalel ka?" → CROP PROTECTION
""",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "crop": types.Schema(type=types.Type.STRING,
                description="Crop in English lowercase e.g. soybean, tomato, cotton. REQUIRED."),
            "pest": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="List of pest/disease names e.g. ['stem_borer', 'aphid']. Empty if symptom-only."),
            "symptom": types.Schema(type=types.Type.STRING,
                description="Visible symptom description if exact pest unknown e.g. 'leaves turning yellow'"),
            "category_intent": types.Schema(type=types.Type.STRING,
                description="Specific chemical category if farmer asked: 'fungicide'/'insecticide'/'herbicide'/'PGR'"),
        },
        required=["crop"],
    ),
)

# Fix 2: FERTILIZER_TOOL added to TOOLS list
TOOLS = [types.Tool(function_declarations=[MANDI_TOOL, WEATHER_TOOL, FERTILIZER_TOOL])]


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
- Pest identified: {extraction.get('pest', 'none')}
- Symptom described: {extraction.get('symptom', 'none')}
- Chemical category: {extraction.get('category_intent', 'none')}
- Intent: {extraction.get('raw_intent', 'unknown')}

Call the correct tool. Key rules:
- TIMING of operations (When to spray/fertilize/harvest/irrigate) → WEATHER
- spray/फवारणी safety → WEATHER always
- pest/disease TREATMENT or chemical needed → CROP PROTECTION
- pest/disease RISK WINDOW forecast → WEATHER
- price/sell/profit/mandi → MANDI"""

    try:
        response = await client.aio.models.generate_content(
            model=MODEL_ROUTING,
            contents=context,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                    )
                ),
                temperature=0.0,
            ),
        )

        function_call = None
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_call = part.function_call
                break

        if not function_call:
            logger.error("[Orchestrator] No function call returned despite mode=ANY")
            return {"service_type": None, "params": {}, "confidence": "low"}

        fn_name = function_call.name
        fn_args = dict(function_call.args)

        # Fix 6: explicit mapping — no silent else clause
        if fn_name == "get_mandi_prices":
            service_type = "mandi"
        elif fn_name == "get_crop_protection":
            service_type = "fertilizer"
        elif fn_name == "get_weather_risk":
            service_type = "weather"
        else:
            logger.error(f"[Orchestrator] Unknown function name from Gemini: {fn_name}")
            service_type = None

        return {"service_type": service_type, "params": fn_args, "confidence": "high"}

    except Exception as e:
        logger.error(f"[Orchestrator] Routing failed: {e}")
        return {"service_type": None, "params": {}, "confidence": "low"}


# ─── Clarification & Response Messages ────────────────────────────────────────
# Fix 8: all strings updated to mention 3 services

CLARIFICATION_MESSAGES = {
    "service": {
        "mr": (
            "🌾 नमस्कार! आम्ही तीन सेवा देतो:\n\n"
            "1️⃣ *मंडी भाव* — कुठे विकायचे, किती नफा\n"
            "2️⃣ *हवामान* — पाऊस, फवारणी, सिंचन\n"
            "3️⃣ *पीक संरक्षण* — कीड, रोग, कीटकनाशक सल्ला\n\n"
            "तुम्हाला काय हवे आहे?"
        ),
        "hi": (
            "🌾 नमस्ते! हम तीन सेवाएं देते हैं:\n\n"
            "1️⃣ *मंडी भाव* — कहाँ बेचें, कितना मुनाफा\n"
            "2️⃣ *मौसम* — बारिश, छिड़काव, सिंचाई\n"
            "3️⃣ *फसल सुरक्षा* — कीट, रोग, कीटनाशक सलाह\n\n"
            "आपको क्या चाहिए?"
        ),
        "en": (
            "🌾 Hello! We offer three services:\n\n"
            "1️⃣ *Mandi Prices* — where to sell, profit calculation\n"
            "2️⃣ *Weather* — rain, spray safety, irrigation\n"
            "3️⃣ *Crop Protection* — pest, disease, chemical advice\n\n"
            "What would you like?"
        ),
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
    },
    "pest": {
        "mr": "कोणती कीड किंवा रोग आहे?\n\nउदाहरण: मावा, खोडकिडा, करपा, भुरी\nकिंवा लक्षणे सांगा: 'पाने पिवळी पडत आहेत'",
        "hi": "कौन सा कीट या रोग है?\n\nउदाहरण: माहू, तना छेदक, झुलसा\nया लक्षण बताएं: 'पत्ते पीले हो रहे हैं'",
        "en": "Which pest or disease?\n\nExample: aphid, stem borer, blight, powdery mildew\nOr describe symptoms: 'leaves turning yellow'",
    },
    # ── NEW (Fix 13): shown BEFORE payment when crop is known but pest/symptom is not.

    "pest_confirm": {
        "mr": (
            "✅ *पीक: {crop}*\n\n"
            "नक्की काय दिसत आहे ते सांगा 👇\n\n"
            "🟡 *रंग बदल* — पाने पिवळी, लाल किंवा काळी पडत आहेत का?\n"
            "🕳️ *छिद्रे / अळी* — पाने/फळे खाल्ली आहेत किंवा अळी दिसतेय का?\n"
            "🕸️ *चुरडा-मुरडा* — पाने वाकडी किंवा गोळा झाली आहेत का?\n"
            "🦠 *बुरशी / डाग* — पानांवर पांढरी भुकटी किंवा करडे डाग आहेत का?\n"
            "🥀 *मर / सुकणे* — पूर्ण झाड अचानक सुकत/कोमेजत आहे का?\n\n"
            "वरीलपैकी जे दिसते ते सांगा — अचूक औषध मिळेल! 🌾\n"
            "(किंवा औषध नको असल्यास 'नाही' लिहा — Growth Booster सल्ला मिळेल)"
        ),
        "hi": (
            "✅ *फसल: {crop}*\n\n"
            "खेत में क्या लक्षण दिख रहे हैं? 👇\n\n"
            "🟡 *रंग बदलना* — पत्ते पीले, लाल या काले हो रहे हैं?\n"
            "🕳️ *छेद / सुंडी* — पत्तों/फलों में छेद हैं या इल्ली दिख रही है?\n"
            "🕸️ *पत्ते मुड़ना* — क्या पत्ते सिकुड़ या मुड़ रहे हैं (Leaf curl)?\n"
            "🦠 *फफूंद / धब्बे* — पत्तों पर सफेद पाउडर या धब्बे हैं?\n"
            "🥀 *सूखना (Wilt)* — क्या पूरा पौधा अचानक सूख रहा है?\n\n"
            "इनमें से जो दिख रहा है वो बताएं — सटीक दवा मिलेगी! 🌾\n"
            "(या दवा नहीं चाहिए तो 'नहीं' लिखें — Growth Booster सलाह मिलेगी)"
        ),
        "en": (
            "✅ *Crop: {crop}*\n\n"
            "What exact symptoms are you seeing? 👇\n\n"
            "🟡 *Color Change* — Leaves turning yellow, red, or black?\n"
            "🕳️ *Holes / Worms* — Holes in leaves/fruits, or visible caterpillars?\n"
            "🕸️ *Curling* — Are the leaves wrinkling or curling up?\n"
            "🦠 *Fungus / Spots* — White powder or brown/black spots?\n"
            "🥀 *Wilting* — Is the whole plant suddenly drying up?\n\n"
            "Reply with what you see to get the exact chemical! 🌾\n"
            "(Or reply 'no' to skip and get Growth Booster advice)"
        ),
    },
}

# ── NEW (Fix 13): keywords that mean "no pest, give me PGR/growth booster instead"
SKIP_PEST_KEYWORDS = {
    "nahi", "nahin", "no", "skip", "nako", "nakoy",
    "नाही", "नको", "नहीं", "growth", "booster", "pgr", "vitamin", "vaadh"
}

NOT_HANDLED_MESSAGES = {
    "mr": "माफ करा, हे आम्ही करत नाही.\n{redirect}\n\nआम्ही *मंडी भाव*, *हवामान* आणि *पीक संरक्षण* माहिती देतो.",
    "hi": "माफ करें, यह हम नहीं करते।\n{redirect}\n\nहम *मंडी भाव*, *मौसम* और *फसल सुरक्षा* जानकारी देते हैं।",
    "en": "Sorry, we don't handle this.\n{redirect}\n\nWe provide *mandi prices*, *weather* and *crop protection* advice.",
}

NOT_AGRI_MESSAGES = {
    "mr": "माफ करा, आम्ही फक्त शेती विषयक मदत करतो — मंडी भाव, हवामान आणि पीक संरक्षण माहिती.",
    "hi": "माफ करें, हम सिर्फ खेती से जुड़ी मदद करते हैं — मंडी भाव, मौसम और फसल सुरक्षा।",
    "en": "Sorry, we only help with farming topics — mandi prices, weather and crop protection.",
}


# ─── Main Orchestrator ────────────────────────────────────────────────────────

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
            "service_type": "mandi"/"weather"/"fertilizer"/None,
            "reply_message": str,
            "session_updated": bool,
            "detected_language": str,
            "crop": str or None,
            "qty": str or None,
            "needs_location": bool,   ← NEW: main.py uses this to decide
        }                               whether to send location request button
    """
    logger.info(f"[Orchestrator] Processing: '{message[:60]}'")

    # ── NEW (Fix 13): if we're waiting on a pest confirmation from a
    # previous turn, handle that reply here BEFORE running the normal
    # preflight/extraction/routing pipeline on it. Otherwise a short reply
    # like "nahi" or a bare pest name would get misclassified from scratch.
    prior = session_store.get_session(session_id) or {}

    if prior.get("awaiting") == "pest_confirmation":
        crop = prior.get("crop", "") or ""
        lang = prior.get("language", "mr")
        msg_clean = message.strip().lower()

        is_skip = any(kw in msg_clean for kw in SKIP_PEST_KEYWORDS)

        if is_skip:
            # Explicit "no pest" confirmation → route to PGR, openly
            session_store.update_session_data(
                session_id,
                service_type="fertilizer",
                crop=crop,
                pest=None,
                symptom=None,
                category_intent="PGR",
                language=lang,
                awaiting=None,
            )
            ack = {
                "mr": f"✅ *{crop.title() if crop else 'पीक'} ग्रोथ बूस्टर सल्ला*\n\n💳 पेमेंट करा आणि सल्ला मिळवा.",
                "hi": f"✅ *{crop.title() if crop else 'फसल'} ग्रोथ बूस्टर सलाह*\n\n💳 पेमेंट करें और सलाह पाएं।",
                "en": f"✅ *{crop.title() if crop else 'Crop'} Growth Booster Advice*\n\n💳 Pay to get recommendations.",
            }
            return {
                "status": "routed", "service_type": "fertilizer",
                "reply_message": ack.get(lang, ack["mr"]),
                "session_updated": True, "detected_language": lang,
                "crop": crop, "qty": None,
                "needs_location": False, "needs_horizon": False,
            }
        else:
            # Treat their reply as the pest/symptom itself (raw text —
            # fertilizermodule's Layer 2 Gemini mapper will resolve it)
            session_store.update_session_data(
                session_id,
                service_type="fertilizer",
                crop=crop,
                pest=None,
                symptom=message.strip(),
                category_intent=None,
                language=lang,
                awaiting=None,
            )
            ack = {
                "mr": f"✅ *{crop.title() if crop else 'पीक'} संरक्षण सल्ला*\n\n🐛 {message.strip()}\n\n💳 पेमेंट करा आणि CIBRC-approved सल्ला मिळवा.",
                "hi": f"✅ *{crop.title() if crop else 'फसल'} सुरक्षा सलाह*\n\n🐛 {message.strip()}\n\n💳 पेमेंट करें और सलाह पाएं।",
                "en": f"✅ *{crop.title() if crop else 'Crop'} Protection Advice*\n\n🐛 {message.strip()}\n\n💳 Pay to get recommendations.",
            }
            return {
                "status": "routed", "service_type": "fertilizer",
                "reply_message": ack.get(lang, ack["mr"]),
                "session_updated": True, "detected_language": lang,
                "crop": crop, "qty": None,
                "needs_location": False, "needs_horizon": False,
            }

    # ── Stage 1: Preflight ──────────────────────────────────────────────────
    lang = "mr"  # Default until extraction detects language
    # ── NEW: Horizon bypass (If we are waiting for the 15/30/60 days button) ──
    if prior.get("awaiting") == "horizon":
        logger.info(f"[Orchestrator] Bypassing extraction/routing (awaiting horizon)")
        # Fall through to the bottom. The routing bypass (Fix B) will catch it 
        # and lock the service_type="weather"
        pass

    # ── CHANGE 1: THE BYPASS: Skip Preflight if we are waiting for a crop or service ──
    if prior.get("awaiting") in ["crop", "service", "horizon"]:
        logger.info(f"[Orchestrator] Bypassing Preflight (User is answering '{prior.get('awaiting')}' prompt)")
        preflight = {"is_agri": True, "is_handled": True}
        lang = prior.get("language", "mr")
    else:
        preflight = await _preflight(message)
        logger.info(f"[Orchestrator] Preflight: {preflight}")

    if not preflight.get("is_agri"):
        return {
            "status": "not_agri",
            "service_type": None,
            "reply_message": NOT_AGRI_MESSAGES["mr"],
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
            "needs_location": False,
        }

    if not preflight.get("is_handled"):
        redirect_key = preflight.get("redirect_key", "general")

        if redirect_key in COMING_SOON_KEYS:
            reply = COMING_SOON_MESSAGES[redirect_key].get(
                lang, COMING_SOON_MESSAGES[redirect_key]["mr"]
            )
            return {
                "status": "coming_soon",
                "service_type": None,
                "reply_message": reply,
                "session_updated": False,
                "detected_language": lang,
                "crop": None, "qty": None,
                "needs_location": False,
            }

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
            "needs_location": False,
        }

    # ── Stage 2: Extraction ─────────────────────────────────────────────────
    extraction = await _extract_intent(message)
    
    # 🚀 MULTI-CROP UX FIX: Protect the database, but warn the farmer!
    raw_crop = extraction.get("crop")
    multi_crop_warning = {"mr": "", "hi": "", "en": ""}
    
    if isinstance(raw_crop, list):
        extraction["crop"] = raw_crop[0] if raw_crop else None
        if len(raw_crop) > 1:
            multi_crop_warning = {
                "mr": "\n💡 *(टीप: एका वेळी एकाच पिकाची माहिती मिळते. दुसऱ्या पिकासाठी नवीन मेसेज पाठवा.)*",
                "hi": "\n💡 *(नोट: एक बार में एक ही फसल की जानकारी मिलती है। दूसरी फसल के लिए नया मेसेज भेजें।)*",
                "en": "\n💡 *(Note: We process one crop at a time. Please send a new message for the second crop.)*"
            }

    lang = extraction.get("language", "mr")
    logger.info(
        f"[Orchestrator] Extraction: crop={extraction.get('crop')}, "
        f"qty={extraction.get('qty')}, pest={extraction.get('pest')}, "
        f"lang={lang}, intent={extraction.get('raw_intent')}"
    )

    # ── NEW (Fix 13): merge in whatever the prior turn already knew...

    # ── NEW (Fix 13): merge in whatever the prior turn already knew, so a
    # crop given two messages ago (or a pest given last message) isn't lost
    # just because this message doesn't repeat it.
    if not extraction.get("crop") and prior.get("crop"):
        extraction["crop"] = prior.get("crop")
    if not extraction.get("pest") and prior.get("pest"):
        extraction["pest"] = prior.get("pest")
    if not extraction.get("symptom") and prior.get("symptom"):
        extraction["symptom"] = prior.get("symptom")


    # ── Stage 3: Routing ────────────────────────────────────────────────────
    # 🚀 MEMORY FIX: Bypass LLM routing if we are just answering a crop or horizon question
    if prior.get("awaiting") in ["crop", "horizon"] and prior.get("service_type"):
        service_type = prior.get("service_type")
        logger.info(f"[Orchestrator] Bypassing Routing, reusing locked service_type: {service_type}")
    else:
        routing = await _route_to_endpoint(extraction, message)
        service_type = routing.get("service_type")
        logger.info(
            f"[Orchestrator] Routing → {service_type} "
            f"(confidence: {routing.get('confidence')})"
        )
    if not service_type:
        reply = CLARIFICATION_MESSAGES["service"].get(
            lang, CLARIFICATION_MESSAGES["service"]["mr"]
        )
        session_store.update_session_data(session_id, awaiting="service") # <--- CHANGE 3: TAG MEMORY
        return {
            "status": "needs_clarification",
            "service_type": None,
            "reply_message": reply,
            "session_updated": False,
            "detected_language": lang,
            "crop": None, "qty": None,
            "needs_location": False,
        }

    crop = extraction.get("crop", "") or ""

    # ── FIX 15: HARDCODED CROP CHECK BEFORE PAYMENT ──
    if service_type in ["mandi", "fertilizer"] and not crop:
        session_store.update_session_data(
            session_id,
            service_type=service_type,
            qty=extraction.get("qty"),             # <-- ADDED THIS (So we don't forget quantity!)
            variety=extraction.get("variety"),     # <-- ADDED THIS
            pest=extraction.get("pest"),
            symptom=extraction.get("symptom"),
            category_intent=extraction.get("category_intent"),
            language=lang,
            awaiting="crop",  # <--- CHANGE 4: TAG MEMORY
        )
        reply = CLARIFICATION_MESSAGES["crop"].get(lang, CLARIFICATION_MESSAGES["crop"]["mr"])
        return {
            "status": "needs_clarification",
            "service_type": service_type,
            "reply_message": reply,
            "session_updated": True,
            "detected_language": lang,
            "crop": None, 
            "qty": extraction.get("qty"),          # <-- ADDED THIS
            "needs_location": False,
            "needs_horizon": False,
        }

    # ── Fertilizer route: confirm crop + pest BEFORE payment (Fix 13) ──────
    # ── Fertilizer route: confirm crop + pest BEFORE payment (Fix 13) ──────
    if service_type == "fertilizer":
        
        raw_pest = extraction.get("pest")
        raw_symptom = extraction.get("symptom")
        cat_intent = (extraction.get("category_intent") or "").upper()

        # It's only a valid request if we have an actual pest/symptom, 
        # OR if they explicitly just want a Growth Booster (PGR)
        pest_mentioned = bool(raw_pest) or bool(raw_symptom) or (cat_intent == "PGR")

        

        if not pest_mentioned:
            session_store.update_session_data(
                session_id,
                service_type="fertilizer",
                crop=crop,
                qty=extraction.get("qty"),         # <-- FIXED: You missed this!
                variety=extraction.get("variety"), # <-- FIXED: You missed this!
                language=lang,
                awaiting="pest_confirmation",
            )
            msg_template = CLARIFICATION_MESSAGES["pest_confirm"].get(
                lang, CLARIFICATION_MESSAGES["pest_confirm"]["mr"]
            )
            reply_msg = msg_template.format(crop=crop.title() if crop else "पीक")
            return {
                "status": "needs_clarification",
                "service_type": "fertilizer",
                "reply_message": reply_msg,
                "session_updated": True,
                "detected_language": lang,
                "crop": crop, 
                "qty": extraction.get("qty"),      # <-- FIXED: You missed this!
                "needs_location": False,
                "needs_horizon": False,            # <-- FIXED: You missed this!
            }

    # ── Save to session ─────────────────────────────────────────────────────
    # Fix 5: pest/symptom/category_intent now saved to session
    session_store.update_session_data(
        session_id,
        service_type=service_type,
        crop=crop,
        qty=extraction.get("qty"),
        variety=extraction.get("variety"),
        radius_km=extraction.get("radius_km") or 100,
        time_horizon=extraction.get("time_horizon", "now"),
        forecast_days=extraction.get("forecast_days", 7),
        sowing_date=extraction.get("sowing_date"),
        harvest_date=extraction.get("harvest_date"),
        pest=extraction.get("pest"),
        symptom=extraction.get("symptom"),
        category_intent=extraction.get("category_intent"),
        language=lang,
        original_message=message,
        raw_intent=extraction.get("raw_intent", ""),
        awaiting=None,  # <--- CHANGE 5: WIPE MEMORY ON SUCCESS
    )

    # ── Final Acks ──
    # Fix 6+7: explicit ack per service, fertilizer gets NO location prompt
    # needs_location tells main.py whether to send the location request button
    if service_type == "mandi":
        needs_horizon = False
        needs_location = True
        ack = {
            "mr": f"✅ *{crop.title() if crop else 'पीक'} मंडी भाव*{multi_crop_warning.get(lang, '')}\n\n📍 आता तुमचे स्थान शेअर करा.",
            "hi": f"✅ *{crop.title() if crop else 'फसल'} मंडी भाव*{multi_crop_warning.get(lang, '')}\n\n📍 अब अपना स्थान शेयर करें।",
            "en": f"✅ *{crop.title() if crop else 'Crop'} Mandi Prices*{multi_crop_warning.get(lang, '')}\n\n📍 Please share your location.",
        }

    elif service_type == "fertilizer":
        # By this point pest_mentioned is guaranteed truthy (handled above)
        needs_location = False
        needs_horizon = False
        
        # ── FIX 16: CLEAN LIST FORMATTING ──
        raw_pest = extraction.get("pest")
        if isinstance(raw_pest, list) and len(raw_pest) > 0:
            pest_display = ", ".join(raw_pest).replace("_", " ").title()
        else:
            pest_display = extraction.get("symptom") or extraction.get("category_intent")

        ack = {
            "mr": (
                f"✅ *{crop.title() if crop else 'पीक'} संरक्षण सल्ला*{multi_crop_warning.get(lang, '')}\n\n"
                f"{'🐛 ' + str(pest_display) + chr(10) + chr(10) if pest_display else ''}"
                "💳 पेमेंट करा आणि CIBRC-approved रासायनिक सल्ला मिळवा."
            ),
            "hi": (
                f"✅ *{crop.title() if crop else 'फसल'} सुरक्षा सलाह*{multi_crop_warning.get(lang, '')}\n\n"
                f"{'🐛 ' + str(pest_display) + chr(10) + chr(10) if pest_display else ''}"
                "💳 पेमेंट करें और CIBRC-approved रासायनिक सलाह पाएं।"
            ),
            "en": (
                f"✅ *{crop.title() if crop else 'Crop'} Protection Advice*{multi_crop_warning.get(lang, '')}\n\n"
                f"{'🐛 ' + str(pest_display) + chr(10) + chr(10) if pest_display else ''}"
                "💳 Pay to get CIBRC-approved chemical recommendations."
            ),
        }

    else:  # weather
        needs_location = True
        # Skip horizon ask if extraction already found harvest_date
        needs_horizon = not bool(extraction.get("harvest_date"))
        
        # 🚀 ADD THIS: Save the state so we know what the next message means!
        if needs_horizon:
            session_store.update_session_data(session_id, awaiting="horizon")
        ack = {
            "mr": (
                f"✅ *{crop.title() + ' ' if crop else ''}हवामान माहिती*{multi_crop_warning.get(lang, '')}\n\n"
                f"तुम्हाला किती दिवसांचा अंदाज हवा आहे? खाली निवडा 👇"
                if needs_horizon else
                f"✅ *{crop.title() + ' ' if crop else ''}हवामान माहिती*{multi_crop_warning.get(lang, '')}\n\n📍 आता तुमचे स्थान शेअर करा."
            ),
            "hi": (
                f"✅ *{crop.title() + ' ' if crop else ''}मौसम जानकारी*{multi_crop_warning.get(lang, '')}\n\n"
                f"कितने दिनों का अनुमान चाहिए? नीचे चुनें 👇"
                if needs_horizon else
                f"✅ *{crop.title() + ' ' if crop else ''}मौसम जानकारी*{multi_crop_warning.get(lang, '')}\n\n📍 अब अपना स्थान शेयर करें।"
            ),
            "en": (
                f"✅ *{crop.title() + ' ' if crop else ''}Weather Info*{multi_crop_warning.get(lang, '')}\n\n"
                f"How many days of forecast do you need? Choose below 👇"
                if needs_horizon else
                f"✅ *{crop.title() + ' ' if crop else ''}Weather Info*{multi_crop_warning.get(lang, '')}\n\n📍 Please share your location."
            ),
        }

    return {
        "status": "routed",
        "service_type": service_type,
        "reply_message": ack.get(lang, ack["mr"]),
        "session_updated": True,
        "detected_language": lang,
        "crop": crop,
        "qty": extraction.get("qty"),
        "needs_location": needs_location,
        "needs_horizon": needs_horizon,
    }