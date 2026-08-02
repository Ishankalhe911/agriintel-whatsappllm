"""
marathi_formatter.py
=====================================
Formatting Layer for AgriIntellect WhatsApp Backend.

Takes raw JSON output from microservices (Weather, Mandi, Crop Protection)
and transforms it into a clear, empathetic, WhatsApp-formatted Marathi narrative
using Gemini 2.5 Flash, matching the farmer's detected dialect.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

# Set up logging
logger = logging.getLogger("agriintellect.formatter")

# Initialize Gemini Client
GEMINI_API_KEY5= os.getenv("GEMINI_API_KEY_5",)
client = genai.Client(api_key=GEMINI_API_KEY5) if GEMINI_API_KEY5 else None

# System prompt enforcing formatting rules & agronomy tone

SYSTEM_FORMATTER_PROMPT = """
You are an expert Maharashtrian agronomist and empathetic WhatsApp assistant for local farmers.
Your job is to take raw JSON data from agricultural endpoints and explain it to a farmer in warm, clear, actionable Marathi.

CRITICAL WHATSAPP FORMATTING RULES (STRICTLY FOLLOW THESE):
1. Output language MUST be Marathi (or Marathi in Devanagari script). Adapt tone to the farmer's context if provided.
2. WHATSAPP MARKDOWN ONLY: 
   - DO NOT use **text** or # headers. WhatsApp will not render them.
   - Use ONLY single asterisks for bolding: *महत्त्वाची माहिती*
   - Use ONLY bullet points (-) or (•) for lists.
3. ANTI-HALLUCINATION: Do NOT invent, guess, or add any agricultural data (prices, dosages, waiting periods) that is not explicitly present in the provided JSON. If a value is null or missing, ignore it.
4. DO NOT print raw technical JSON keys, null fields, or database field names.
5. Include relevant agricultural emojis (🌾, 🌧️, 🐛, 💰, 🚜, 💡, 🧪, ⚠️).
6. Keep paragraphs short and scannable on mobile screens.
7. End with a polite, encouraging sign-off (e.g., "शेतीविषयक अधिक माहितीसाठी कधीही विचारा! 🚜").
"""

def format_weather_response(data: Dict[str, Any], dialect_context: str = "") -> str:
    """Formats raw Weather-Risk JSON into Marathi narrative."""
    prompt = f"""
Farmer's Context/Dialect: {dialect_context or "Standard Marathi"}
Service: Weather & Advisory Risk
Raw JSON Payload:
{json.dumps(data, ensure_ascii=False, indent=2)}

Task: Create a concise, practical weather update for the farmer in Marathi.
Highlight:
- Upcoming rainfall estimates and temperatures.
- Key operational risks (e.g., if spray window is blocked or heavy rain expected).
- Irrigation or drone spray advice if present.
"""
    return _call_gemini_formatter(prompt)

def format_mandi_response(data: Dict[str, Any], dialect_context: str = "") -> str:
    """Formats raw Mandi-Optimization JSON into Marathi narrative."""
    prompt = f"""
Farmer's Context/Dialect: {dialect_context or "Standard Marathi"}
Service: Mandi Price Optimization
Raw JSON Payload:
{json.dumps(data, ensure_ascii=False, indent=2)}

Task: Create a clear mandi price breakdown and selling recommendation in Marathi.
Highlight:
- Best mandis to sell with estimated prices (रु/क्विंटल).
- Distance and net profit estimation after transport.
- Actionable selling recommendation (e.g., sell today vs. hold).
"""
    return _call_gemini_formatter(prompt)

def format_pest_response(data: Dict[str, Any], dialect_context: str = "") -> str:
    """Formats raw Pest Advisory / Fertilizer JSON into Marathi narrative."""
    prompt = f"""
Farmer's Context/Dialect: {dialect_context or "Standard Marathi"}
Service: Crop Protection & Pest Advisory
Raw JSON Payload:
{json.dumps(data, ensure_ascii=False, indent=2)}

Task: Create a step-by-step crop protection guide in Marathi.
Highlight:
- Crop and identified pests/diseases (कीड/रोग).
- Recommended chemical or bio-pesticide combinations (with popular brand names like Saaf, Amistar, etc.).
- Exact dosage (प्रमाण) per 15L pump or acre.
- Waiting period before harvest (काढणीपूर्वीचा कालावधी).
"""
    return _call_gemini_formatter(prompt)

def _call_gemini_formatter(user_prompt: str) -> str:
    """Internal helper to execute Gemini 2.5 Flash formatting call."""
    if not client:
        logger.error("GEMINI_API_KEY missing in marathi_formatter!")
        return "क्षमा करा, तांत्रिक अडचणीमुळे माहिती तयार करता आली नाही. कृपया थोड्या वेळाने प्रयत्न करा."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[SYSTEM_FORMATTER_PROMPT, user_prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,  # Low temperature for factual precision
                max_output_tokens=4096,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error formatting Marathi response with Gemini: {e}")
        return "माहिती मिळवण्यात अडचण येत आहे. कृपया पुन्हा संदेश पाठवा."

def format_response_for_whatsapp(service_type: str, raw_data: Dict[str, Any], original_user_text: str = "") -> str:
    """
    Main entry point called by delivery.py / orchestrator.
    Routes raw JSON to the specific formatter function based on service_type.
    """
    if not raw_data or raw_data.get("status") in ["no_match", "error"]:
        msg = raw_data.get("message", "माहिती उपलब्ध नाही.") if isinstance(raw_data, dict) else "माहिती उपलब्ध नाही."
        return f"⚠️ *माहिती आढळली नाही*\n\n{msg}\n\nकृपया पिक किंवा समस्येचे नाव पुन्हा तपासून पाठवा."

    # Route based on service
    if service_type == "weather-risk":
        return format_weather_response(raw_data, dialect_context=original_user_text)
    elif service_type == "mandi-optimize":
        return format_mandi_response(raw_data, dialect_context=original_user_text)
    elif service_type in ["pest-advisory", "fertilizer-info"]:
        return format_pest_response(raw_data, dialect_context=original_user_text)
    else:
        # Generic fallback formatter
        prompt = f"Convert this agricultural data to friendly Marathi text for WhatsApp: {json.dumps(raw_data)}"
        return _call_gemini_formatter(prompt)


# ── LOCAL TESTING BLOCK ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with dummy Fertilizer JSON
    sample_pest_json = {
        "status": "success",
        "resolved_parameters": {
            "crop": "grapes",
            "crop_display": "द्राक्षे",
            "targets_resolved": ["करपा (AI Mapped)", "भुरी (AI Mapped)"]
        },
        "recommendations": {
            "overlap_best_matches": [
                {
                    "chemical_name": "Azoxystrobin 25% + Boscalid 35% WG",
                    "pests_covered": ["powdery_mildew", "anthracnose"],
                    "dosage": {
                        "formulation_dose": "500 gm/ha",
                        "water_dilution": "1000 L",
                        "waiting_period": "5 days"
                    },
                    "brands": ["Amistar", "Katyayani Azodharma"]
                }
            ]
        }
    }

    print("--- TESTING MARATHI FORMATTER ---")
    formatted_msg = format_response_for_whatsapp("pest-advisory", sample_pest_json, original_user_text="माझ्या द्राक्षावर करपा अन भुरी आलीय काय मारू?")
    print(formatted_msg)