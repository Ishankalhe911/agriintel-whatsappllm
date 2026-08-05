import os
import json
from marathi_formatter import format_response_for_whatsapp

# 1. Ensure your GEMINI_API_KEY is available
# os.environ["GEMINI_API_KEY"] = "your_gemini_api_key_here"

# 2. Raw JSON string from your Mandi endpoint
raw_json_string = """
{
  "error": false,
  "mode": "price_only",
  "crop": "nimbu",
  "variety_requested": null,
  "qty_quintals": null,
  "lat": 18.4,
  "lon": 76.56,
  "radius_km": 60,
  "time_horizon": "now",
  "is_llm_estimate": true,
  "data_source": "gemini_estimate",
  "method_note": null,
  "nearest_mandi": {
    "market": "latur",
    "is_local_baseline": true,
    "is_within_requested_radius": true,
    "exact_scraped_data": {
      "modal_price_per_quintal": 5800,
      "variety": "Kagzi Nimbu",
      "data_source": "msamb_live",
      "is_estimated": false
    },
    "driving_distance": {
      "value_km": 1.5,
      "source": "osrm",
      "is_estimated": false
    }
  },
  "top_mandis": [
    {
      "market": "latur",
      "is_local_baseline": true,
      "is_within_requested_radius": true,
      "exact_scraped_data": {
        "modal_price_per_quintal": 5800,
        "variety": "Kagzi Nimbu",
        "data_source": "msamb_live",
        "is_estimated": false
      },
      "driving_distance": {
        "value_km": 1.5,
        "source": "osrm",
        "is_estimated": false
      }
    },
    {
      "market": "ausa",
      "is_local_baseline": false,
      "is_within_requested_radius": true,
      "exact_scraped_data": {
        "modal_price_per_quintal": 5600,
        "variety": "Kagzi Nimbu",
        "data_source": "msamb_live",
        "is_estimated": false
      },
      "driving_distance": {
        "value_km": 20.2,
        "source": "osrm",
        "is_estimated": false
      }
    },
    {
      "market": "chakur",
      "is_local_baseline": false,
      "is_within_requested_radius": true,
      "exact_scraped_data": {
        "modal_price_per_quintal": 5700,
        "variety": "Kagzi Nimbu",
        "data_source": "msamb_live",
        "is_estimated": false
      },
      "driving_distance": {
        "value_km": 39.4,
        "source": "osrm",
        "is_estimated": false
      }
    }
  ],
  "disclaimer": "Prices are live government APMC modal rates. Transport, driver allowances, and APMC deductions are estimates based on standard regional averages.",
  "agent_execution_rules": {
    "presentation_rule": "First, state the price at the nearest_mandi to establish trust and a local baseline.",
    "pre_dispatch_checklist_to_show_user": [
      "Call your transport driver now to lock in the exact freight rate.",
      "Ensure your crop moisture meets FAQ (Fair Average Quality) standards to get the optimistic price.",
      "Call a local contact at the target APMC to ensure the market is open tomorrow and not on strike."
    ],
    "variety_warning_to_show_user": "Warning: Price based on the highest-priced variety available at the market today (no specific variety was specified)."
  }
}
"""

# Convert JSON string safely to a Python dictionary
mandi_payload = json.loads(raw_json_string)

# Simulate call
SERVICE_TYPE = "mandi-optimize"
USER_QUERY = "लिंबाचा आजचा बाजारभाव काय चालू आहे लातूर भागात?"

print("--- RUNNING MARATHI FORMATTER ---")
formatted_output = format_response_for_whatsapp(SERVICE_TYPE, mandi_payload, original_user_text=USER_QUERY)
print("\n" + formatted_output)