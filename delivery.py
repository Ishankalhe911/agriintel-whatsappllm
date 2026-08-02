"""
delivery.py
───────────
Called by main.py after Razorpay payment.captured webhook fires.
Reads session from Redis, routes to correct x402_client function.

NO AI here — pure deterministic routing.
All type casting done here to shield x402 endpoints from AI anomalies.

Validation rules per service:
    weather     → lat + lon required. crop optional. harvest_date optional.
    mandi       → lat + lon required. crop optional (None = discovery mode).
    fertilizer  → crop required. NO lat/lon needed or sent.
"""

import logging
from x402_client import call_weather_risk, call_mandi_optimize, call_fertilizer

logger = logging.getLogger(__name__)


async def deliver(session: dict) -> dict:
    """
    Main entry point. Called by Razorpay webhook handler in main.py.

    Session keys (set by session_store.py + orchestrator):
        service_type, crop, qty, lat, lon,
        radius_km, time_horizon, variety,
        forecast_days, sowing_date, harvest_date,
        pest, symptom, category_intent,
        language
    """
    service_type = session.get("service_type")
    lat          = session.get("lat")
    lon          = session.get("lon")
    crop         = session.get("crop")

    # ── Universal check — service_type always required ─────────────────────
    if not service_type:
        logger.error(f"[Delivery] No service_type in session: {session}")
        return {
            "error": True,
            "error_type": "SESSION_INCOMPLETE",
            "error_reason": "Missing service_type.",
        }

    # ── Per-service field validation ────────────────────────────────────────
    if service_type in ("weather", "mandi"):
        # Both weather and mandi need lat/lon
        if lat is None or lon is None:
            logger.error(f"[Delivery] {service_type} requires lat/lon, got lat={lat} lon={lon}")
            return {
                "error": True,
                "error_type": "SESSION_INCOMPLETE",
                "error_reason": f"'{service_type}' service requires lat and lon.",
            }

    elif service_type == "fertilizer":
        # Fertilizer needs crop but NOT lat/lon
        if not crop:
            logger.error(f"[Delivery] fertilizer requires crop, got None")
            return {
                "error": True,
                "error_type": "SESSION_INCOMPLETE",
                "error_reason": "'fertilizer' service requires a crop name.",
            }

    else:
        logger.error(f"[Delivery] Unknown service_type: '{service_type}'")
        return {
            "error": True,
            "error_type": "UNKNOWN_SERVICE",
            "error_reason": f"No handler registered for service_type: '{service_type}'",
        }

    logger.info(f"[Delivery] Routing '{service_type}' | crop={crop} | lat={lat}, lon={lon}")

    # ── Route: weather ──────────────────────────────────────────────────────
    if service_type == "weather":
        # forecast_days: safe int cast, cap at 16
        try:
            days = min(int(session.get("forecast_days", 16)), 16)
        except (ValueError, TypeError):
            days = 16

        return await call_weather_risk(
            lat=lat,
            lon=lon,
            crop=crop,                              # optional — None sends no crop key
            forecast_days=days,
            sowing_date=session.get("sowing_date"),
            harvest_date=session.get("harvest_date"),
        )

    # ── Route: mandi ────────────────────────────────────────────────────────
    elif service_type == "mandi":
        # qty_quintals: safe float cast — shields against Gemini string/list hallucination
        raw_qty = session.get("qty")
        try:
            qty_float = float(raw_qty) if raw_qty is not None else None
        except (ValueError, TypeError):
            logger.warning(f"[Delivery] Could not cast qty '{raw_qty}' to float — passing None")
            qty_float = None

        # radius_km: safe int cast, cap at 150
        try:
            radius = min(int(session.get("radius_km", 100)), 150)
        except (ValueError, TypeError):
            radius = 100

        return await call_mandi_optimize(
            lat=lat,
            lon=lon,
            crop=crop,                              # optional — None = discovery mode
            variety=session.get("variety"),
            qty_quintals=qty_float,
            time_horizon=session.get("time_horizon", "now"),
            radius_km=radius,
        )

    # ── Route: fertilizer ───────────────────────────────────────────────────
    elif service_type == "fertilizer":
        # pest: normalize to list — could be string, list, or None from session
        raw_pest = session.get("pest")
        if isinstance(raw_pest, str) and raw_pest.strip():
            pest_list = [raw_pest.strip()]
        elif isinstance(raw_pest, list) and raw_pest:
            pest_list = raw_pest
        else:
            pest_list = None

        return await call_fertilizer(
            crop=crop,
            pest=pest_list,
            symptom=session.get("symptom"),
            category_intent=session.get("category_intent"),
            missing_info=bool(session.get("missing_info", False)),
        )