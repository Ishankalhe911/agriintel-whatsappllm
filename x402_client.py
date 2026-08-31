"""
x402_client.py
──────────────
Reusable x402 payment client for the AgriIntel WhatsApp agent.
Wraps all three live endpoints behind a clean async function each.

Endpoints:
    call_weather_risk()   → POST /weather-risk    ($0.015 USDC, 15000 atomic)
    call_mandi_optimize() → POST /mandi-optimize  ($0.06  USDC, 60000 atomic)
    call_fertilizer()     → POST /fertilizer-info ($0.04  USDC, 40000 atomic)

CRITICAL RULES (proven on mainnet — do not change):
    - MnemonicSigner internals are verified working. Do not touch.
    - wrapHttpxWithPayment usage is verified working. Do not touch.
    - Mnemonic MUST come from env var FLOAT_WALLET_MNEMONIC only.
    - Prices are in atomic USDC units (6 decimals). Never pass bare strings.
    - Always call await request.aread() before http.send() for retry safety.
"""

import asyncio
import base64
import logging
import os
from typing import Optional
import json
import httpx
from algosdk import account, encoding, mnemonic
from x402.client import x402Client
from x402.http.clients.httpx import wrapHttpxWithPayment
from x402.mechanisms.avm.exact import ExactAvmScheme

logger = logging.getLogger(__name__)

# ─── Network ──────────────────────────────────────────────────────────────────

AVM_MAINNET = "algorand:wGHE2Pwdvd7S12BL5FaOP20EGYesN73ktiC1qzkkit8="

# ─── Endpoint URLs ────────────────────────────────────────────────────────────

WEATHER_URL    = os.getenv("WEATHER_URL",    "https://agriintellect.site/weather-risk")
MANDI_URL      = os.getenv("MANDI_URL",      "https://agriintellect.site/mandi-optimize")
FERTILIZER_URL = os.getenv("FERTILIZER_URL", "https://agriintellect.site/fertilizer-info")

# ─── Prices (atomic USDC units, 6 decimals) ───────────────────────────────────
# 0.015 USDC = 15000 | 0.06 USDC = 60000 | 0.04 USDC = 40000

WEATHER_PRICE_ATOMIC    = int(os.getenv("WEATHER_PRICE_ATOMIC",    "15000"))
MANDI_PRICE_ATOMIC      = int(os.getenv("MANDI_PRICE_ATOMIC",      "60000"))
FERTILIZER_PRICE_ATOMIC = int(os.getenv("FERTILIZER_PRICE_ATOMIC", "40000"))

# ─── Treasury monitoring ──────────────────────────────────────────────────────

USDC_ASSET_ID  = 31566704  # Mainnet USDC ASA ID
ALGONODE_URL   = "https://mainnet-api.algonode.cloud/v2/accounts"
_ALERTS_SENT   = {"1.0": False, "0.5": False, "0.25": False}


# ─── Signer ───────────────────────────────────────────────────────────────────

class MnemonicSigner:
    """Native Algorand transaction signer. Proven on mainnet — do not modify."""

    def __init__(self, mnemonic_phrase: str):
        self._private_key_b64 = mnemonic.to_private_key(mnemonic_phrase)
        self._address = account.address_from_private_key(self._private_key_b64)

    @property
    def address(self) -> str:
        return self._address

    def sign_transactions(
        self,
        unsigned_txns: list[bytes],
        indexes_to_sign: list[int],
    ) -> list[bytes | None]:
        results = [None] * len(unsigned_txns)
        for i in indexes_to_sign:
            b64_txn = base64.b64encode(unsigned_txns[i]).decode("utf-8")
            txn = encoding.msgpack_decode(b64_txn)
            stxn = txn.sign(self._private_key_b64)
            b64_stxn = encoding.msgpack_encode(stxn)
            results[i] = base64.b64decode(b64_stxn)
        return results


# ─── Client factory ───────────────────────────────────────────────────────────

def _build_client() -> tuple[x402Client, str]:
    """
    Builds x402Client from env mnemonic.
    Returns (client, float_wallet_address).
    Raises ValueError if mnemonic not set.
    """
    phrase = os.getenv("FLOAT_WALLET_MNEMONIC", "").strip()
    if not phrase:
        raise ValueError("FLOAT_WALLET_MNEMONIC env var is not set")

    signer = MnemonicSigner(phrase)
    logger.info(f"[x402] Float wallet: {signer.address[:8]}...")

    client = x402Client()
    client.register(AVM_MAINNET, ExactAvmScheme(signer=signer))
    return client, signer.address


# ─── Treasury balance + alerting ─────────────────────────────────────────────

async def _get_usdc_balance(wallet_address: str) -> float:
    """Query AlgoNode for live USDC balance. Returns -1.0 on failure."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            res = await http.get(f"{ALGONODE_URL}/{wallet_address}")
            if res.status_code == 200:
                for asset in res.json().get("assets", []):
                    if asset.get("asset-id") == USDC_ASSET_ID:
                        return asset.get("amount", 0) / 1_000_000.0
                return 0.0
            logger.error(f"[x402] Balance check HTTP {res.status_code}")
            return -1.0
    except Exception as e:
        logger.error(f"[x402] Balance check failed: {e}")
        return -1.0


async def _check_treasury(wallet_address: str) -> None:
    """Non-blocking background task. Logs warnings at $1.0, $0.5, $0.25 thresholds."""
    balance = await _get_usdc_balance(wallet_address)
    if balance < 0:
        return

    logger.info(f"[x402 Treasury] Float balance: ${balance:.4f} USDC")

    if balance <= 0.25 and not _ALERTS_SENT["0.25"]:
        logger.critical(f"🚨 [TREASURY] Float wallet at ${balance:.4f} USDC — TOP UP NOW")
        _ALERTS_SENT["0.25"] = True
    elif balance <= 0.50 and not _ALERTS_SENT["0.5"]:
        logger.warning(f"⚠️  [TREASURY] Float wallet at ${balance:.4f} USDC — top up soon")
        _ALERTS_SENT["0.5"] = True
    elif balance <= 1.00 and not _ALERTS_SENT["1.0"]:
        logger.info(f"ℹ️  [TREASURY] Float wallet at ${balance:.4f} USDC")
        _ALERTS_SENT["1.0"] = True

    # Reset flags once topped up past $1.50
    if balance > 1.50:
        _ALERTS_SENT["1.0"] = False
        _ALERTS_SENT["0.5"] = False
        _ALERTS_SENT["0.25"] = False


# ─── Shared error response builder ────────────────────────────────────────────

def _err(error_type: str, reason: str, status_code: int = 0) -> dict:
    payload = {"error": True, "error_type": error_type, "error_reason": reason}
    if status_code:
        payload["status_code"] = status_code
    return payload


# ─── call_weather_risk ────────────────────────────────────────────────────────

async def call_weather_risk(
    lat: float,
    lon: float,
    crop: Optional[str] = None,
    sowing_date: Optional[str] = None,
    harvest_date: Optional[str] = None,
    forecast_days: int = 16,
) -> dict:
    """
    Calls POST /weather-risk via x402 ($0.015 USDC).

    Required: lat, lon
    Optional: crop (default "generic"), sowing_date, harvest_date, forecast_days (1-16)

    harvest_date triggers ECMWF sub-seasonal (horizon 2) and NASA POWER
    seasonal (horizon 3). Must be a future date, max 270 days out.
    forecast_days capped at 16 by endpoint — anything above 16 will be rejected.
    """
    # Build payload — only include optional fields if actually provided
    payload: dict = {
        "lat": float(lat),
        "lon": float(lon),
        "forecast_days": min(int(forecast_days), 16),
    }
    if crop:
        payload["crop"] = str(crop).lower()
    if sowing_date:
        payload["sowing_date"] = sowing_date
    if harvest_date:
        payload["harvest_date"] = harvest_date

    logger.info(
        f"[x402] weather-risk → lat={lat}, lon={lon}, crop={crop}, "
        f"forecast_days={payload['forecast_days']}, harvest_date={harvest_date}",
        
    )
    
    try:
        client, float_address = _build_client()

        async with wrapHttpxWithPayment(client) as http:
            request = http.build_request(
                "POST", WEATHER_URL, json=payload, timeout=30.0
            )
            # Buffer for x402 retry logic
            await request.aread()
            response = await http.send(request)

        if response.status_code == 200:
            logger.info("[x402] ✅ weather-risk payment settled")
            asyncio.create_task(_check_treasury(float_address))
            data = response.json()
            # 🚀 PLUGS THE X402_TX_ID HOLE
            x402_tx = None
            payment_header = response.headers.get("payment-response")
            if payment_header:
                try:
                    import json
                    decoded_json = base64.b64decode(payment_header).decode("utf-8")
                    payment_data = json.loads(decoded_json)
                    x402_tx = payment_data.get("transaction")
                except Exception as e:
                    logger.warning(f"[x402] Failed to decode payment-response header: {e}")
            
            data["x402_tx_id"] = x402_tx
            return data

        elif response.status_code == 400:
            logger.warning(f"[x402] weather-risk 400: {response.text}")
            return _err("BAD_REQUEST", response.text, 400)

        elif response.status_code == 503:
            logger.warning("[x402] weather-risk 503: data unavailable")
            return _err("DATA_UNAVAILABLE", "Weather data unavailable. Retry in 60s.", 503)

        else:
            logger.error(f"[x402] weather-risk unexpected {response.status_code}")
            return _err("UNEXPECTED_ERROR", f"HTTP {response.status_code}: {response.text}", response.status_code)

    except ValueError as e:
        logger.error(f"[x402] Config error: {e}")
        return _err("CONFIG_ERROR", str(e))

    except Exception as e:
        logger.error(f"[x402] weather-risk pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))


# ─── call_mandi_optimize ─────────────────────────────────────────────────────

async def call_mandi_optimize(
    lat: float,
    lon: float,
    crop: Optional[str] = None,
    variety: Optional[str] = None,
    qty_quintals: Optional[float] = None,
    time_horizon: str = "now",
    radius_km: int = 100,
) -> dict:
    """
    Calls POST /mandi-optimize via x402 ($0.06 USDC).

    Required: lat, lon
    Optional: crop (None = discovery mode — returns nearest active mandis only),
              variety (Marathi script preferred e.g. शरबती),
              qty_quintals (needed for profit calculation),
              time_horizon ("now" or "<N>_days", max 180),
              radius_km (1-150, default 100)

    Note: crop=None is valid (discovery mode). Endpoint returns nearest mandis
    without profit calc. qty_quintals without crop is ignored by endpoint.
    """
    _VALID_HORIZONS = {"now", "tomorrow", "week"}
    raw_horizon = str(time_horizon).lower().strip() if time_horizon else "now"
    safe_horizon = raw_horizon if raw_horizon in _VALID_HORIZONS else "now"
    payload: dict = {
        "lat": float(lat),
        "lon": float(lon),
        "time_horizon": "now",
        "radius_km": min(int(radius_km), 150),
    }
    if crop:
        payload["crop"] = str(crop).lower()
    if variety:
        payload["variety"] = variety  # keep original script — Marathi/Hindi
    if qty_quintals is not None:
        payload["qty_quintals"] = float(qty_quintals)

    logger.info(
        f"[x402] mandi-optimize → lat={lat}, lon={lon}, crop={crop}, "
        f"qty={qty_quintals}, radius={radius_km}km, horizon={time_horizon}"
    )
    logger.info(f"[x402] mandi payload being sent: {str(payload)}")  # ← ADD THIS LINE

    try:
        client, float_address = _build_client()

        async with wrapHttpxWithPayment(client) as http:
            request = http.build_request(
                "POST", MANDI_URL, json=payload, timeout=240.0
            )
            await request.aread()
            response = await http.send(request)

        if response.status_code == 200:
            logger.info("[x402] ✅ mandi-optimize payment settled")
            asyncio.create_task(_check_treasury(float_address))
            data = response.json()
            data["x402_tx_id"] = response.headers.get("x-transaction-hash") or response.headers.get("x-tx-id")
            return data

        elif response.status_code == 400:
            logger.warning(f"[x402] mandi-optimize 400: {response.text}")
            return _err("BAD_REQUEST", response.text, 400)

        elif response.status_code == 503:
            logger.warning("[x402] mandi-optimize 503: scraper down")
            return _err("DATA_UNAVAILABLE", "MSAMB data unavailable. Retry in 60s.", 503)

        else:
            logger.error(f"[x402] mandi-optimize unexpected {response.status_code}")
            return _err("UNEXPECTED_ERROR", f"HTTP {response.status_code}: {response.text}", response.status_code)

    except ValueError as e:
        logger.error(f"[x402] Config error: {e}")
        return _err("CONFIG_ERROR", str(e))

    except Exception as e:
        logger.error(f"[x402] mandi-optimize pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))


# ─── call_fertilizer ─────────────────────────────────────────────────────────

async def call_fertilizer(
    crop: str,
    pest: Optional[list] = None,
    symptom: Optional[str] = None,
    category_intent: Optional[str] = None,
    missing_info: bool = False,
) -> dict:
    """
    Calls POST /fertilizer-info via x402 ($0.04 USDC).

    Required: crop
    Optional: pest (list of pest names, regional slang ok e.g. ['मावा', 'stem borer']),
              symptom (fuzzy description if exact pest unknown),
              category_intent (filter by type e.g. 'fungicide', 'PGR'),
              missing_info (True if upstream LLM could not determine crop)

    Note: NO lat/lon required or sent for this endpoint.
    pest and symptom are complementary — send pest if known, symptom if not.
    Endpoint handles Marathi pest names via alias lookup.
    """
    payload: dict = {
        "crop": str(crop).lower(),
        "missing_info": bool(missing_info),
    }
    if pest:
        # Normalize: always send as list
        payload["pest"] = pest if isinstance(pest, list) else [pest]
    if symptom:
        payload["symptom"] = str(symptom)
    if category_intent:
        payload["category_intent"] = str(category_intent)

    logger.info(
        f"[x402] fertilizer-info → crop={crop}, pest={pest}, "
        f"symptom={'yes' if symptom else 'none'}, category={category_intent}"
    )

    try:
        client, float_address = _build_client()

        async with wrapHttpxWithPayment(client) as http:
            request = http.build_request(
                "POST", FERTILIZER_URL, json=payload, timeout=480.0
            )
            await request.aread()
            response = await http.send(request)

        if response.status_code == 200:
            logger.info("[x402] ✅ fertilizer-info payment settled")
            asyncio.create_task(_check_treasury(float_address))
            data = response.json()
            data["x402_tx_id"] = response.headers.get("x-transaction-hash") or response.headers.get("x-tx-id")
            return data

        elif response.status_code == 400:
            logger.warning(f"[x402] fertilizer-info 400: {response.text}")
            return _err("BAD_REQUEST", response.text, 400)

        elif response.status_code == 404:
            logger.warning(f"[x402] fertilizer-info 404: crop not found")
            return _err("CROP_NOT_FOUND", f"Crop '{crop}' not found in database.", 404)

        else:
            logger.error(f"[x402] fertilizer-info unexpected {response.status_code}")
            return _err("UNEXPECTED_ERROR", f"HTTP {response.status_code}: {response.text}", response.status_code)

    except ValueError as e:
        logger.error(f"[x402] Config error: {e}")
        return _err("CONFIG_ERROR", str(e))

    except Exception as e:
        logger.error(f"[x402] fertilizer-info pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))