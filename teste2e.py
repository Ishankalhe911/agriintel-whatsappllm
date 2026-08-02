"""
test_e2e.py
───────────
End-to-end integration test.
Mocks ONLY the orchestrator (fixed intents).
Everything else is REAL:
    - Redis session (Upstash)
    - delivery.py routing
    - x402_client.py signing
    - Algorand mainnet transaction
    - agriintellect.site endpoints

WARNING: This costs real USDC from your float wallet.
Each mandi test = $0.10 USDC
Each weather test = $0.083 USDC

Run: python test_e2e.py
"""

import asyncio
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from session_store import SessionStore
from delivery import deliver

# ─── Mock Intents (what orchestrator would normally provide) ──────────────────

MOCK_INTENTS = [
    {
        "name": "Soybean Mandi — Latur region, with qty",
        "phone": "9999999991",
        "create": {
            "crop": "soybean",
            "qty": "200",
            "intent": "sell_crop",
            "service_type": "mandi"
        },
        "location": {"lat": 18.40, "lon": 76.56},
        "extras": {"radius_km": 80, "time_horizon": "now"},
    },
    {
        "name": "Onion Mandi — Nashik region, no qty (price only mode)",
        "phone": "9999999992",
        "create": {
            "crop": "onion",
            "qty": None,
            "intent": "check_price",
            "service_type": "mandi"
        },
        "location": {"lat": 20.00, "lon": 73.78},
        "extras": {"radius_km": 100},
    },
    {
        "name": "Weather Risk — Soybean, Marathwada",
        "phone": "9999999993",
        "create": {
            "crop": "soybean",
            "qty": None,
            "intent": "check_weather",
            "service_type": "weather"
        },
        "location": {"lat": 18.71, "lon": 76.94},
        "extras": {"forecast_days": 7},
    },
    {
        "name": "Cotton Mandi — Vidarbha region, with qty",
        "phone": "9999999994",
        "create": {
            "crop": "cotton",
            "qty": "50",
            "intent": "sell_crop",
            "service_type": "mandi"
        },
        "location": {"lat": 20.71, "lon": 76.51},
        "extras": {"radius_km": 100, "time_horizon": "now"},
    },
    {
        "name": "Weather Risk — Cotton, sowing date given",
        "phone": "9999999995",
        "create": {
            "crop": "cotton",
            "qty": None,
            "intent": "check_weather",
            "service_type": "weather"
        },
        "location": {"lat": 20.71, "lon": 76.51},
        "extras": {"forecast_days": 10, "sowing_date": "2026-06-15"},
    },
]

# ─── Single Test Runner ───────────────────────────────────────────────────────

async def run_single_test(store: SessionStore, intent: dict, index: int):
    print(f"\n{'='*60}")
    print(f"TEST {index + 1}: {intent['name']}")
    print(f"{'='*60}")

    # Step 1 — Create session (mocking orchestrator output)
    session_id = store.create_session(
        phone=intent["phone"],
        crop=intent["create"]["crop"],
        qty=str(intent["create"]["qty"]) if intent["create"]["qty"] else None,
        intent=intent["create"]["intent"],
        service_type=intent["create"]["service_type"]
    )

    if not session_id:
        print(f"❌ FAILED — Could not create Redis session")
        return False

    print(f"✅ Session created: {session_id}")

    # Step 2 — Update location (mocking farmer GPS share)
    loc = intent["location"]
    updated = store.update_location(
        session_id,
        lat=loc["lat"],
        lon=loc["lon"],
        location_name="test_location"
    )

    if not updated:
        print(f"❌ FAILED — Could not update location in Redis")
        return False

    print(f"✅ Location saved: {loc['lat']}, {loc['lon']}")

    # Step 3 — Save extra params (mocking orchestrator extras)
    if intent.get("extras"):
        store.update_session_data(session_id, **intent["extras"])
        print(f"✅ Extras saved: {intent['extras']}")

    # Step 4 — Simulate payment confirmed (mock Razorpay webhook)
    store.update_payment_status(session_id, "paid")
    print(f"✅ Payment status set to 'paid'")

    # Step 5 — Retrieve full session (what main.py passes to delivery)
    session = store.get_session(session_id)
    if not session:
        print(f"❌ FAILED — Could not retrieve session from Redis")
        return False

    print(f"✅ Session retrieved from Redis")
    print(f"   service_type: {session['service_type']}")
    print(f"   crop: {session['crop']}")
    print(f"   qty: {session['qty']}")
    print(f"   lat: {session['lat']}, lon: {session['lon']}")

    # Step 6 — Call delivery (real x402 transaction)
    print(f"\n🚀 Firing real x402 transaction to agriintellect.site...")
    print(f"   ⚠️  This costs real USDC from float wallet")

    try:
        result = await deliver(session)
    except Exception as e:
        print(f"❌ FAILED — delivery.py raised exception: {e}")
        return False

    # Step 7 — Evaluate result
    if result.get("error"):
        print(f"❌ FAILED — Endpoint returned error:")
        print(f"   error_type: {result.get('error_type')}")
        print(f"   error_reason: {result.get('error_reason')}")
        return False

    print(f"\n✅ SUCCESS — Real data received!")
    print(f"\n--- Response Preview ---")

    # Print key fields only — full response can be huge
    if session["service_type"] == "mandi":
        print(f"mode: {result.get('mode')}")
        print(f"crop: {result.get('crop')}")
        print(f"is_llm_estimate: {result.get('is_llm_estimate')}")
        nearest = result.get("nearest_mandi", {})
        print(f"nearest_mandi: {nearest.get('market')}")
        top = result.get("top_mandis", [])
        if top:
            print(f"top_mandis[0]: {top[0].get('market')} — "
                  f"₹{top[0].get('exact_scraped_data', {}).get('modal_price_per_quintal')}/q")

    elif session["service_type"] == "weather":
        h1 = result.get("horizon_1_forecast", {})
        print(f"crop_stress_risk: {h1.get('crop_stress_risk_level')}")
        print(f"operational_risk: {h1.get('operational_risk_level')}")
        print(f"irrigation_recommended: {h1.get('irrigation_recommended')}")
        print(f"next_rain_date: {h1.get('next_rain_date')}")
        print(f"partial_data: {result.get('partial_data')}")

    # Clean up session after test
    store.clear_session(session_id)
    print(f"\n✅ Session cleaned up from Redis")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("\n🌿 AgriIntel — End to End Integration Test")
    print("============================================")
    print("Mocking: orchestrator.py (fixed intents)")
    print("Real:    Redis, delivery.py, x402_client, Algorand mainnet")
    print("Cost:    ~$0.50 USDC total for all 5 tests")
    print("============================================\n")

    # Confirm before spending real USDC
    confirm = input("⚠️  This will spend real USDC. Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted. No funds spent.")
        return

    # Which tests to run
    print("\nWhich tests to run?")
    print("  a) All 5 tests (~$0.50 USDC)")
    print("  b) Mandi only — Tests 1,2,4 (~$0.30 USDC)")
    print("  c) Weather only — Tests 3,5 (~$0.17 USDC)")
    print("  d) Single test (cheapest — Test 1 only, $0.10 USDC)")

    choice = input("\nChoice (a/b/c/d): ").lower().strip()

    if choice == 'a':
        tests = MOCK_INTENTS
    elif choice == 'b':
        tests = [MOCK_INTENTS[0], MOCK_INTENTS[1], MOCK_INTENTS[3]]
    elif choice == 'c':
        tests = [MOCK_INTENTS[2], MOCK_INTENTS[4]]
    elif choice == 'd':
        tests = [MOCK_INTENTS[0]]
    else:
        print("Invalid choice. Running Test 1 only.")
        tests = [MOCK_INTENTS[0]]

    store = SessionStore()
    results = []

    for i, intent in enumerate(tests):
        success = await run_single_test(store, intent, i)
        results.append((intent["name"], success))

        # Small delay between tests to avoid rate limits
        if i < len(tests) - 1:
            print(f"\n⏳ Waiting 3s before next test...")
            await asyncio.sleep(3)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} — {name}")
        if success:
            passed += 1

    print(f"\n{passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 Full pipeline verified — ready to build orchestrator.py")
    else:
        print("\n⚠️  Fix failures before building orchestrator.py")


if __name__ == "__main__":
    asyncio.run(main())