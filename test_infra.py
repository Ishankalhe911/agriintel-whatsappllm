"""
test_session_store.py
Run: python test_session_store.py
Requires: REDIS_URL env var set to your Upstash URL
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from session_store import SessionStore

def test_all():
    store = SessionStore()
    print("\n=== SESSION STORE TESTS ===\n")

    # Test 1 — create session
    session_id = store.create_session(
        phone="9876543210",
        crop="soybean",
        qty="100",
        intent="mandi",
        service_type="mandi"
    )
    assert session_id is not None, "❌ create_session returned None"
    print(f"✅ Test 1 PASS — created session: {session_id}")

    # Test 2 — get session
    session = store.get_session(session_id)
    assert session is not None, "❌ get_session returned None"
    assert session["crop"] == "soybean", "❌ crop mismatch"
    assert session["language"] == "mr", "❌ language default missing"
    assert session["payment_status"] == "pending", "❌ payment status wrong"
    print(f"✅ Test 2 PASS — retrieved session, language={session['language']}")

    # Test 3 — update location
    result = store.update_location(session_id, lat=18.71, lon=76.94, location_name="Latur")
    assert result is True, "❌ update_location failed"
    session = store.get_session(session_id)
    assert session["lat"] == 18.71, "❌ lat not saved"
    assert session["lon"] == 76.94, "❌ lon not saved"
    print(f"✅ Test 3 PASS — location updated: {session['lat']}, {session['lon']}")

    # Test 4 — update session data (variety + radius_km)
    result = store.update_session_data(
        session_id,
        variety="Local",
        radius_km=80,
        forecast_days=7
    )
    assert result is True, "❌ update_session_data failed"
    session = store.get_session(session_id)
    assert session["variety"] == "Local", "❌ variety not saved"
    assert session["radius_km"] == 80, "❌ radius_km not saved"
    print(f"✅ Test 4 PASS — dynamic fields saved: variety={session['variety']}")

    # Test 5 — update payment status
    result = store.update_payment_status(session_id, "paid")
    assert result is True, "❌ update_payment_status failed"
    session = store.get_session(session_id)
    assert session["payment_status"] == "paid", "❌ payment status not updated"
    print(f"✅ Test 5 PASS — payment status: {session['payment_status']}")

    # Test 6 — get session by phone
    session_by_phone = store.get_session_by_phone("9876543210")
    assert session_by_phone is not None, "❌ get_session_by_phone failed"
    assert session_by_phone["session_id"] == session_id, "❌ wrong session returned"
    print(f"✅ Test 6 PASS — found session by phone")

    # Test 7 — orphan cleanup on create
    session_id_2 = store.create_session(
        phone="9876543210",  # same phone
        crop="cotton",
        qty="50",
        intent="mandi",
        service_type="mandi"
    )
    old_session = store.get_session(session_id)
    assert old_session is None, "❌ orphaned session not cleaned up"
    print(f"✅ Test 7 PASS — orphaned session cleaned up on restart")

    # Test 8 — clear session
    store.clear_session(session_id_2)
    cleared = store.get_session(session_id_2)
    assert cleared is None, "❌ session not cleared"
    print(f"✅ Test 8 PASS — session cleared")

    print("\n✅ ALL SESSION STORE TESTS PASSED\n")

if __name__ == "__main__":
    test_all()