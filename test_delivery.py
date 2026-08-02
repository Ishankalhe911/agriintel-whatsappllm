"""
test_delivery.py
Run: python test_delivery.py
No Redis, no x402, no real money — pure logic test
"""
import asyncio
from unittest.mock import AsyncMock, patch

async def test_delivery():
    print("\n=== DELIVERY TESTS ===\n")

    # Mock x402_client so no real blockchain calls
    mock_mandi_response = {
        "error": False,
        "mode": "now",
        "crop": "soybean",
        "top_mandis": [{"market": "Latur", "net_return": {"optimistic_value_inr": 85000}}]
    }
    mock_weather_response = {
        "error": False,
        "horizon_1_forecast": {"crop_stress_risk_level": "LOW"}
    }

    with patch("delivery.call_mandi_optimize", new_callable=AsyncMock) as mock_mandi, \
         patch("delivery.call_weather_risk", new_callable=AsyncMock) as mock_weather:

        mock_mandi.return_value = mock_mandi_response
        mock_weather.return_value = mock_weather_response

        from delivery import deliver

        # Test 1 — mandi routing
        session = {
            "service_type": "mandi",
            "crop": "soybean",
            "lat": 18.71,
            "lon": 76.94,
            "qty": "100",
            "radius_km": "80",      # string — tests int cast
            "time_horizon": "now",
            "variety": None
        }
        result = await deliver(session)
        assert result["error"] == False, "❌ mandi routing failed"
        mock_mandi.assert_called_once()
        call_args = mock_mandi.call_args
        assert call_args.kwargs["qty_quintals"] == 100.0, "❌ qty not cast to float"
        assert call_args.kwargs["radius_km"] == 80, "❌ radius not cast to int"
        print("✅ Test 1 PASS — mandi routing + type casting correct")

        # Test 2 — weather routing
        session = {
            "service_type": "weather",
            "crop": "soybean",
            "lat": 18.71,
            "lon": 76.94,
            "forecast_days": "7",   # string — tests int cast
            "sowing_date": None,
            "harvest_date": None
        }
        result = await deliver(session)
        assert result["error"] == False, "❌ weather routing failed"
        mock_weather.assert_called_once()
        call_args = mock_weather.call_args
        assert call_args.kwargs["forecast_days"] == 7, "❌ forecast_days not cast to int"
        print("✅ Test 2 PASS — weather routing + type casting correct")

        # Test 3 — incomplete session
        session = {
            "service_type": "mandi",
            "crop": "soybean",
            "lat": None,  # missing
            "lon": 76.94,
        }
        result = await deliver(session)
        assert result["error"] == True, "❌ incomplete session not caught"
        assert result["error_type"] == "SESSION_INCOMPLETE"
        print("✅ Test 3 PASS — incomplete session caught correctly")

        # Test 4 — unknown service
        session = {
            "service_type": "unknown_service",
            "crop": "soybean",
            "lat": 18.71,
            "lon": 76.94
        }
        result = await deliver(session)
        assert result["error"] == True
        assert result["error_type"] == "UNKNOWN_SERVICE"
        print("✅ Test 4 PASS — unknown service caught correctly")

        # Test 5 — bad qty type (list instead of float)
        session = {
            "service_type": "mandi",
            "crop": "soybean",
            "lat": 18.71,
            "lon": 76.94,
            "qty": ["bad", "data"],  # should not crash
            "radius_km": 100,
            "time_horizon": "now"
        }
        result = await deliver(session)
        assert result["error"] == False, "❌ bad qty type crashed delivery"
        call_args = mock_mandi.call_args
        assert call_args.kwargs["qty_quintals"] is None, "❌ bad qty not converted to None"
        print("✅ Test 5 PASS — bad qty type handled gracefully")

    print("\n✅ ALL DELIVERY TESTS PASSED\n")

if __name__ == "__main__":
    asyncio.run(test_delivery())