import json
import logging
import os
import re
import uuid
from typing import Optional, Dict, Any
import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_TTL_SECONDS = 1800  # 15 minutes


def normalize_phone(phone: str) -> str:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"+91{digits}"
    elif len(digits) == 11 and digits.startswith("0"):
        return f"+91{digits[1:]}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    return f"+{digits}"


def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


class SessionStore:
    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.client = get_redis_client()
        self.ttl = ttl_seconds

    def create_session(
        self,
        phone: str,
        crop: str,
        qty: str,
        intent: str,
        service_type: str
    ) -> Optional[str]:
        clean_phone = normalize_phone(phone)

        # Flaw 2 fix — clean up orphaned session before creating new one
        try:
            existing_session_id = self.client.get(f"active_session:{clean_phone}")
            if existing_session_id:
                self.client.delete(f"session:{existing_session_id}")
                logger.info(
                    f"Cleaned up orphaned session {existing_session_id} "
                    f"for phone {clean_phone[-4:]}"
                )
        except redis.RedisError as e:
            # Non-fatal — log and continue creating new session
            logger.warning(f"Could not clean orphaned session: {e}")

        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        session_data: Dict[str, Any] = {
            "session_id": session_id,
            "phone": clean_phone,
            "language": "mr",
            "crop": crop,
            "qty": qty,
            "intent": intent,
            "service_type": service_type,
            "lat": None,
            "lon": None,
            "location_name": None,
            "payment_status": "pending"
        }

        session_key = f"session:{session_id}"
        phone_key = f"active_session:{clean_phone}"

        try:
            self.client.setex(session_key, self.ttl, json.dumps(session_data))
            self.client.setex(phone_key, self.ttl, session_id)
            logger.info(f"Created session {session_id} for phone {clean_phone[-4:]}")
            return session_id
        except redis.RedisError as e:
            logger.error(f"Failed to create session in Redis: {e}")
            return None

    def _save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Persists session updates with sliding window TTL.
        Every interaction resets the 15-minute countdown.
        Handles zombie sessions (-1) and expired sessions (-2) correctly.
        """
        session_id = session_data.get("session_id")
        if not session_id:
            return False

        session_key = f"session:{session_id}"

        try:
            # Only check if key is completely dead (-2)
            # -1 (zombie) and >0 (normal) both get full TTL reset
            if self.client.ttl(session_key) == -2:
                logger.warning(f"Session {session_id} expired. Cannot update.")
                return False

            # Flaw 1 fix — always reset to full TTL (sliding window)
            self.client.setex(session_key, self.ttl, json.dumps(session_data))

            clean_phone = session_data.get("phone")
            if clean_phone:
                self.client.setex(
                    f"active_session:{clean_phone}",
                    self.ttl,
                    session_id
                )

            return True
        except redis.RedisError as e:
            logger.error(f"Redis error while saving session {session_id}: {e}")
            return False

    def update_location(
        self,
        session_id: str,
        lat: float,
        lon: float,
        location_name: Optional[str] = None
    ) -> bool:
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        session_data["lat"] = lat
        session_data["lon"] = lon
        if location_name:
            session_data["location_name"] = location_name

        return self._save_session(session_data)

    def update_payment_status(self, session_id: str, status: str) -> bool:
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        session_data["payment_status"] = status
        return self._save_session(session_data)

    def update_session_data(self, session_id: str, **kwargs) -> bool:
        """
        Dynamically updates any optional keys in session.
        Called by orchestrator to inject: variety, sowing_date,
        harvest_date, radius_km, forecast_days, language etc.
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return False

        for key, value in kwargs.items():
            session_data[key] = value

        return self._save_session(session_data)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session_key = f"session:{session_id}"
        try:
            raw_data = self.client.get(session_key)
            if raw_data:
                return json.loads(raw_data)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching session {session_id}: {e}")
        return None

    def get_session_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        clean_phone = normalize_phone(phone)
        phone_key = f"active_session:{clean_phone}"
        try:
            session_id = self.client.get(phone_key)
            if session_id:
                return self.get_session(session_id)
        except redis.RedisError as e:
            logger.error(f"Error fetching session by phone {clean_phone[-4:]}: {e}")
        return None

    def clear_session(self, session_id: str) -> None:
        session_data = self.get_session(session_id)
        try:
            if session_data:
                clean_phone = session_data.get("phone")
                if clean_phone:
                    self.client.delete(f"active_session:{clean_phone}")
            self.client.delete(f"session:{session_id}")
            logger.info(f"Cleared session {session_id}")
        except redis.RedisError as e:
            logger.error(f"Error clearing session {session_id}: {e}")