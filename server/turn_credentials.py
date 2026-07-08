import os
import time

import requests

CF_TURN_KEY_ID = os.environ.get("CF_TURN_KEY_ID")
CF_TURN_API_TOKEN = os.environ.get("CF_TURN_API_TOKEN")
CREDENTIAL_TTL_SECONDS = 86400

_cache: list[dict] | None = None
_cache_expires_at = 0.0


def get_ice_servers() -> list[dict]:
    global _cache, _cache_expires_at

    if not CF_TURN_KEY_ID or not CF_TURN_API_TOKEN:
        raise RuntimeError("CF_TURN_KEY_ID / CF_TURN_API_TOKEN are not set")

    now = time.monotonic()
    if _cache is not None and now < _cache_expires_at:
        return _cache

    response = requests.post(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{CF_TURN_KEY_ID}/credentials/generate",
        headers={"Authorization": f"Bearer {CF_TURN_API_TOKEN}"},
        json={"ttl": CREDENTIAL_TTL_SECONDS},
        timeout=10,
    )
    response.raise_for_status()
    ice_server = response.json()["iceServers"]

    _cache = [ice_server]
    _cache_expires_at = now + CREDENTIAL_TTL_SECONDS - 60
    return _cache
