import hmac
import os

APP_PASSWORD = os.environ.get("APP_PASSWORD")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-only-insecure-secret")


def check_password(password: str) -> bool:
    if not APP_PASSWORD:
        return False
    return hmac.compare_digest(password, APP_PASSWORD)


def is_authed(request_or_websocket) -> bool:
    return request_or_websocket.session.get("authed") is True
