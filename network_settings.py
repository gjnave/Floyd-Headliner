"""Local, LAN, and temporary-link access settings for Floyd Headliner."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parent / "network_settings.json"
MODES = {"local", "lan", "public"}
HASH_ITERATIONS = 600_000


def default_settings() -> dict:
    return {"version": 1, "mode": "local", "username": "", "salt": "", "digest": ""}


def read_settings() -> dict:
    if not SETTINGS_PATH.is_file():
        return default_settings()
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if data.get("version") != 1 or data.get("mode") not in MODES:
            raise ValueError("Unsupported network settings format or mode.")
        for field in ("username", "salt", "digest"):
            if not isinstance(data.get(field), str):
                raise ValueError(f"Network settings field {field!r} is invalid.")
        if bool(data["salt"]) != bool(data["digest"]) or (data["digest"] and not data["username"]):
            raise ValueError("Network login settings are incomplete.")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Network settings could not be read: {error}") from error


def password_hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), HASH_ITERATIONS,
    ).hex()


def verify_login(username: str, password: str, settings: dict) -> bool:
    if not username or not password or not settings.get("salt") or not settings.get("digest"):
        return False
    try:
        user_matches = hmac.compare_digest(username.encode("utf-8"), settings["username"].encode("utf-8"))
        digest_matches = hmac.compare_digest(password_hash(password, settings["salt"]), settings["digest"])
        return user_matches and digest_matches
    except (TypeError, ValueError):
        return False


def save_settings(mode: str, username: str, password: str) -> dict:
    if mode not in MODES:
        raise ValueError("Choose Local only, Local network, or Temporary public link.")
    read_settings()  # Reject damaged settings rather than silently replacing them.
    username = (username or "").strip()
    password = password or ""
    result = default_settings()
    result["mode"] = mode
    if mode != "local" and password:
        username = username or "ggf"
        if len(username) > 32 or any(ord(c) < 32 for c in username):
            raise ValueError("Username must be 32 characters or fewer, without control characters.")
        result["username"] = username
        result["salt"] = secrets.token_hex(16)
        result["digest"] = password_hash(password, result["salt"])
    SETTINGS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
