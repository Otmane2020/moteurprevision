"""
Redis cache service – caches forecast results.
Falls back gracefully if REDIS_URL is not set.
"""
import os
import json
import hashlib
import logging
from typing import Optional

log = logging.getLogger("moteur.cache")

REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 3600))  # 1 hour default

_client = None

if REDIS_URL:
    try:
        import redis
        _client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _client.ping()
        log.info("Redis connected – cache enabled (TTL=%ds)", CACHE_TTL)
    except Exception as exc:
        log.warning("Redis unavailable: %s – running without cache", exc)
        _client = None
else:
    log.info("No REDIS_URL – running without cache")


def _make_key(prefix: str, payload: dict) -> str:
    """Create a deterministic cache key from a dict payload."""
    raw = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"moteur:{prefix}:{h}"


def get_cached(prefix: str, payload: dict) -> Optional[dict]:
    """Return cached result or None."""
    if _client is None:
        return None
    try:
        key = _make_key(prefix, payload)
        val = _client.get(key)
        if val:
            log.info("Cache HIT  %s", key)
            return json.loads(val)
    except Exception as exc:
        log.warning("Cache get error: %s", exc)
    return None


def set_cached(prefix: str, payload: dict, result: dict, ttl: int = CACHE_TTL):
    """Store result in cache."""
    if _client is None:
        return
    try:
        key = _make_key(prefix, payload)
        _client.setex(key, ttl, json.dumps(result, default=str))
        log.info("Cache SET  %s (TTL=%ds)", key, ttl)
    except Exception as exc:
        log.warning("Cache set error: %s", exc)


def flush_cache():
    """Flush all moteur:* keys."""
    if _client is None:
        return 0
    try:
        keys = _client.keys("moteur:*")
        if keys:
            _client.delete(*keys)
        return len(keys)
    except Exception as exc:
        log.warning("Cache flush error: %s", exc)
        return 0


def cache_info() -> dict:
    """Return cache statistics."""
    if _client is None:
        return {"status": "disabled", "reason": "REDIS_URL not set"}
    try:
        info = _client.info("memory")
        keys = len(_client.keys("moteur:*"))
        return {
            "status": "connected",
            "url": REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL[:20],
            "moteur_keys": keys,
            "used_memory_human": info.get("used_memory_human"),
            "ttl_seconds": CACHE_TTL,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
