import json
import os

try:
    import redis
except Exception:
    redis = None


_CLIENT = None
_DISABLED = False
TTL_SECONDS = int(os.environ.get("HOOKBRAIN_REDIS_TTL_SECONDS", "2592000"))


def _client():
    global _CLIENT, _DISABLED
    if _DISABLED or redis is None:
        return None
    if _CLIENT is not None:
        return _CLIENT

    try:
        redis_url = os.environ.get("REDIS_URL", "").strip()
        if redis_url:
            _CLIENT = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1,
            )
        else:
            _CLIENT = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                db=int(os.environ.get("REDIS_DB", "0")),
                password=os.environ.get("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1,
            )
        _CLIENT.ping()
        return _CLIENT
    except Exception:
        _DISABLED = True
        _CLIENT = None
        return None


def _set_json(key, payload):
    client = _client()
    if client is None:
        return False
    try:
        client.setex(key, TTL_SECONDS, json.dumps(payload))
        return True
    except Exception:
        return False


def _get_json(key):
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def cache_scan(record):
    return _set_json(f"hookbrain:scan:{record['id']}", record)


def get_cached_scan(scan_id):
    return _get_json(f"hookbrain:scan:{scan_id}")


def cache_rewrites(scan_id, payload):
    return _set_json(f"hookbrain:rewrites:{scan_id}", payload)


def get_cached_rewrites(scan_id):
    return _get_json(f"hookbrain:rewrites:{scan_id}")
