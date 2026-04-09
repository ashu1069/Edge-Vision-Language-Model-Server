"""
Shared Redis connection utilities.

Used by both the FastAPI API (async) and the worker (sync).
"""


def parse_redis_url(url: str) -> tuple[str, int, int]:
    """
    Parse a Redis URL into host, port, and db components.

    Args:
        url: Redis connection URL (e.g., 'redis://localhost:6379/0')

    Returns:
        Tuple of (host, port, db)
    """
    if url.startswith("redis://"):
        parts = url.replace("redis://", "").split("/")
        host_port = parts[0].split(":")
        host = host_port[0] if len(host_port) > 0 else "redis"
        port = int(host_port[1]) if len(host_port) > 1 else 6379
        db = int(parts[1]) if len(parts) > 1 else 0
    else:
        host, port, db = "redis", 6379, 0

    return host, port, db
