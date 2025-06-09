"""Cache configuration for MyJobSpyAI."""
import logging
from typing import Any, Optional, Callable, TypeVar, Type
from functools import wraps
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

# Type variable for generic function typing
T = TypeVar('T')

logger = logging.getLogger('cache')

class CacheConfig:
    """Configuration for caching."""

    def __init__(
        self,
        enabled: bool = True,
        backend: str = 'redis',
        ttl: int = 3600,
        redis_url: str = 'redis://localhost:6379/0',
        namespace: str = 'myjobspyai',
        **kwargs
    ):
        """Initialize cache configuration."""
        self.enabled = enabled
        self.backend = backend
        self.ttl = ttl
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis_client = None

        if self.enabled and self.backend == 'redis':
            try:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=False,  # We'll handle encoding/decoding
                    **kwargs
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enabled = False

    def get_key(self, key: str) -> str:
        """Get the namespaced cache key."""
        return f"{self.namespace}:{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            ttl = ttl if ttl is not None else self.ttl
            serialized = self._serialize(value)
            return self.redis_client.setex(
                name=self.get_key(key),
                time=ttl,
                value=serialized
            )
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache."""
        if not self.enabled or not self.redis_client:
            return default

        try:
            serialized = self.redis_client.get(self.get_key(key))
            if serialized is None:
                return default
            return self._deserialize(serialized)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            return bool(self.redis_client.delete(self.get_key(key)))
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def clear(self, pattern: str = '*') -> int:
        """Clear keys matching a pattern from the cache."""
        if not self.enabled or not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(self.get_key(pattern))
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache with pattern {pattern}: {e}")
            return 0

    def get_or_set(
        self,
        key: str,
        default: Any = None,
        ttl: Optional[int] = None,
        callback: Optional[Callable[[], T]] = None
    ) -> T:
        """Get a value from the cache, or set it if it doesn't exist."""
        if not self.enabled or not self.redis_client:
            return default() if callable(default) else default

        value = self.get(key)
        if value is not None:
            return value

        if callback is not None:
            value = callback()
        elif callable(default):
            value = default()
        else:
            value = default

        if value is not None:
            self.set(key, value, ttl=ttl)

        return value

    def cached(
        self,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        key_func: Optional[Callable[..., str]] = None
    ) -> Callable:
        """Decorator to cache function results."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                if not self.enabled or not self.redis_client:
                    return func(*args, **kwargs)

                # Generate cache key
                if key_func is not None:
                    cache_key = key_func(*args, **kwargs)
                elif key is not None:
                    cache_key = key
                else:
                    # Generate key from function name and arguments
                    key_parts = [func.__module__, func.__name__]
                    if args:
                        key_parts.append(','.join(str(arg) for arg in args))
                    if kwargs:
                        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

                    key_str = ':'.join(key_parts)
                    cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result

                # Call the function and cache the result
                logger.debug(f"Cache miss for {cache_key}")
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result

            return wrapper
        return decorator

    def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage in the cache."""
        return pickle.dumps({
            'value': value,
            'version': 1,
            'timestamp': datetime.utcnow().isoformat()
        })

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize a value from the cache."""
        try:
            result = pickle.loads(data)
            if isinstance(result, dict) and 'value' in result:
                return result['value']
            return result
        except (pickle.PickleError, AttributeError, EOFError, ImportError, IndexError) as e:
            logger.error(f"Error deserializing cached data: {e}")
            return None

# Global cache instance
cache = CacheConfig()

def get_cache() -> CacheConfig:
    """Get the global cache instance."""
    return cache

def init_cache(config: Optional[dict] = None) -> CacheConfig:
    """Initialize the global cache with the given configuration."""
    global cache
    if config is not None:
        cache = CacheConfig(**config)
    return cache
