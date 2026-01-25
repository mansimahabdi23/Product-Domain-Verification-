import time
from typing import Any, Optional
import hashlib
import pickle
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple in-memory cache with optional persistence"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, persist: bool = False):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.persist = persist
        self.cache_file = Path("cache/cache_data.pkl")
        
        if persist and self.cache_file.exists():
            self._load_from_disk()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        logger.debug(f"Cache hit for key: {key}")
        return value
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
        logger.debug(f"Cached value for key: {key}")
        
        # Persist to disk if enabled
        if self.persist:
            self._save_to_disk()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        if self.persist and self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        current_time = time.time()
        valid_count = 0
        expired_count = 0
        
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp <= self.ttl:
                valid_count += 1
            else:
                expired_count += 1
        
        return {
            'total_items': len(self.cache),
            'valid_items': valid_count,
            'expired_items': expired_count,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl
        }
    
    def _save_to_disk(self):
        """Save cache to disk"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded cache from disk: {len(self.cache)} items")
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            self.cache = {}


# Global cache instance
cache = SimpleCache(max_size=500, ttl=1800, persist=True)
