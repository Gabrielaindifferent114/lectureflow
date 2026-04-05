"""Tests for _LRUCache in pipeline."""

from src.core.pipeline import _LRUCache


class TestLRUCache:
    def test_basic_set_get(self):
        cache = _LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("missing") is None

    def test_evicts_oldest(self):
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_order(self):
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache.get("a")  # refresh "a"
        cache["c"] = 3  # should evict "b" (oldest after refresh)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_overwrite_refreshes(self):
        cache = _LRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 10  # overwrite and refresh "a"
        cache["c"] = 3   # should evict "b"
        assert cache.get("a") == 10
        assert cache.get("b") is None

    def test_maxsize_one(self):
        cache = _LRUCache(maxsize=1)
        cache["a"] = 1
        cache["b"] = 2
        assert cache.get("a") is None
        assert cache.get("b") == 2
