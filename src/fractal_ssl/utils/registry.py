"""Light-weight registry utility used to plug datasets and models."""

from __future__ import annotations

from typing import Dict, Generic, Iterable, Iterator, MutableMapping, TypeVar

T = TypeVar("T")

__all__ = ["Registry"]


class Registry(Generic[T]):
    """Simple name → object mapping with helpful errors."""

    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        if key in self._items:
            raise KeyError(f"{self._name} '{key}' already registered")
        self._items[key] = value

    def get(self, key: str) -> T:
        if key not in self._items:
            options = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self._name} '{key}'. Available: {options}")
        return self._items[key]

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:  # pragma: no cover - repr not critical for tests
        return f"Registry(name={self._name!r}, keys={list(self._items)})"
