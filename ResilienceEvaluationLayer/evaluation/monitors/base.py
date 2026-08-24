#!/usr/bin/env python3

"""Common lifecycle contracts for online evaluation monitors."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class EpisodeMonitor(Protocol):
    """Minimal lifecycle implemented by an online episode monitor."""

    monitor_name: str

    def reset(self) -> None:
        ...

    def update(self, step_data: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def get_summary(self) -> Mapping[str, Any]:
        ...


class MonitorRegistry:
    """Own monitor lifecycle without exposing collectors to runner internals."""

    def __init__(
        self,
        monitors: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._monitors: Dict[str, Any] = {}
        for name, monitor in (monitors or {}).items():
            self.register(name, monitor)

    def register(
        self,
        name: str,
        monitor: Any,
        *,
        replace: bool = False,
    ) -> Any:
        key = str(name).strip()
        if not key:
            raise ValueError("Monitor name cannot be empty")
        if key in self._monitors and not replace:
            raise ValueError(f"Monitor already registered: {key}")
        for method_name in ("reset", "update", "get_summary"):
            if not callable(getattr(monitor, method_name, None)):
                raise TypeError(
                    f"Monitor {key!r} does not implement {method_name}()"
                )
        declared = str(getattr(monitor, "monitor_name", key))
        if declared != key:
            raise ValueError(
                f"Monitor name mismatch: registered={key}, declared={declared}"
            )
        self._monitors[key] = monitor
        return monitor

    def get(self, name: str, default: Any = None) -> Any:
        return self._monitors.get(str(name), default)

    def require(self, name: str) -> Any:
        key = str(name)
        if key not in self._monitors:
            raise KeyError(f"Monitor is not registered: {key}")
        return self._monitors[key]

    def update(self, name: str, step_data: Mapping[str, Any]) -> Dict[str, Any]:
        result = self.require(name).update(step_data)
        return dict(result or {})

    def update_all(
        self,
        payloads: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Route typed payloads to every named monitor present in ``payloads``.

        A monitor is deliberately skipped when it has no payload.  This keeps
        the registry generic without pretending that safety and future
        monitors consume the same step schema.
        """

        unknown = set(payloads).difference(self._monitors)
        if unknown:
            raise KeyError(
                "Monitor payload supplied for unregistered monitor(s): "
                + ", ".join(sorted(unknown))
            )
        return {
            name: self.update(name, payloads[name])
            for name in self._monitors
            if name in payloads
        }

    def reset(self) -> None:
        for monitor in self._monitors.values():
            monitor.reset()

    def summary(self, name: str) -> Dict[str, Any]:
        monitor = self.get(name)
        if monitor is None:
            return {}
        return copy.deepcopy(dict(monitor.get_summary() or {}))

    def summaries(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.summary(name) for name in self._monitors}

    def as_mapping(self) -> Mapping[str, Any]:
        return dict(self._monitors)

    def names(self) -> Iterable[str]:
        return tuple(self._monitors)


__all__ = ["EpisodeMonitor", "MonitorRegistry"]
