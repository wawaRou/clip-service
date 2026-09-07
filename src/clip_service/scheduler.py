"""Fixed-rate camera scheduling, independent of threads and clocks."""

from collections.abc import Mapping
from math import floor


class FrameScheduler:
    """Schedule fresh frames without accumulating missed detection periods."""

    def __init__(self, rates: Mapping[str, float], now: float) -> None:
        self._rates = dict(rates)
        self._starts = dict.fromkeys(rates, now)
        self._deadlines = dict.fromkeys(rates, now)
        self._turns = {name: turn for turn, name in enumerate(rates)}
        self._next_turn = len(rates)

    def set_rates(self, rates: Mapping[str, float], now: float) -> None:
        """Keep unchanged schedules; start added or adjusted cameras immediately."""
        for name in self._rates.keys() - rates.keys():
            del self._starts[name]
            del self._deadlines[name]
            del self._turns[name]
        for name, rate in rates.items():
            if self._rates.get(name) != rate:
                self._starts[name] = now
                self._deadlines[name] = now
                self._turns[name] = self._next_turn
                self._next_turn += 1
        self._rates = dict(rates)

    def due(self, now: float) -> list[str]:
        """Return due cameras in deadline order without consuming their turn."""
        return sorted(
            (name for name, deadline in self._deadlines.items() if deadline <= now),
            key=lambda name: (self._deadlines[name], self._turns[name]),
        )

    def complete(self, name: str, now: float) -> None:
        """Advance the camera's fixed schedule beyond its completion time."""
        rate = self._rates[name]
        start = self._starts[name]
        next_period = floor((now - start) * rate) + 1
        deadline = start + next_period / rate
        # Rounding at an exact period boundary must not schedule another turn now.
        if deadline <= now:
            deadline = start + (next_period + 1) / rate
        self._deadlines[name] = deadline
        self._turns[name] = self._next_turn
        self._next_turn += 1

    def delay(self, now: float) -> float:
        """Return seconds until work is due, or a short idle wait if empty."""
        if not self._deadlines:
            return 0.1
        return max(0.0, min(self._deadlines.values()) - now)
