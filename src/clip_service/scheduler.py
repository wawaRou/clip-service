"""Fair scheduling of ready cameras with independent start-to-start rate limits."""

from collections.abc import Collection, Mapping


class FrameScheduler:
    """Rotate eligible cameras without accumulating unused inference capacity."""

    def __init__(self, rates: Mapping[str, float], now: float) -> None:
        self._rates = dict(rates)
        self._deadlines = dict.fromkeys(rates, now)
        self._turns = {name: turn for turn, name in enumerate(rates)}
        self._next_turn = len(rates)

    def set_rates(self, rates: Mapping[str, float], now: float) -> None:
        """Keep unchanged schedules; start added or adjusted cameras immediately."""
        for name in self._rates.keys() - rates.keys():
            del self._deadlines[name]
            del self._turns[name]
        for name, rate in rates.items():
            if self._rates.get(name) != rate:
                self._deadlines[name] = now
                self._turns[name] = self._next_turn
                self._next_turn += 1
        self._rates = dict(rates)

    def due(self, now: float, ready: Collection[str]) -> list[str]:
        """Return ready, rate-eligible cameras in least-recently-served order."""
        return sorted(
            (name for name in ready if self._deadlines[name] <= now),
            key=lambda name: self._turns[name],
        )

    def started(self, name: str, now: float) -> None:
        """Reserve the next turn relative to this inference's start, not completion."""
        self._deadlines[name] = now + 1 / self._rates[name]
        self._turns[name] = self._next_turn
        self._next_turn += 1

    def delay(self, now: float, ready: Collection[str]) -> float:
        """Wait for eligible queued work, or a frame notification when empty."""
        if not ready:
            return 0.1
        return max(0.0, min(self._deadlines[name] for name in ready) - now)
