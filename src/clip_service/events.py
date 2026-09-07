"""Bounded event fan-out; clients reconcile candidates after reconnecting."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    event_id: int
    event_type: str
    data: dict

    def as_sse(self) -> bytes:
        payload = json.dumps(self.data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return (f"id: {self.event_id}\nevent: {self.event_type}\ndata: {payload}\n\n").encode()


class EventBroker:
    def __init__(self, subscriber_queue_size: int = 32) -> None:
        self.closed = threading.Event()
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[Event | None]] = set()
        self._next_id = 1
        self._queue_size = subscriber_queue_size

    def publish(self, event_type: str, data: dict) -> Event:
        with self._lock:
            event = Event(self._next_id, event_type, data)
            self._next_id += 1
            if not self.closed.is_set():
                self._broadcast(event)
            return event

    def close(self) -> None:
        with self._lock:
            self.closed.set()
            self._broadcast(None)

    def _broadcast(self, event: Event | None) -> None:
        for subscriber in self._subscribers:
            if subscriber.full():
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass  # A subscriber may have consumed the oldest event meanwhile.
            subscriber.put_nowait(event)

    @contextmanager
    def subscribe(self) -> Iterator[queue.Queue[Event | None]]:
        subscriber: queue.Queue[Event | None] = queue.Queue(self._queue_size)
        with self._lock:
            if self.closed.is_set():
                subscriber.put_nowait(None)
            else:
                self._subscribers.add(subscriber)
        try:
            yield subscriber
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)
