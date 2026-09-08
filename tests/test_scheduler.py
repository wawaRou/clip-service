import pytest

from clip_service.scheduler import FrameScheduler


def test_only_ready_cameras_are_due_and_empty_sources_do_not_spin():
    scheduler = FrameScheduler({"door": 25, "room": 25}, now=0)
    assert scheduler.due(0, ready=[]) == []
    assert scheduler.delay(0, ready=[]) == 0.1
    assert scheduler.due(0.01, ready=["room"]) == ["room"]


def test_rate_limit_is_measured_from_start_without_waiting_an_extra_period():
    scheduler = FrameScheduler({"room": 25}, now=0)
    scheduler.started("room", now=0.01)
    assert scheduler.due(0.035, ready=["room"]) == []
    assert scheduler.delay(0.035, ready=["room"]) == pytest.approx(0.015)
    assert scheduler.due(0.051, ready=["room"]) == ["room"]


def test_slow_inference_can_continue_immediately_without_catching_up_missed_ticks():
    scheduler = FrameScheduler({"room": 25}, now=0)
    scheduler.started("room", now=0)
    assert scheduler.due(0.1, ready=["room"]) == ["room"]
    scheduler.started("room", now=0.1)
    assert scheduler.due(0.1, ready=["room"]) == []
    assert scheduler.delay(0.1, ready=["room"]) == pytest.approx(0.04)


def test_busy_cameras_take_turns_even_with_different_deadlines():
    scheduler = FrameScheduler({"door": 100, "room": 25, "yard": 25}, now=0)
    selected = []
    for now in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        name = scheduler.due(now, ready=["door", "room", "yard"])[0]
        selected.append(name)
        scheduler.started(name, now)
    assert selected == ["door", "room", "yard", "door", "room", "yard"]


def test_missing_or_rate_limited_camera_does_not_hold_up_another():
    scheduler = FrameScheduler({"door": 10, "room": 2}, now=0)
    scheduler.started("room", now=0)
    scheduler.started("door", now=0)
    assert scheduler.due(0.11, ready=["door", "room"]) == ["door"]
    assert scheduler.due(1, ready=["door"]) == ["door"]


def test_reload_preserves_unchanged_limits_and_resets_changed_or_added_cameras():
    scheduler = FrameScheduler({"door": 4, "room": 2, "removed": 5}, now=10)
    scheduler.started("door", now=10.03)
    scheduler.started("room", now=10.07)
    scheduler.set_rates({"door": 4, "room": 4, "yard": 2}, now=10.1)
    ready = ["door", "room", "yard"]
    assert scheduler.due(10.1, ready=ready) == ["room", "yard"]
    scheduler.started("room", now=10.12)
    scheduler.started("yard", now=10.14)
    assert scheduler.due(10.29, ready=ready) == ["door"]
    assert scheduler.delay(10.2, ready=["door"]) == pytest.approx(0.08)


def test_reload_to_and_from_empty_registry():
    scheduler = FrameScheduler({}, now=0)
    assert scheduler.due(0, ready=[]) == []
    scheduler.set_rates({"door": 5}, now=10)
    assert scheduler.due(10, ready=["door"]) == ["door"]
    scheduler.set_rates({}, now=10.1)
    assert scheduler.due(11, ready=[]) == []
    assert scheduler.delay(11, ready=[]) == 0.1
