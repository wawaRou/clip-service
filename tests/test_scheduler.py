import pytest

from clip_service.scheduler import FrameScheduler


def test_cameras_with_equal_rates_keep_the_same_schedule():
    scheduler = FrameScheduler({"entrance": 5, "room": 5}, now=0.0)

    assert scheduler.due(0.0) == ["entrance", "room"]
    # Reading due work does not consume it.
    assert scheduler.due(0.0) == ["entrance", "room"]
    scheduler.complete("entrance", now=0.03)
    scheduler.complete("room", now=0.07)

    assert scheduler.due(0.19) == []
    assert scheduler.delay(0.19) == pytest.approx(0.01)
    assert scheduler.due(0.2) == ["entrance", "room"]

    scheduler.complete("entrance", now=0.23)
    scheduler.complete("room", now=0.27)
    assert scheduler.delay(0.3) == pytest.approx(0.1)
    assert scheduler.due(0.4) == ["entrance", "room"]


def test_late_completion_skips_missed_periods_without_moving_the_schedule():
    scheduler = FrameScheduler({"entrance": 5}, now=0.0)
    scheduler.complete("entrance", now=0.55)

    assert scheduler.due(0.55) == []
    assert scheduler.delay(0.55) == pytest.approx(0.05)
    assert scheduler.due(0.61) == ["entrance"]

    scheduler.complete("entrance", now=1.0)
    assert scheduler.due(1.0) == []
    assert scheduler.delay(1.0) == pytest.approx(0.2)


def test_equal_deadlines_prioritize_the_camera_that_waited_longest():
    scheduler = FrameScheduler({"entrance": 5, "room": 5}, now=0.0)
    scheduler.complete("room", now=0.03)
    scheduler.complete("entrance", now=0.07)

    assert scheduler.due(0.2) == ["room", "entrance"]


def test_slow_inference_does_not_starve_any_camera():
    scheduler = FrameScheduler({"entrance": 10, "room": 10, "garden": 10}, now=0.0)
    completed = []
    for now in [0.0, 0.35, 0.7, 1.05, 1.4, 1.75]:
        camera = scheduler.due(now)[0]
        completed.append(camera)
        scheduler.complete(camera, now=now + 0.35)

    assert completed == ["entrance", "room", "garden", "entrance", "room", "garden"]


def test_camera_rate_overrides_have_independent_periods():
    scheduler = FrameScheduler({"entrance": 4, "room": 2}, now=10.0)
    scheduler.complete("entrance", now=10.03)
    scheduler.complete("room", now=10.07)

    assert scheduler.due(10.25) == ["entrance"]
    scheduler.complete("entrance", now=10.3)
    assert scheduler.due(10.5) == ["room", "entrance"]
    assert scheduler.delay(10.6) == 0.0


def test_reloading_rates_preserves_unchanged_schedules_and_applies_changes():
    scheduler = FrameScheduler({"entrance": 4, "room": 2, "removed": 5}, now=10.0)
    scheduler.complete("entrance", now=10.03)
    scheduler.complete("room", now=10.07)

    scheduler.set_rates({"entrance": 4, "room": 4, "garden": 2}, now=10.1)

    assert scheduler.due(10.1) == ["room", "garden"]
    scheduler.complete("room", now=10.12)
    scheduler.complete("garden", now=10.14)
    assert scheduler.due(10.25) == ["entrance"]
    assert scheduler.due(10.36) == ["entrance", "room"]
    scheduler.complete("entrance", now=10.36)
    scheduler.complete("room", now=10.37)
    assert scheduler.due(10.6) == ["entrance", "garden", "room"]


def test_empty_schedule_can_accept_cameras_and_become_empty_again():
    scheduler = FrameScheduler({}, now=0.0)
    assert scheduler.due(0.0) == []
    assert scheduler.delay(0.0) == 0.1

    scheduler.set_rates({"entrance": 5}, now=10.0)
    assert scheduler.due(10.0) == ["entrance"]
    scheduler.set_rates({}, now=10.1)
    assert scheduler.due(11.0) == []
    assert scheduler.delay(11.0) == 0.1


def test_fractional_periods_remain_due_on_the_original_time_grid():
    scheduler = FrameScheduler({"entrance": 5}, now=0.0)
    for now in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        assert scheduler.due(now) == ["entrance"]
        scheduler.complete("entrance", now=now)
        assert scheduler.due(now) == []
