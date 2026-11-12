"""Unit tests for the per-seat temporal inference engine (report Section 4.6)."""
from src.temporal_inference import SeatState, SeatTemporalBuffer, TemporalInferenceEngine


def feed(buf: SeatTemporalBuffer, pattern: str) -> list[SeatState]:
    """Feed a string of '1' (person present) / '0' (absent) frames into a buffer."""
    return [buf.update(ch == "1") for ch in pattern]


def make_buffer() -> SeatTemporalBuffer:
    # Same values as the pilot configuration: window 6, threshold 0.5, cooldown 10.
    return SeatTemporalBuffer("seat_01", window_size=6, occupied_threshold=0.5, cooldown_frames=10)


def test_starts_unknown_then_settles_to_vacant():
    buf = make_buffer()
    states = feed(buf, "0" * 20)
    assert states[0] == SeatState.UNKNOWN
    assert states[-1] == SeatState.VACANT


def test_single_frame_detection_of_passer_by_does_not_occupy_seat():
    buf = make_buffer()
    feed(buf, "0" * 20)
    states = feed(buf, "1" + "0" * 10)
    assert all(s == SeatState.VACANT for s in states)


def test_sustained_presence_switches_seat_to_occupied():
    buf = make_buffer()
    feed(buf, "0" * 20)
    feed(buf, "1" * 20)
    assert buf.state == SeatState.OCCUPIED
    assert buf.is_occupied


def test_single_missed_frame_does_not_vacate_occupied_seat():
    buf = make_buffer()
    feed(buf, "0" * 20 + "1" * 20)
    states = feed(buf, "0" + "1" * 5)
    assert all(s == SeatState.OCCUPIED for s in states)


def test_cooldown_blocks_a_second_transition_too_soon():
    buf = SeatTemporalBuffer("seat_01", window_size=4, occupied_threshold=0.5, cooldown_frames=50)
    feed(buf, "0" * 60)                 # settle to vacant, cooldown elapsed
    feed(buf, "1" * 5)                  # switch to occupied (the transition restarts the cooldown)
    assert buf.state == SeatState.OCCUPIED
    states = feed(buf, "0" * 10)        # clear absence, but still within the 50-frame cooldown
    assert all(s == SeatState.OCCUPIED for s in states)


def test_engine_updates_every_registered_seat():
    engine = TemporalInferenceEngine(window_size=6, occupied_threshold=0.5, cooldown_frames=0)
    engine.register_seats(["a", "b"])
    for _ in range(10):
        engine.update({"a"})
    assert engine.get_state("a") == SeatState.OCCUPIED
    assert engine.get_state("b") == SeatState.VACANT
    assert engine.occupied_count == 1
