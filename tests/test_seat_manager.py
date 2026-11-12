"""Unit tests for seat ROI geometry and person-to-seat matching (report Section 4.5)."""
import pytest

from src.seat_manager import SeatManager, SeatROI


def test_overlap_fraction_is_share_of_person_box_inside_seat():
    seat = SeatROI("seat_01", (0, 0, 100, 100))
    # Person box 100 x 200, of which the top 100 x 100 lies inside the seat.
    assert seat.overlap_fraction((0, 0, 100, 200)) == pytest.approx(0.5)


def test_overlap_fraction_is_zero_without_intersection():
    seat = SeatROI("seat_01", (0, 0, 100, 100))
    assert seat.overlap_fraction((200, 200, 300, 300)) == 0.0


def test_iou_of_identical_boxes_is_one():
    seat = SeatROI("seat_01", (10, 10, 60, 60))
    assert seat.iou_with((10, 10, 60, 60)) == pytest.approx(1.0)


def test_match_person_to_seat_respects_overlap_threshold():
    mgr = SeatManager(overlap_threshold=0.30)
    mgr.add_seat((0, 0, 100, 100))
    mgr.add_seat((300, 0, 400, 100))
    matched = mgr.match_person_to_seat((0, 0, 100, 200))       # 0.5 overlap with seat_01
    assert [s.seat_id for s in matched] == ["seat_01"]
    assert mgr.match_person_to_seat((0, 80, 100, 280)) == []   # 0.1 overlap: below threshold


def test_seats_round_trip_through_json(tmp_path):
    mgr = SeatManager()
    mgr.add_seat((1, 2, 3, 4), label="window desk")
    mgr.add_seat((5, 6, 7, 8))
    path = tmp_path / "seats_cam0.json"
    mgr.save_to_json(path)

    loaded = SeatManager()
    loaded.load_from_json(path)
    assert [(s.seat_id, s.bbox) for s in loaded.seats] == [("seat_01", (1, 2, 3, 4)), ("seat_02", (5, 6, 7, 8))]
