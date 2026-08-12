"""The detector must gate on legibility, group a sign's lines, and locate it."""
import numpy as np

from c3_reasoning.detector import StubTextBackend, TextSignDetector, merge_boxes


def _frame(w=640, h=480, boxes=()):
    """Grey frame with sharp text-like strokes inside each given box."""
    img = np.full((h, w, 3), 200, np.uint8)
    for (x0, y0, x1, y1) in boxes:
        for x in range(int(x0) + 4, int(x1) - 4, 12):
            img[int(y0) + 3:int(y1) - 3, x:x + 4] = 15
    return img


def test_small_text_does_not_reach_the_reasoner():
    bb = (300, 200, 380, 212)                       # 12px tall — under the OCR floor
    d = TextSignDetector(StubTextBackend([(bb, 0.9)]))
    assert d.detect(_frame(boxes=[bb])) == []
    assert d.last[0][1].reason == "too_small"       # and it says why


def test_large_sharp_text_reaches_the_reasoner():
    bb = (240, 180, 460, 240)
    d = TextSignDetector(StubTextBackend([(bb, 0.9)]))
    dets = d.detect(_frame(boxes=[bb]))
    assert len(dets) == 1 and d.last[0][1].legible
    assert dets[0].label == "sign"                  # detected, deliberately NOT read


def test_a_directory_sign_is_one_sign_not_eleven():
    """The Amundson-style case: many lines, one board, one VLM call.

    Treating each text line as its own sign would fire the reasoner once per line and destroy
    the contribution. Lines stacked within the merge gap collapse into one region.
    """
    lines = [(200, 100 + 30 * i, 420, 124 + 30 * i) for i in range(8)]
    d = TextSignDetector(StubTextBackend([(b, 0.9) for b in lines]))
    dets = d.detect(_frame(boxes=lines))
    assert len(dets) == 1
    assert dets[0].bbox[1] < 105 and dets[0].bbox[3] > 300      # spans all the lines


def test_two_signs_far_apart_stay_separate():
    a, b = (60, 100, 260, 160), (400, 340, 600, 400)
    d = TextSignDetector(StubTextBackend([(a, 0.9), (b, 0.9)]))
    assert len(d.detect(_frame(boxes=[a, b]))) == 2


def test_bearing_is_positive_for_a_sign_on_the_left():
    left, right = (60, 200, 260, 260), (400, 200, 600, 260)
    d = TextSignDetector(StubTextBackend([(left, 0.9), (right, 0.9)]))
    dets = {round(x.bbox[0]): x for x in d.detect(_frame(boxes=[left, right]))}
    assert dets[60].est_bearing_rad > 0             # left of centre -> +left
    assert dets[400].est_bearing_rad < 0


def test_depth_gives_distance_and_is_robust_to_outliers():
    bb = (240, 180, 460, 240)
    d = TextSignDetector(StubTextBackend([(bb, 0.9)]))
    depth = np.full((480, 640), 50.0)               # far wall
    depth[180:240, 240:460] = 3.0                   # the sign
    depth[200:205, 300:305] = 0.01                  # a few rogue pixels
    d.set_depth(depth)
    assert abs(d.detect(_frame(boxes=[bb]))[0].est_distance_m - 3.0) < 0.1


def test_without_depth_distance_is_unknown_not_wrong():
    bb = (240, 180, 460, 240)
    d = TextSignDetector(StubTextBackend([(bb, 0.9)]))
    assert d.detect(_frame(boxes=[bb]))[0].est_distance_m == float("inf")


def test_no_text_no_detections():
    d = TextSignDetector(StubTextBackend([]))
    assert d.detect(_frame()) == []


def test_crop_includes_a_margin_around_the_sign():
    bb = (240, 180, 460, 240)
    d = TextSignDetector(StubTextBackend([(bb, 0.9)]))
    img = _frame(boxes=[bb])
    det = d.detect(img)[0]
    c = d.crop(img, det, margin=0.15)
    assert c.shape[0] > 60 and c.shape[1] > 220      # bigger than the raw box


def test_merge_is_transitive_down_a_column():
    boxes = [(0, 0, 100, 20), (0, 30, 100, 50), (0, 60, 100, 80)]
    assert len(merge_boxes(boxes, gap_px=24.0)) == 1
    assert len(merge_boxes(boxes, gap_px=5.0)) == 3
