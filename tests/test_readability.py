"""The gate must fire exactly when the sign becomes legible — not before, not after."""
import numpy as np

from c3_reasoning.readability import (assess, expected_height_px, laplacian_variance,
                                      text_height_from_bbox)


def _text_crop(h=40, w=200, sharp=True):
    """A crop with text-like structure: hard black/white vertical strokes on white."""
    img = np.full((h, w), 235.0)
    for x in range(10, w - 10, 14):
        img[6:h - 6, x:x + 5] = 20.0
    if not sharp:                       # crude motion blur: average along x
        k = 9
        img = np.stack([img[:, max(0, i - k):i + k].mean(axis=1) for i in range(w)], axis=1)
    return img


def test_blur_score_separates_sharp_from_blurred():
    assert laplacian_variance(_text_crop(sharp=True)) > \
           laplacian_variance(_text_crop(sharp=False)) * 3


def test_flat_image_has_no_texture():
    assert laplacian_variance(np.full((40, 40), 128.0)) == 0.0


def test_tiny_text_is_rejected_before_touching_pixels():
    r = assess(_text_crop(), bbox=(0, 0, 100, 12), det_conf=0.9)
    assert not r.legible and r.reason == "too_small"
    assert r.blur_var == 0.0            # cheapest check first: pixels never read


def test_blurred_text_is_rejected_even_when_big():
    r = assess(_text_crop(h=60, sharp=False), bbox=(0, 0, 300, 60), det_conf=0.9,
               min_blur_var=500.0)
    assert not r.legible and r.reason == "too_blurry"


def test_low_detector_confidence_is_rejected():
    r = assess(_text_crop(), bbox=(0, 0, 200, 40), det_conf=0.2)
    assert not r.legible and r.reason == "low_detector_confidence"


def test_big_sharp_confident_text_is_legible():
    r = assess(_text_crop(h=40), bbox=(0, 0, 200, 40), det_conf=0.9)
    assert r.legible and r.reason == ""


def test_no_detection_is_not_legible():
    r = assess(None, bbox=None, det_conf=0.0)
    assert not r.legible and r.reason == "no_text_detected"


def test_gate_fires_once_as_the_robot_approaches():
    """THE contribution, as a test.

    A sign with 8cm characters, 480px camera, fy=500. The robot drives from 8m to 1m. The gate
    must stay shut while the text is too small and open once — at a predictable distance — not
    flicker on and off. Monotone approach => single threshold crossing => one VLM call.
    """
    # 8cm characters, fy=500px -> 5.0, 6.7, 10.0, 13.3, 20.0, 26.7, 40.0 px at these ranges.
    # The 22px floor is crossed between 2.0m and 1.5m, so the sign becomes readable at ~1.8m.
    fy, char_h, min_h = 500.0, 0.08, 22.0
    legible = [expected_height_px(char_h, d, fy) >= min_h
               for d in [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]]
    assert legible == [False, False, False, False, False, True, True]
    # exactly one False->True transition: the gate opens once and stays open
    assert sum(1 for a, b in zip(legible, legible[1:]) if a != b) == 1


def test_expected_height_matches_bbox_height_convention():
    assert text_height_from_bbox((10, 100, 210, 140)) == 40.0
    assert expected_height_px(0.08, 2.0, 500.0) == 20.0
