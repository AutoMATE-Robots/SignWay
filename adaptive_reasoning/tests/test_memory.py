def test_memory_recognises_same_sign_under_ocr_noise_but_not_a_door_plate():
    from adaptive_reasoning.evidence.buffer import stable_plate_id
    from adaptive_reasoning.memory import Memory, MemoryUpdate
    m = Memory("g")
    m.apply(MemoryUpdate(stable_plate_id("Rooms 43-58 | Rooms 1-37, 63-71"), "vlm", "turn_left", 0))
    assert not m.is_novel(stable_plate_id("Rooms 43 58 | Rooms 1-37 63-71"))   # dash dropped by OCR
    m2 = Memory("g")
    m2.apply(MemoryUpdate(stable_plate_id("6-146 to 6-199 | 6-201 to 6-254"), "vlm", "turn_left", 0))
    assert m2.is_novel(stable_plate_id("6-201"))                                # door plate stays novel
    assert m2.is_novel(stable_plate_id("Rooms 251-276"))                        # different sign stays novel


def test_exact_memory_for_baselines_never_fuzzy_matches():
    from adaptive_reasoning.evidence.buffer import stable_plate_id
    from adaptive_reasoning.memory import Memory, MemoryUpdate
    m = Memory("g", fuzzy=False)
    full = stable_plate_id("5-231 to 5-250 | 5-201 to 5-217 | 5-117 to 5-196")
    part = stable_plate_id("5-201 to 5-217 | 5-117 to 5-196")     # partial read: overlapping but different id
    m.apply(MemoryUpdate(full, "not_applicable", None, 0))
    assert m.is_novel(part), "exact memory must not fuzzy-match a partial read"
    assert not m.is_novel(full)
    fz = Memory("g", fuzzy=True)
    fz.apply(MemoryUpdate(full, "not_applicable", None, 0))
    assert not fz.is_novel(part), "our fuzzy memory should recognise the partial read"