#!/usr/bin/env python3
"""
make_bags_rows.py -- emit the BAGS rows for tfds_builder.py from the day-5
annotations, resolving timestamp -> t-name through data/bag_rename_map.csv.

Run AFTER ingest_day5.sh (the map must contain the new rows).

    cd ~/SignWay && python scripts/make_bags_rows.py            # print rows
    cd ~/SignWay && python scripts/make_bags_rows.py --check    # stats only

Two things it handles that a hand-edit would get wrong:
  * legible_frame None (straight bags) -> 0. build_steps compares `i < flip_frame`
    and would raise TypeError on None.
  * Amundson bags are a SECOND BUILDING; the composition report breaks the mix
    down per building so the straight-fraction target (~30%) can be checked.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# (stamp, legible_frame, decision, turn_done_frame, stride)
AMUNDSON = [
    ("rosbag2_2026_08_11-17_50_09", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-17_51_30", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-17_52_36", 0,    "turn_right", 363,  3),
    ("rosbag2_2026_08_11-17_53_40", 0,    "turn_left",  316,  3),
    ("rosbag2_2026_08_11-17_54_58", 0,    "straight",   None, 3),
    ("rosbag2_2026_08_11-17_55_58", 56,   "turn_right", 426,  3),
    ("rosbag2_2026_08_11-17_56_46", 0,    "turn_left",  247,  3),
    ("rosbag2_2026_08_11-17_58_50", 19,   "turn_left",  371,  3),
    ("rosbag2_2026_08_11-17_59_45", 22,   "turn_right", 306,  3),
    ("rosbag2_2026_08_11-18_01_32", 82,   "turn_left",  320,  3),
    ("rosbag2_2026_08_11-18_02_41", 9,    "turn_right", 237,  3),
    ("rosbag2_2026_08_11-18_04_27", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-18_04_48", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-18_08_24", 191,  "turn_left",  448,  3),
    ("rosbag2_2026_08_11-18_11_48", 298,  "turn_right", 514,  3),
    ("rosbag2_2026_08_11-18_12_09", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-18_12_28", None, "straight",   None, 3),
    ("rosbag2_2026_08_11-18_12_54", 131,  "turn_right", 346,  3),
    ("rosbag2_2026_08_11-18_13_29", 55,   "turn_left",  289,  3),
    ("rosbag2_2026_08_11-18_14_53", 55,   "turn_right", 352,  3),
    ("rosbag2_2026_08_11-18_15_12", 358,  "turn_right", 640,  3),
    ("rosbag2_2026_08_11-18_16_20", 106,  "turn_left",  428,  3),
    ("rosbag2_2026_08_11-18_16_44", None, "straight",   None, 3),
]

KELLER = [
    ("rosbag2_2026_08_12-14_54_06", 53,  "turn_left",  366,  3),
    ("rosbag2_2026_08_12-14_54_59", 131, "turn_right", 414,  3),
    ("rosbag2_2026_08_12-15_01_17", 245, "turn_left",  493,  3),
    ("rosbag2_2026_08_12-15_01_53", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_03_56", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_05_07", 103, "turn_right", 484,  3),
    ("rosbag2_2026_08_12-15_13_17", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_14_07", 99,  "turn_left",  342,  3),
    ("rosbag2_2026_08_12-15_36_43", 66,  "turn_left",  299,  3),
    ("rosbag2_2026_08_12-15_37_34", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_13_41", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_38_02", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_39_34", 24,  "turn_left",  295,  3),
    ("rosbag2_2026_08_12-15_40_27", 21,  "turn_right", 403,  3),
    ("rosbag2_2026_08_12-15_45_02", 123, "turn_right", 442,  3),
    ("rosbag2_2026_08_12-15_46_24", 171, "turn_left",  464,  3),
    ("rosbag2_2026_08_12-15_47_54", 162, "turn_right", 425,  3),
    ("rosbag2_2026_08_12-15_48_36", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_49_29", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_50_44", 123, "turn_right", 374,  3),
    ("rosbag2_2026_08_12-15_54_00", 108, "turn_left",  397,  3),
    ("rosbag2_2026_08_12-15_55_00", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_55_22", 120, "turn_right", 349,  3),
    ("rosbag2_2026_08_12-15_56_15", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_58_23", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-15_59_43", 147, "turn_right", 415,  3),
    ("rosbag2_2026_08_12-16_14_18", 0,   "turn_left",  260,  3),
    ("rosbag2_2026_08_12-16_15_38", 179, "turn_right", 507,  3),
    ("rosbag2_2026_08_12-16_16_17", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-16_17_29", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-16_17_58", 149, "turn_left",  460,  3),
    ("rosbag2_2026_08_12-16_18_26", 0,   "straight",   None, 3),
    ("rosbag2_2026_08_12-16_18_55", 204, "turn_left",  510,  3),
    ("rosbag2_2026_08_12-16_30_10", 0,   "turn_right", 308,  3),
    ("rosbag2_2026_08_12-16_31_56", 0,   "turn_right", 262,  3),
    ("rosbag2_2026_08_12-16_43_57", 0,   "turn_right", 335,  3),
    ("rosbag2_2026_08_12-16_44_43", 102, "turn_right", 462,  3),
    ("rosbag2_2026_08_12-16_45_16", 47,  "turn_left",  406,  3),
]


def load_map(path: Path) -> dict:
    """stamp (basename) -> t-name, from the 5-column rename map."""
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] and row[1]:
                out[row[0].split("/")[-1].strip()] = row[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/bag_rename_map.csv")
    ap.add_argument("--cache-dir", default=None,
                    help="if given, verify each t-name has a .npz")
    ap.add_argument("--check", action="store_true", help="stats only, no rows")
    args = ap.parse_args()

    m = load_map(Path(args.map))
    rows, missing = [], []
    for label, table in (("AMUNDSON (2nd building)", AMUNDSON), ("KELLER day 5", KELLER)):
        block = []
        for stamp, flip, dec, td, st in table:
            t = m.get(stamp)
            if not t:
                missing.append(stamp)
                continue
            # None legible_frame (straight bags) -> 0: build_steps does `i < flip_frame`
            block.append((t, 0 if flip is None else flip, dec, td, st))
        rows.append((label, block))

    if missing:
        print(f"# !! {len(missing)} bags not in the rename map (ingest incomplete?):")
        for s in missing[:8]:
            print(f"#    {s}")
        print("#    run scripts/ingest_day5.sh first\n")

    if args.cache_dir:
        cd = Path(args.cache_dir)
        gone = [t for _, blk in rows for t, *_ in blk if not (cd / f"{t}.npz").exists()]
        if gone:
            print(f"# !! {len(gone)} t-names have no cache: {gone[:6]}\n")

    if not args.check:
        for label, block in rows:
            print(f"    # ---- {label} ----")
            for t, flip, dec, td, st in block:
                tds = "None" if td is None else str(td)
                print(f'    ("train", "{t}", {flip}, "{dec}", {tds}, {st}),')
            print()

    print("# ---------------- composition ----------------")
    grand = Counter()
    for label, block in rows:
        c = Counter(d for _, _, d, _, _ in block)
        z = sum(1 for _, f, d, _, _ in block if f == 0 and d != "straight")
        grand.update(c)
        n = sum(c.values())
        print(f"# {label:<26} n={n:>3}  straight={c['straight']:>2} "
              f"left={c['turn_left']:>2} right={c['turn_right']:>2}  "
              f"flip0_turns={z}")
    n = sum(grand.values())
    print(f"# {'NEW TOTAL':<26} n={n:>3}  straight={grand['straight']:>2} "
          f"left={grand['turn_left']:>2} right={grand['turn_right']:>2}")
    print(f"# existing 96 bags: 22 straight / 36 left / 38 right")
    print(f"# COMBINED: {96 + n} bags, straights "
          f"{22 + grand['straight']}/{96 + n} = "
          f"{100 * (22 + grand['straight']) / (96 + n):.0f}%  (target ~30%)")


if __name__ == "__main__":
    main()
