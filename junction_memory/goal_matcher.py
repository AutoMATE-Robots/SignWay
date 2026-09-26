# junction_memory/goal_matcher.py — resolve a room number to a printed range.
#
# Your AR gate already parses floor-room ranges for goal affinity scoring;
# prefer importing that scorer over this fallback so memory and the gate agree
# on what "6-118 is inside 6-115 to 6-189" means.
import re

_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)\s*(?:to|-|–|through)\s*(?:(\d+)\s*-\s*)?(\d+)",
                    re.I)


def parse_range(label):
    """'6-115 to 6-189' -> (6, 115, 189). None if not a range."""
    m = _RANGE.search(str(label))
    if not m:
        return None
    floor, lo, floor2, hi = m.group(1), m.group(2), m.group(3), m.group(4)
    if floor2 is not None and floor2 != floor:
        return None                      # spans floors: not our case
    return int(floor), int(lo), int(hi)


def parse_room(goal):
    """'6-118' -> (6, 118). Tolerates 'room 6-118', 'UMSEC 6-202'."""
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(goal))
    return (int(m.group(1)), int(m.group(2))) if m else None


def match(goal, labels):
    """matcher(goal, [stored_labels]) -> label | None, for MemoryRuntime."""
    room = parse_room(goal)
    if room is None:
        return None
    floor, num = room
    best, span = None, None
    for lab in labels:
        if str(lab).strip().casefold() == str(goal).strip().casefold():
            return lab                                   # exact hit wins
        rng = parse_range(lab)
        if rng and rng[0] == floor and rng[1] <= num <= rng[2]:
            width = rng[2] - rng[1]
            if span is None or width < span:              # tightest range wins
                best, span = lab, width
    return best


if __name__ == "__main__":
    labels = ["6-115 to 6-189", "6-201 to 6-225", "umsec 6-202"]
    for g, exp in (("6-118", "6-115 to 6-189"), ("6-210", "6-201 to 6-225"),
                   ("6-202", "6-201 to 6-225"), ("6-999", None),
                   ("cafeteria", None)):
        got = match(g, labels)
        print(f"{g:12} -> {got!r:20} {'ok' if got == exp else 'MISMATCH ' + repr(exp)}")