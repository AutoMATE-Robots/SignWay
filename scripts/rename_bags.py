#!/usr/bin/env python3
# one-off: renamed timestamp-named bags to rosbag2-keller-t*, keeping .npz cache names in sync, used 2026-08-08, kept for provenance
r"""
rename_bags.py -- rename the timestamp-named bags on R2 to rosbag2-keller-t30, t31, ...
and keep the local .npz cache filenames in sync.

WHAT IT TOUCHES
  * only bags whose directory name looks like  rosbag2_<timestamp>
  * bags already named rosbag2-keller-t*/e* are LEFT ALONE
  * numbering starts at --start (default 30) in chronological order, since the
    directory names sort chronologically (YYYY_MM_DD-HH_MM_SS)

DRY RUN BY DEFAULT. Nothing moves until you pass --apply.

  python rename_bags.py --remote r2:rosbags --cache-dir $SCRATCH/bag_cache
  python rename_bags.py --remote r2:rosbags --cache-dir $SCRATCH/bag_cache --apply

A mapping CSV is written either way (old_path,new_name,old_cache,new_cache) -- hand
that to whoever is annotating so their labels line up with the new names.

NOTE ON COST: renaming on object storage is server-side copy + delete, not a metadata
tweak. It is free of egress on R2 but does cost Class A operations and takes time
proportional to the data. The cache rename is instant.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

TS = re.compile(r"^rosbag2_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr.strip()}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="r2:rosbags")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--prefix", default="rosbag2-keller-t")
    ap.add_argument("--map", default="bag_rename_map.csv")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    print(f"listing {a.remote} ...")
    listing = sh(f"rclone lsf --recursive --files-only {a.remote}")
    bags = sorted({ln[: -len("/metadata.yaml")]
                   for ln in listing.splitlines() if ln.endswith("/metadata.yaml")})
    if not bags:
        print("no bags found"); sys.exit(1)

    to_rename = [b for b in bags if TS.match(Path(b).name)]
    skipped = [b for b in bags if not TS.match(Path(b).name)]
    print(f"{len(bags)} bags total: {len(to_rename)} to rename, {len(skipped)} left alone")
    if skipped:
        print("  leaving alone: " + ", ".join(Path(s).name for s in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))

    # chronological: the timestamp is in the directory name, so lexical == chronological
    to_rename.sort(key=lambda p: Path(p).name)

    cache_dir = Path(a.cache_dir)
    rows = []
    for i, old in enumerate(to_rename):
        new = f"{a.prefix}{a.start + i}"
        old_cache = cache_dir / (old.replace("/", "_") + ".npz")
        new_cache = cache_dir / f"{new}.npz"
        rows.append({"old_path": old, "new_name": new,
                     "old_cache": old_cache.name, "new_cache": new_cache.name,
                     "cache_present": old_cache.exists()})

    with open(a.map, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"mapping written to {a.map}")

    missing = [r for r in rows if not r["cache_present"]]
    if missing:
        print(f"\nWARNING: {len(missing)} bags have no matching cache "
              f"(these were never ingested, or the cache name differs):")
        for r in missing[:10]:
            print("   " + r["old_path"])

    print("\nfirst 10 renames:")
    for r in rows[:10]:
        print(f"   {r['old_path']}  ->  {r['new_name']}")
    if len(rows) > 10:
        print(f"   ... and {len(rows)-10} more")

    if not a.apply:
        print("\nDRY RUN -- nothing moved. Re-run with --apply when the mapping looks right.")
        return

    print("\napplying ...")
    for i, r in enumerate(rows, 1):
        src = f"{a.remote}/{r['old_path']}"
        dst = f"{a.remote}/{r['new_name']}"
        print(f"[{i}/{len(rows)}] {r['old_path']} -> {r['new_name']}")
        try:
            sh(f"rclone move '{src}' '{dst}' --transfers=4")
        except RuntimeError as e:
            print(f"   R2 MOVE FAILED: {e}")
            continue
        oc, nc = cache_dir / r["old_cache"], cache_dir / r["new_cache"]
        if oc.exists():
            oc.rename(nc)
            print(f"   cache {oc.name} -> {nc.name}")
    print("\ndone. Empty day folders can be cleaned with:")
    print(f"   rclone rmdirs {a.remote} --leave-root")


if __name__ == "__main__":
    main()
