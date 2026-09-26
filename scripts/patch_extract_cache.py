#!/usr/bin/env python3
"""
patch_extract_cache.py -- fix extract_cache.py's broken import.

extract_cache.py did:
    from tfds_builder import IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO, _resize

but patch_builder_for_cache.py (2026-08-09) made the builder cache-first: it no
longer reads bags, so _resize was removed. The extractor has been stranded since.

Fix: inline _resize VERBATIM from
    _attic/2026-08/signway_dataset/tfds_builder.py.prebag  (lines 101-106)
so the extractor is self-contained and a future builder refactor cannot strand it
again. The topic/distro constants still import fine (they survive at lines 53-55).

Byte-identical guarantee: this is the exact function that produced the existing 96
caches -- bilinear, clip to 0..255, cast to uint8. Mixing resize behaviours inside
one training set would be a silent data bug.

    python patch_extract_cache.py            # patches ~/SignWay/scripts/extract_cache.py
"""
from pathlib import Path

p = Path.home() / "SignWay" / "scripts" / "extract_cache.py"
s = p.read_text()

OLD = ("from bag_to_episode import interp_odom, read_bag          # noqa: E402\n"
       "from tfds_builder import IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO, _resize  # noqa: E402\n")

NEW = '''from bag_to_episode import interp_odom, read_bag          # noqa: E402
from tfds_builder import IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO  # noqa: E402

# ---------------------------------------------------------------------------
# _resize is INLINED here, verbatim from the pre-cache builder
# (_attic/2026-08/signway_dataset/tfds_builder.py.prebag lines 101-106).
#
# The builder became cache-first on 2026-08-09 and dropped this function, which
# broke the import. Copying it here rather than re-importing keeps the extractor
# self-contained AND guarantees new caches are byte-identical to the existing 96:
# bilinear, clipped, cast to uint8. Do not "improve" this function.
# ---------------------------------------------------------------------------
import tensorflow as tf  # noqa: E402

IMAGE_SIZE = (224, 224)


def _resize(img: np.ndarray) -> np.ndarray:
    out = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    return tf.cast(tf.clip_by_value(out, 0, 255), tf.uint8).numpy()
'''

if "_resize is INLINED here" in s:
    print("already patched -- nothing to do")
elif OLD in s:
    p.write_text(s.replace(OLD, NEW, 1))
    print(f"patched {p}")
else:
    # import line may differ slightly; fall back to a targeted replacement
    import re
    m = re.search(r"^from tfds_builder import .*_resize.*$", s, re.M)
    if not m:
        raise SystemExit(
            "could not find the tfds_builder import line in extract_cache.py -- "
            "paste the file's import block and patch by hand")
    s = s.replace(m.group(0),
                  "from tfds_builder import IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO  "
                  "# noqa: E402\n" + NEW.split("# noqa: E402\n", 1)[1], 1)
    p.write_text(s)
    print(f"patched {p} (fallback path)")

print("\nverify:")
print("  python -c \"import sys; sys.path.insert(0,'scripts'); "
      "sys.path.insert(0,'tools'); sys.path.insert(0,'signway_dataset'); "
      "import extract_cache; print('import OK')\"")