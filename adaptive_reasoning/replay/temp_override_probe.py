"""Before/after probe on the six temporary bags: re-ask ONLY the frames our gate
could fire on, with the override clause now in the hint.  ~30-60 calls, minutes.
Mode B (notice on a SEPARATE plate) is also handled: if any plate in the frame
carries notice text, its crop is sent alongside the goal plate's crop.
"""
import json, os, csv, collections
import numpy as np
from pathlib import Path
from PIL import Image
from adaptive_reasoning.reasoner import Reasoner, VLMClient, structural_hint, has_override

S = os.environ['SCRATCH']
ann = {r['bag']: r for r in csv.DictReader(open('adaptive_reasoning/annotations/ar_extra-sep14.csv'))}
temp = sorted(b for b in ann if '2026_09_14' in b)
client = VLMClient(model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", base_url="http://localhost:8000/v1",
                   reasoning_effort=None, rpm=0)
out = Path(S) / "ar_eval"
reasoner = Reasoner(client, model_tag="32B:override", cache_dir=out / "vlm_cache_override",
                    payload_dir=out / "payloads" / "override", ledger_path=out / "ledger_override.jsonl",
                    eval_tag="temp_override")

for bag in temp:
    a = ann[bag]; goal = a['goal']
    recs = [json.loads(l) for l in open(f"{S}/ar_replay/{bag}.jsonl")]
    crops_dir = Path(f"{S}/ar_replay/{bag}_crops")
    byframe = collections.defaultdict(list)
    for r in recs:
        if r.get("crop"): byframe[r["frame"]].append(r)
    # frames our gate could fire on
    cand = [f for f, rs in byframe.items()
            if any((r.get("R") or 0) >= 0.5 and (r.get("ell") or 0) >= 0.55 for r in rs)]
    cand = sorted(cand)[:5]
    print("=" * 96); print(f"{bag}  goal={goal}  CORRECT={a['decision']}  notice='{a.get('temp_text','')}'")
    for f in cand:
        rs = byframe[f]
        goal_plate = max(rs, key=lambda r: (r.get("R") or 0))
        notice = next((r for r in rs if has_override(r.get("text") or "")
                       and r["plate_id"] != goal_plate["plate_id"]), None)
        imgs = [np.asarray(Image.open(crops_dir / goal_plate["crop"]).convert("RGB"))]
        texts = [goal_plate["text"]]
        if notice:                                   # mode B: send the notice crop too
            imgs.append(np.asarray(Image.open(crops_dir / notice["crop"]).convert("RGB")))
            texts.append(notice["text"])
        hint = structural_hint(" | ".join(texts), goal)
        ansr = reasoner.ask(goal, texts, imgs, meta={"bag": bag, "frame": f, "probe": "override"}, hint=hint)
        got = ansr.direction if ansr.applicable else "not_applicable"
        mark = "OK " if got == a['decision'] else "XX "
        print(f"  {mark}f{f:<5} notice_crop={'yes' if notice else 'no '} -> {got:15s} "
              f"| {(ansr.summary or '')[:90]}")
