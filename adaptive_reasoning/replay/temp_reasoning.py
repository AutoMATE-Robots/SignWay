"""Dump every VLM exchange on the six temporary-sign bags: what plate was sent,
what the model said, and what the annotation says the correct action was."""
import json, os, csv, collections
S = os.environ['SCRATCH']
ann = {r['bag']: r for r in csv.DictReader(open('adaptive_reasoning/annotations/ar_extra-sep14.csv'))}
temp = [b for b in ann if '2026_09_14' in b]
recs = collections.defaultdict(list)
for l in open(f"{S}/ar_eval/ledger.jsonl"):
    try: r = json.loads(l)
    except: continue
    if r.get("bag") in temp and r.get("eval") == "crop_pass":
        recs[r["bag"]].append(r)
for bag in sorted(temp):
    a = ann[bag]
    print("=" * 100)
    print(f"{bag}   goal={a['goal']}   CORRECT={a['decision']}   conflicting={a.get('temp_conflicting')}")
    print(f"   permanent sign: {a.get('sign_text','')[:70]}")
    print(f"   temp notice   : {a.get('temp_text','')[:70]}")
    rs = sorted(recs[bag], key=lambda r: (r.get("frame") or 0))
    if not rs:
        print("   (no VLM calls logged)"); continue
    seen = set()
    for r in rs:
        k = (r.get("plate_id"), r.get("frame"))
        if k in seen: continue
        seen.add(k)
        got = r.get("direction") if r.get("applicable") else "NOT-APPLICABLE"
        ok = "OK " if got == a['decision'] else "   "
        print(f"  {ok}f{r.get('frame'):<5} R={r.get('R')} ell={r.get('ell')} -> {str(got):15s} "
              f"plate='{(r.get('text') or '')[:52]}'")
        raw = (r.get("raw") or r.get("summary") or "").replace("\n", " ")
        if raw: print(f"        reason: {raw[:230]}")
