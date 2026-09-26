"""What OUR gate actually did on the six temporary-sign bags:
the plate it chose, the frame it fired on, the answer, and the model's reasoning.
Uses per_approach.csv (what the gate did) joined to the ledger (what was asked)."""
import json, os, csv, collections
S = os.environ['SCRATCH']
ann = {r['bag']: r for r in csv.DictReader(open('adaptive_reasoning/annotations/ar_extra-sep14.csv'))}
temp = sorted(b for b in ann if '2026_09_14' in b)

pa = {}
for r in csv.DictReader(open(f"{S}/ar_eval/final/per_approach.csv")):
    if r["gate"] == "ours" and r["bag"] in temp:
        pa[r["bag"]] = r

led = collections.defaultdict(list)
for l in open(f"{S}/ar_eval/ledger.jsonl"):
    try: r = json.loads(l)
    except: continue
    if r.get("bag") in temp and r.get("eval") == "crop_pass" and r.get("goal") == ann[r["bag"]]["goal"]:
        led[r["bag"]].append(r)

for bag in temp:
    a = ann[bag]; p = pa.get(bag, {})
    print("=" * 104)
    print(f"{bag}")
    print(f"  goal={a['goal']}   CORRECT={a['decision']}   conflicting={a.get('temp_conflicting')}")
    print(f"  permanent: {a.get('sign_text','')[:74]}")
    print(f"  notice   : {a.get('temp_text','')[:74]}")
    print(f"  OURS -> calls={p.get('calls')} correct={p.get('correct')} decided={p.get('decided')} "
          f"overhead={p.get('overhead_s')}s")
    rs = sorted(led[bag], key=lambda r: (r.get("frame") or 0))
    if not rs:
        print("  (no own-goal calls in ledger)"); continue
    # the gate fires on the goal-relevant plate once it is legible: show those asks
    fired = [r for r in rs if (r.get("R") or 0) >= 0.5 and (r.get("ell") or 0) >= 0.55]
    show = fired or rs
    print(f"  --- {len(show)} candidate asks (R>=0.5, ell>=0.55)" if fired else
          f"  --- no ask met the gate; showing all {len(rs)} asks")
    for r in show[:6]:
        got = r.get("direction") if r.get("applicable") else "NOT-APPLICABLE"
        mark = "OK " if got == a['decision'] else "XX "
        print(f"  {mark}f{r.get('frame'):<5} R={r.get('R')} ell={r.get('ell')} -> {str(got):15s}")
        print(f"       plate : '{(r.get('text') or '')[:78]}'")
        raw = (r.get("raw") or r.get("summary") or "").replace("\n", " ")
        print(f"       reason: {raw[:300]}")
    # did ANY plate in this bag carry the notice text?
    notice_words = [w for w in (a.get('temp_text','') or '').upper().split() if len(w) > 3]
    hits = {(r.get('text') or '')[:60] for r in rs
            if any(w in (r.get('text') or '').upper() for w in notice_words)}
    print(f"  notice text seen in any asked plate: {'YES -> ' + str(list(hits)[:2]) if hits else 'NO'}")
