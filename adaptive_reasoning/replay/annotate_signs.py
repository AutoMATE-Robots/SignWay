"""
annotate_signs.py — assign heatmap categories to a folder of sign frames.

Opens a small web page (stdlib only).  For each image you pick a category,
type the goal, pick the expected answer, click Save.  One image can get several
rows (different goals).  Writes <out>/<category>.csv with image,goal,expected
— exactly what fig_heatmap.py reads with --csv-dir <out> --images-root <images>.
Progress is kept in <out>/annotate_state.json, so you can stop and resume.

  python -m adaptive_reasoning.replay.annotate_signs --images figs/heatmap_pool --out heatmap --port 8765
then open http://localhost:8765  (VS Code forwards the port automatically when
run on a remote node; or run it on your laptop against a synced folder).

Categories: numeric_sign, named_goal, goal_irrelevant, unclear_sign.
Expected:   turn_left, turn_right, straight, stop, not_applicable.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

CATS = ["numeric_sign", "named_goal", "goal_irrelevant", "unclear_sign"]
EXPECTED = ["turn_left", "turn_right", "straight", "stop", "not_applicable"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}

RULES = {
    "numeric_sign": "goal = a room number INSIDE a printed range (not an endpoint); expected = that line's arrow",
    "named_goal": "goal = a place name as printed; expected = that line's arrow",
    "goal_irrelevant": "goal = something NOT on the sign; expected = not_applicable",
    "unclear_sign": "sign too far/blurry/oblique; goal + true answer anyway (abstaining is scored correct)",
}


class State:
    def __init__(self, images: Path, out: Path):
        self.images, self.out = images, out
        self.files = sorted(p for p in images.rglob("*") if p.suffix.lower() in EXTS)
        self.state_f = out / "annotate_state.json"
        self.state = json.loads(self.state_f.read_text()) if self.state_f.exists() else {"idx": 0, "rows": []}
        out.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.state_f.write_text(json.dumps(self.state, indent=1))
        by = {}
        for r in self.state["rows"]:
            by.setdefault(r["category"], []).append(r)
        for c in CATS:
            f = self.out / f"{c}.csv"
            rows = by.get(c, [])
            with open(f, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["image", "goal", "expected"])
                for r in rows:
                    w.writerow([r["image"], r["goal"], r["expected"]])

    def counts(self):
        c = {k: 0 for k in CATS}
        for r in self.state["rows"]:
            c[r["category"]] = c.get(r["category"], 0) + 1
        return c

    def rows_for(self, rel):
        return [r for r in self.state["rows"] if r["image"] == rel]


def page(st: State) -> str:
    n = len(st.files); i = max(0, min(st.state["idx"], n - 1)) if n else 0
    if not n:
        return "<h2>No images found.</h2>"
    rel = str(st.files[i].relative_to(st.images))
    counts = " · ".join(f"{k}: <b>{v}</b>" for k, v in st.counts().items())
    existing = "".join(
        f"<li>{html.escape(r['category'])} — goal <b>{html.escape(r['goal'])}</b> → {html.escape(r['expected'])} "
        f"<a href='/del?i={k}'>[x]</a></li>"
        for k, r in enumerate(st.state["rows"]) if r["image"] == rel) or "<li><i>none yet</i></li>"
    cat_radios = "".join(
        f"<label style='display:block;margin:2px 0'><input type=radio name=category value='{c}' "
        f"{'checked' if c == 'numeric_sign' else ''}> <b>{c}</b> <span style='color:#666'>— {RULES[c]}</span></label>"
        for c in CATS)
    exp_opts = "".join(f"<option value='{e}'>{e}</option>" for e in EXPECTED)
    return f"""<!doctype html><html><head><meta charset=utf-8><title>annotate</title>
<style>body{{font-family:sans-serif;margin:16px}} img{{max-width:100%;max-height:70vh;border:1px solid #ccc}}
.row{{display:flex;gap:24px}} .left{{flex:3}} .right{{flex:2}} input[type=text]{{width:100%;font-size:16px;padding:6px}}
select,button{{font-size:16px;padding:6px 10px}} .nav a{{margin-right:12px}}</style></head><body>
<div class=nav><a href='/go?i={i-1}'>&larr; prev</a> <b>{i+1} / {n}</b> <a href='/go?i={i+1}'>next &rarr;</a>
&nbsp;&nbsp;<span style='color:#444'>{counts}</span></div>
<div class=row><div class=left><p><code>{html.escape(rel)}</code></p><img src='/img?p={urllib.parse.quote(rel)}'></div>
<div class=right>
<form method=post action='/save'>
<input type=hidden name=image value='{html.escape(rel)}'><input type=hidden name=idx value='{i}'>
<h3>Category</h3>{cat_radios}
<h3>Goal (as printed on the sign)</h3><input type=text name=goal autofocus placeholder='e.g. 215  or  Lind Hall'>
<h3>Expected</h3><select name=expected>{exp_opts}</select>
<p><button type=submit name=act value=save_next>Save &amp; next</button>
&nbsp;<button type=submit name=act value=save_stay>Save, add another goal</button>
&nbsp;<a href='/go?i={i+1}'><button type=button>Skip</button></a></p>
</form>
<h4>Rows for this image</h4><ul>{existing}</ul>
</div></div></body></html>"""


def serve(st: State, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype="text/html; charset=utf-8", code=200):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
            if u.path == "/img":
                p = st.images / urllib.parse.unquote(q["p"][0])
                ext = p.suffix.lower().lstrip(".").replace("jpg", "jpeg")
                return self._send(p.read_bytes(), f"image/{ext}")
            if u.path == "/go":
                st.state["idx"] = max(0, min(int(q.get("i", ["0"])[0]), len(st.files) - 1)); st.save()
            if u.path == "/del":
                k = int(q["i"][0])
                if 0 <= k < len(st.state["rows"]):
                    st.state["rows"].pop(k); st.save()
            self._send(page(st).encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            f = urllib.parse.parse_qs(self.rfile.read(n).decode())
            g = lambda k: f.get(k, [""])[0].strip()
            if g("goal") and g("category") in CATS and g("expected") in EXPECTED:
                st.state["rows"].append({"image": g("image"), "category": g("category"),
                                         "goal": g("goal"), "expected": g("expected")})
            if g("act") == "save_next":
                st.state["idx"] = min(int(g("idx")) + 1, len(st.files) - 1)
            st.save()
            self.send_response(303); self.send_header("Location", "/"); self.end_headers()

        def log_message(self, *a):  # quiet
            pass

    print(f"{len(st.files)} images · open http://localhost:{port}  (Ctrl-C to stop; progress is saved on every action)")
    HTTPServer(("0.0.0.0", port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of frames (searched recursively)")
    ap.add_argument("--out", required=True, help="where <category>.csv files are written (use as --csv-dir later)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    st = State(Path(a.images), Path(a.out))
    st.save()
    serve(st, a.port)


if __name__ == "__main__":
    main()
