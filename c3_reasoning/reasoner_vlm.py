"""Component 3, expensive half: read the sign, think about it, decide.

This is the only place a large model runs, and it runs rarely — the readability gate in
c3_reasoning/readability.py decides when, and the leg gating in c5_orchestrator/triggers.py
makes sure it is once per sign. Everything here is allowed to be slow, because it almost never
happens.

WHAT IT IS ALLOWED TO DO: return a Decision. That is all. It never touches motors, never emits
a velocity, never picks a waypoint. The Decision becomes a pose subgoal and OmniVLA does the
driving. That separation is deliberate — the VLM moves the goalpost, the VLA plays the game.

WHY CHAIN-OF-THOUGHT AND NOT JUST OCR: a sign like "2-270 to 2-276 / Main Elevators / Restrooms"
under a left arrow, above a right arrow with different rooms, is not solved by reading the text.
Every string can be read perfectly and the answer still requires knowing WHICH ARROW GOVERNS
WHICH LINE, and that room 2-272 falls inside the range 2-270..2-276. The layout is the meaning.
So the model is asked to reason first and answer second, and the reasoning is logged — it is
the evidence for the paper, and the thing to read when a decision looks wrong.

FAILURE IS NORMAL AND MUST BE SAFE: an API call over a network, mid-run, on a robot. It will
time out, get rate-limited, and occasionally return prose instead of JSON. Every failure path
returns a Decision — STOP when unsure — because raising here would strand a moving robot.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import numpy as np

from common.interfaces import Reasoner
from common.types import Decision, DecisionType

SYSTEM_PROMPT = """You are the reasoning module of an indoor mobile robot following signs.

You are shown a photograph of a sign the robot has just been able to read, and told where the
robot is trying to go. Decide what the robot should do.

Reason step by step before answering:
1. Read every piece of text and note every arrow.
2. Work out which arrow governs which lines. On directory signs an arrow applies to the lines
   BELOW it, until the next arrow. This grouping is usually the whole meaning of the sign.
3. Check whether the destination matches any group. Ranges count: room 2-272 lies inside
   "2-270 to 2-276". A category counts too: "Main Elevators" matches a goal of "elevator".
4. If nothing matches the destination, say so and continue straight rather than guessing.
5. If the sign forbids entry or warns of a hazard, obey it — safety beats the destination.

Then answer with ONLY a JSON object, no markdown fence:
{"reasoning": "<your step-by-step reasoning, brief>",
 "text_read": "<every word you can read on the sign>",
 "decision": "<continue|turn_left|turn_right|u_turn|stop|arrived>",
 "confidence": <0.0-1.0>}

decision meanings, from the robot's point of view as it faces the sign:
  continue   - keep going the way you are pointed
  turn_left  - the destination is to your left
  turn_right - the destination is to your right
  u_turn     - go back the way you came (a dead end, or entry forbidden)
  stop       - a hazard or instruction means do not proceed
  arrived    - this sign marks the destination ITSELF. The goal names this place, with no
               arrow sending you onward (e.g. goal "the cafeteria" and the sign is the
               "Cafeteria" entrance/nameplate). You are here; the trip is over. Only use this
               when the sign labels the destination at this spot, never when an arrow points
               further on.

Set confidence below 0.5 if the sign is unreadable, ambiguous, or irrelevant to the
destination. A low-confidence honest answer is far better than a confident guess."""

_ALLOWED = {d.value: d for d in DecisionType}


def _to_png_bytes(image) -> bytes:
    from io import BytesIO

    from PIL import Image
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    buf = BytesIO()
    Image.fromarray(a[:, :, :3]).save(buf, format="PNG")
    return buf.getvalue()


def parse_decision(text: str) -> Decision:
    """Turn the model's reply into a Decision, and never raise.

    Models wrap JSON in markdown fences, prepend "Here's my answer:", or invent decision names.
    Anything unrecognised becomes STOP with the raw reply kept as the rationale, so a mangled
    response makes the robot cautious instead of crashing it — and leaves the evidence in the log.
    """
    raw = (text or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)          # first {...}, fences and preamble ignored
    if not m:
        return Decision(type=DecisionType.STOP, rationale=f"unparseable reply: {raw[:200]}",
                        confidence=0.0)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Decision(type=DecisionType.STOP, rationale=f"bad json ({e}): {raw[:200]}",
                        confidence=0.0)

    name = str(d.get("decision", "")).strip().lower()
    if name not in _ALLOWED:
        return Decision(type=DecisionType.STOP,
                        rationale=f"unknown decision {name!r}; said: {d.get('reasoning', '')}"[:400],
                        confidence=0.0)
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Decision(
        type=_ALLOWED[name],
        rationale=str(d.get("reasoning", ""))[:600],
        confidence=max(0.0, min(1.0, conf)),
        target={"text_read": str(d.get("text_read", ""))[:300]},
    )


class GeminiReasoner(Reasoner):
    """Gemini over HTTPS. Verified reachable from MSI (login node) on 2026-07-17.

    thinking_budget is the latency dial. gemini-2.5-flash reports "thinking": true, so
    chain-of-thought is a first-class parameter rather than a prompt trick: 0 disables it for a
    fast shallow read, higher values buy real deliberation at the cost of seconds. Sweeping it
    is the "how much thinking does a sign actually need" ablation.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Measured 2026-07-17 on one real directory sign (three arrow groups, eleven lines), asked
    # for "the main elevators". All four reachable models answered turn_left correctly:
    #     gemini-3.1-flash-lite            turn_left    1.44 s   <- default
    #     gemini-robotics-er-1.6-preview   turn_left    4.37 s
    #     gemini-3.5-flash                 turn_left    6.73 s
    #     gemini-3-flash-preview           turn_left   16.47 s
    # Flash-Lite is the default because it was right and 4.7x faster than 3.5-flash. Note the
    # /models endpoint is a CATALOGUE, not an entitlement — gemini-2.5-flash and
    # robotics-er-1.5-preview are listed but return 404 for new keys.

    def __init__(self, model: str = "gemini-3.1-flash-lite", api_key: Optional[str] = None,
                 thinking_budget: Optional[int] = None, timeout_s: float = 20.0,
                 max_output_tokens: int = 800):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("no GEMINI_API_KEY in the environment")
        self.thinking_budget = thinking_budget
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.calls = 0                  # the headline metric: how often did we have to think?
        self.total_s = 0.0
        self.log = []                   # one entry per call — the paper's raw data

    def _ssl_context(self):
        """Build a verifying SSL context that works inside the Isaac container.

        The container's Python has no CA bundle where OpenSSL looks by default, so an
        unconfigured urlopen dies with CERTIFICATE_VERIFY_FAILED even though Gemini is
        reachable. Try, in order: an explicit bundle from the env, certifi's bundle, the
        Ubuntu system bundle. Only if SIGNWAY_SSL_NOVERIFY=1 is set do we skip verification,
        and we say so loudly — that switch is a debugging escape hatch, not a default.
        """
        import ssl
        if getattr(self, "_ctx", None) is not None:
            return self._ctx

        if os.environ.get("SIGNWAY_SSL_NOVERIFY") == "1":
            print("[reasoner] WARNING: SSL verification DISABLED (SIGNWAY_SSL_NOVERIFY=1)",
                  flush=True)
            self._ctx = ssl._create_unverified_context()
            return self._ctx

        # explicit override first, then certifi, then the Ubuntu system bundle
        for var in ("GEMINI_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            p = os.environ.get(var)
            if p and os.path.exists(p):
                self._ctx = ssl.create_default_context(cafile=p)
                return self._ctx
        try:
            import certifi
            self._ctx = ssl.create_default_context(cafile=certifi.where())
            return self._ctx
        except Exception:
            pass
        for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
            if os.path.exists(p):
                self._ctx = ssl.create_default_context(cafile=p)
                return self._ctx
        # nothing found — hand back the default and let it fail informatively
        self._ctx = ssl.create_default_context()
        return self._ctx

    def _post(self, body: dict) -> dict:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self.URL.format(model=self.model),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s,
                                        context=self._ssl_context()) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return {"_error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
        except Exception as e:
            return {"_error": f"{e.__class__.__name__}: {e}"}

    def reason(self, request: dict) -> Decision:
        image = request.get("sign_image")
        if image is None:
            image = request.get("image")
        if image is None:
            return Decision(type=DecisionType.STOP, rationale="no image to reason about",
                            confidence=0.0)

        mission = request.get("mission_goal") or "unknown"
        prompt = (f"{SYSTEM_PROMPT}\n\nThe robot is trying to reach: {mission}\n"
                  f"Robot state: {request.get('context', '')}")

        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png",
                                 "data": __import__("base64").b64encode(
                                     _to_png_bytes(image)).decode()}},
            ]}],
            "generationConfig": {"temperature": 0.0,          # a sign has one meaning
                                 "maxOutputTokens": self.max_output_tokens,
                                 "responseMimeType": "application/json"},
        }
        if self.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        t0 = time.time()
        resp = self._post(body)
        dt = time.time() - t0
        self.calls += 1
        self.total_s += dt

        if "_error" in resp:
            # Rate limits and timeouts are expected on a robot, not exceptional. Stop safely.
            dec = Decision(type=DecisionType.STOP, rationale=f"vlm unreachable: {resp['_error']}",
                           confidence=0.0)
        else:
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = ""
            dec = parse_decision(text)

        self.log.append({"latency_s": round(dt, 2), "decision": dec.type.value,
                         "confidence": dec.confidence, "mission": mission,
                         "rationale": dec.rationale[:200]})
        return dec

    def stats(self) -> dict:
        return {"calls": self.calls,
                "mean_latency_s": round(self.total_s / self.calls, 2) if self.calls else 0.0,
                "total_s": round(self.total_s, 1)}