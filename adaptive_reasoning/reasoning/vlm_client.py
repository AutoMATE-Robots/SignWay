"""VLM sign reading: schema, prompt, and pluggable clients.

The reading is GOAL-AGNOSTIC on purpose (see sign_memory: cache the reading,
not the decision). The decision field is still requested so single-call
operation works, but memory stores `content` and re-derives decisions per goal.

Clients:
  * AnthropicVLM      -- Anthropic API (multi-image).
  * OpenAICompatVLM   -- any OpenAI-compatible endpoint, incl. a local vLLM
                         serving Qwen2.5-VL on the A40 (base_url).
  * DryRunVLM         -- no network; returns a configured reading. For pipeline
                         tests and cost-free dry runs of offline_curve.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import List, Optional

import numpy as np

SIGN_TYPES = ("permanent", "temporary", "not_a_sign")
DECISIONS = ("turn_left", "turn_right", "straight", "stop", "not_applicable")

PROMPT_TEMPLATE = """You are the sign-reading module of an indoor navigation robot.
You receive {n} image crop(s) of the SAME sign taken at different distances during
the approach (later crops are usually closer/sharper). The robot's current goal:
"{goal}".

Read the sign and answer with ONLY a JSON object, no other text:
{{
  "sign_type": "permanent" | "temporary" | "not_a_sign",
  "arrows": [{{"direction": "turn_left"|"turn_right"|"straight", "targets": ["<room range or name>", ...]}}],
  "raw_text": "<all text you can read>",
  "decision": "turn_left" | "turn_right" | "straight" | "stop" | "not_applicable",
  "confidence": <0..1>,
  "sufficiency": <0..1, how legible/complete the evidence was>,
  "retry_hint": "<empty, or what would help: e.g. 'retry when closer'>"
}}
"decision" = the action THIS robot should take at the upcoming junction to reach
the goal, based only on the sign. If the sign does not relate to the goal, use
"not_applicable"."""


def parse_reading(text: str) -> dict:
    """Robust parse: strip code fences, find the outermost JSON object, validate."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi <= lo:
        raise ValueError(f"no JSON object in VLM output: {text[:200]!r}")
    obj = json.loads(t[lo:hi + 1])
    obj.setdefault("sign_type", "permanent")
    obj.setdefault("arrows", [])
    obj.setdefault("raw_text", "")
    obj.setdefault("decision", "not_applicable")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("sufficiency", 0.0)
    obj.setdefault("retry_hint", "")
    if obj["decision"] not in DECISIONS:
        obj["decision"] = "not_applicable"
    if obj["sign_type"] not in SIGN_TYPES:
        obj["sign_type"] = "permanent"
    obj["read_conf"] = float(obj.get("confidence", 0.0))
    return obj


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class BaseVLM:
    name = "base"

    def read_sign(self, crops: List[np.ndarray], goal: str) -> dict:
        raise NotImplementedError

    def timed_read(self, crops, goal):
        t0 = time.time()
        out = self.read_sign(crops, goal)
        out["latency_s"] = time.time() - t0
        return out


class DryRunVLM(BaseVLM):
    name = "dry"

    def __init__(self, decision: str = "turn_right", confidence: float = 0.9):
        self.decision, self.confidence = decision, confidence

    def read_sign(self, crops, goal):
        return parse_reading(json.dumps(dict(
            sign_type="permanent",
            arrows=[dict(direction=self.decision, targets=["room 301-320"])],
            raw_text="ROOM 301-320 ->", decision=self.decision,
            confidence=self.confidence, sufficiency=0.9, retry_hint="")))


class AnthropicVLM(BaseVLM):
    def __init__(self, model: str, max_tokens: int = 500):
        import anthropic  # lazy

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens
        self.name = model

    def read_sign(self, crops, goal):
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": _png_b64(c)}} for c in crops]
        content.append({"type": "text",
                        "text": PROMPT_TEMPLATE.format(n=len(crops), goal=goal)})
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}])
        return parse_reading("".join(b.text for b in r.content
                                     if getattr(b, "type", "") == "text"))


class OpenAICompatVLM(BaseVLM):
    """Works with OpenAI API or a local vLLM server (e.g. Qwen2.5-VL on the A40):
    OpenAICompatVLM(model="Qwen/Qwen2.5-VL-7B-Instruct",
                    base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")"""

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: int = 500):
        from openai import OpenAI  # lazy

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.max_tokens = model, max_tokens
        self.name = model

    def read_sign(self, crops, goal):
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_png_b64(c)}"}}
                   for c in crops]
        content.append({"type": "text",
                        "text": PROMPT_TEMPLATE.format(n=len(crops), goal=goal)})
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}])
        return parse_reading(r.choices[0].message.content)


GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_vlm(spec: str) -> BaseVLM:
    """Specs:
      dry | dry:turn_left                     -- no network, pipeline tests
      gemini:<model>                          -- Gemini via its OpenAI-compatible
                                                 endpoint; needs GEMINI_API_KEY
      anthropic:<model>                       -- needs ANTHROPIC_API_KEY
      openai:<model>                          -- OpenAI API; needs OPENAI_API_KEY
      openai:<model>@<base_url>               -- any OpenAI-compatible server
                                                 (e.g. local vLLM Qwen2.5-VL);
                                                 key from VLLM_API_KEY or 'EMPTY'
    """
    import os

    if spec.startswith("dry"):
        parts = spec.split(":")
        return DryRunVLM(parts[1]) if len(parts) > 1 else DryRunVLM()
    kind, rest = spec.split(":", 1)
    if kind == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("gemini:* needs GEMINI_API_KEY set")
        return OpenAICompatVLM(rest, base_url=GEMINI_OPENAI_BASE, api_key=key)
    if kind == "anthropic":
        return AnthropicVLM(rest)
    if kind == "openai":
        model, _, base = rest.partition("@")
        if base:
            key = os.environ.get("VLLM_API_KEY", "EMPTY")
            return OpenAICompatVLM(model, base_url=base, api_key=key)
        return OpenAICompatVLM(model, api_key=os.environ.get("OPENAI_API_KEY"))
    raise ValueError(f"unknown VLM spec {spec!r}")
