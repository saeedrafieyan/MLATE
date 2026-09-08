from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlate import config as cfg

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSION = "protocol_v2"

DEFAULT_TEMPERATURE = 0.15
DEFAULT_TOP_P = 0.85
DEFAULT_MAX_TOKENS = 16384

SUGGESTED_MODELS = [
    ("openrouter/free", "OpenRouter", "free",
     "Free, auto-routed across providers. The most reliable free option and "
     "the default, because a named free model is often busy. It selects the "
     "model per request, so pick a named one for any protocol reported in a "
     "paper; the generation record states which model actually served each "
     "request either way."),
    ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA", "free",
     "Free. Complete and well-formed in testing; caught a physically "
     "implausible input rather than silently correcting it."),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "NVIDIA", "free",
     "Free, larger, slower. Complete in testing."),
    ("minimax/minimax-m3:free", "MiniMax", "free",
     "Free, very long context. Complete in testing."),
    ("z-ai/glm-5.2:free", "Z.ai", "free",
     "Free. Often rate-limited; retry or choose another."),
    ("google/gemma-4-31b-it:free", "Google", "free",
     "Free, smaller. Often rate-limited."),

    ("qwen/qwen3.7-flash", "Alibaba", "cheap",
     "Inexpensive and thorough; stated the cell-line mismatch between the "
     "candidate and its nearest precedents unprompted."),
    ("deepseek/deepseek-v3.2", "DeepSeek", "cheap",
     "Inexpensive; read the class probabilities correctly in testing."),
    ("z-ai/glm-5.3-flash", "Z.ai", "cheap",
     "Very cheap, long context, slow to respond."),
    ("meta-llama/llama-4-maverick", "Meta", "cheap",
     "Open-weight; useful if provider choice is constrained."),
    ("mistralai/mistral-medium-3-5", "Mistral", "cheap",
     "European provider, if data residency matters."),

    ("anthropic/claude-opus-4.5", "Anthropic", "paid",
     "Strongest reasoning; best adherence to the no-invention rules."),
    ("anthropic/claude-sonnet-5", "Anthropic", "paid",
     "Near-Opus quality, faster and cheaper."),
    ("openai/gpt-5.2", "OpenAI", "paid",
     "Strong general instruction-following."),
    ("google/gemini-3.1-pro-preview", "Google", "paid",
     "Long context; the closest equivalent to the previous release."),
    ("x-ai/grok-4.3", "xAI", "paid",
     "Long context, strong technical writing."),
]

DEFAULT_MODEL = "openrouter/free"


def api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.strip()
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    for candidate in (cfg.ROOT / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def key_status(key: str | None) -> str:
    if not key:
        return "not set"
    return f"set ({len(key)} chars, ending {key[-4:]})"


@dataclass
class Formulation:
    cell_line: str
    biomaterials: dict[str, float] = field(default_factory=dict)
    printing: dict[str, float] = field(default_factory=dict)
    cell_density: float | None = None
    expected_printability: float | None = None
    expected_cell_response: float | None = None
    printability_proba: dict | None = None
    cell_response_proba: dict | None = None
    wssq: float | None = None
    neighbours: pd.DataFrame | None = None
    extrapolation: dict | None = None

    @property
    def is_acellular(self) -> bool:
        return self.cell_line == cfg.ACELLULAR_TOKEN


def _fmt_proba(proba: dict | None) -> str:
    if not proba:
        return ""
    parts = [f"class {k}: {100 * v:.0f}%" for k, v in sorted(proba.items())
             if v >= 0.01]
    return ", ".join(parts)


def _neighbour_block(nb: pd.DataFrame | None) -> str:
    if nb is None or len(nb) == 0:
        return "No sufficiently similar formulation was found in the corpus."
    lines = []
    for i, row in enumerate(nb.to_dict("records"), 1):
        bits = [f"{k}: {v:g}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in row.items()
                if pd.notna(v) and str(v) not in ("0", "0.0", "")]
        lines.append(f"{i}. " + "; ".join(bits))
    return "\n".join(lines)


def build_prompt(f: Formulation, n_records: int = 2646,
                 user_constraints: str = "") -> tuple[str, str]:
    template = (PROMPT_DIR / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
    system = template.split("## SYSTEM", 1)[1].split("## USER", 1)[0]
    system = system.strip().lstrip("-").strip()
    user_tpl = template.split("## USER", 1)[1].strip()

    bio = "\n".join(f"- {k}: {v:.3g}" for k, v in sorted(f.biomaterials.items())
                    if v and v > 0) or "- none specified"
    pp = "\n".join(f"- {k}: {v:.4g}" for k, v in sorted(f.printing.items())
                   if v is not None) or "- none specified"

    preds = []
    if f.expected_printability is not None:
        preds.append(f"- Predicted printability (scale 0-3, higher is better): "
                     f"{f.expected_printability:.2f}")
        if f.printability_proba:
            preds.append(f"  class probabilities: "
                         f"{_fmt_proba(f.printability_proba)}")
    if not f.is_acellular and f.expected_cell_response is not None:
        preds.append(f"- Predicted cell response (scale 1-5, higher is better): "
                     f"{f.expected_cell_response:.2f}")
        if f.cell_response_proba:
            preds.append(f"  class probabilities: "
                         f"{_fmt_proba(f.cell_response_proba)}")
    if f.wssq is not None:
        preds.append(f"- Combined scaffold-quality score (WSSQ): {f.wssq:.1f}%")

    hedged = [name for name, proba in
              (("printability", f.printability_proba),
               ("cell response", f.cell_response_proba))
              if proba and max(proba.values()) < 0.5]
    if hedged:
        which = " and ".join(hedged)
        confidence = (f"The {which} prediction is not confident: no single "
                      f"class exceeds 50% probability. Treat the corresponding "
                      f"parameters as a starting point requiring pilot "
                      f"optimisation, and say so in the protocol.")
    elif f.printability_proba or f.cell_response_proba:
        confidence = ("The predictions above carry the stated class "
                      "probabilities. Where a probability is spread across "
                      "classes, treat the corresponding parameter as requiring "
                      "pilot verification.")
    else:
        confidence = ("Class probabilities were not supplied for these "
                      "predictions, so their confidence is unknown. Do not "
                      "describe any predicted outcome as likely or reliable.")

    extrap = "The proposed formulation lies within the range of the corpus."
    if f.extrapolation and f.extrapolation.get("n_out_of_range"):
        lines = []
        for name, d in list(f.extrapolation["out_of_range"].items())[:8]:
            lines.append(f"- {name}: proposed {d['value']:.4g}, observed range "
                         f"in the corpus {d['observed_min']:.4g} to "
                         f"{d['observed_max']:.4g}")
        listing = "\n".join(lines)
        extrap = (f"{f.extrapolation['n_out_of_range']} parameter(s) lie "
                  f"outside the range observed in the corpus:\n\n{listing}\n\n"
                  f"The models are extrapolating for this candidate. State "
                  f"this explicitly in Deviations and Cautions, quoting these "
                  f"observed ranges and no others. Do not state a corpus range "
                  f"for any parameter not listed above: you have not been "
                  f"given one, and those parameters are within range.")

    user = user_tpl.format(
        mode=("acellular 3D printing (no cells; cell response is not "
              "applicable)" if f.is_acellular else "cell-laden bioprinting"),
        cell_line=("none - acellular" if f.is_acellular else f.cell_line),
        cell_density_line=("" if f.is_acellular else
                           f"Cell density: {f.cell_density:.3g} x10^6 cells/mL"
                           if f.cell_density else "Cell density: not specified"),
        biomaterials=bio,
        printing=pp,
        n_records=f"{n_records:,}",
        predictions="\n".join(preds) or "- none available",
        confidence_note=confidence,
        neighbours=_neighbour_block(f.neighbours),
        extrapolation=extrap,
        step3=("Ink Preparation" if f.is_acellular else "Bioink Preparation"),
        step6=("Post-printing Handling and Storage" if f.is_acellular
               else "Cell Culture and Incubation"),
        user_constraints=(
            f"### Additional constraints from the user\n\n{user_constraints}\n\n"
            f"Integrate these into the protocol. If a constraint conflicts with "
            f"the proposed parameters, follow the constraint and record the "
            f"conflict in Deviations and Cautions."
            if user_constraints.strip() else ""),
    )
    return system, user


def fetch_models(key: str | None = None, timeout: int = 20) -> list[dict]:
    req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                 headers={"Accept": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("data", [])


def available_models(key: str | None = None,
                     timeout: int = 20) -> tuple[list[tuple], bool]:
    try:
        live = {m["id"] for m in fetch_models(key, timeout)}
    except Exception:
        return list(SUGGESTED_MODELS), False
    if not live:
        return list(SUGGESTED_MODELS), False
    return [m for m in SUGGESTED_MODELS if m[0] in live], True


def _explain_http_error(e, model: str) -> str:
    try:
        body = json.loads(e.read().decode("utf-8", errors="replace"))
        detail = (body.get("error") or {}).get("message") or str(body)
        raw = ((body.get("error") or {}).get("metadata") or {}).get("raw")
        if raw:
            detail = f"{detail} ({raw})"
    except Exception:
        detail = "no further detail"

    if e.code == 429:
        return (f"{model} is rate-limited right now. Free models are shared "
                f"and refuse requests under load; wait a minute and retry, or "
                f"choose another model. ({detail})")
    if e.code == 402:
        return (f"{model} requires credit on the OpenRouter account and the "
                f"balance is insufficient. Choose a model marked free, or add "
                f"credit. ({detail})")
    if e.code in (400, 404):
        return (f"{model} was not accepted by OpenRouter - most often because "
                f"the identifier has been withdrawn from the catalogue. Pick "
                f"another model from the list. ({detail})")
    if e.code in (401, 403):
        return f"The OpenRouter API key was rejected. ({detail})"
    return f"OpenRouter returned {e.code}: {detail}"


def _consume_stream(response, on_chunk) -> tuple[str, str, dict, str]:
    parts, finish, usage, served_model = [], None, {}, None
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            packet = json.loads(body)
        except json.JSONDecodeError:
            continue

        served_model = packet.get("model") or served_model
        if packet.get("usage"):
            usage = packet["usage"]
        for choice in packet.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                parts.append(piece)
                if on_chunk is not None:
                    on_chunk(piece)
    return "".join(parts), finish, usage, served_model


def _read_whole(response) -> tuple[str, str, dict, str]:
    payload = json.loads(response.read().decode("utf-8"))
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    text = message.get("content") or message.get("reasoning") or ""
    return (text, choice.get("finish_reason"), payload.get("usage") or {},
            payload.get("model"))


RETRY_CODES = (429, 500, 502, 503, 504)
DEFAULT_RETRIES = 2


def generate(formulation: Formulation, key: str | None = None,
             model: str = DEFAULT_MODEL,
             temperature: float = DEFAULT_TEMPERATURE,
             top_p: float = DEFAULT_TOP_P,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             user_constraints: str = "", n_records: int = 2646,
             timeout: int = 300, on_chunk=None,
             retries: int = DEFAULT_RETRIES, backoff: float = 4.0,
             on_retry=None) -> dict:
    key = api_key(key)
    if not key:
        raise RuntimeError(
            "No OpenRouter API key. Set OPENROUTER_API_KEY in the environment "
            "or a .env file, or pass one in.")

    system, user = build_prompt(formulation, n_records, user_constraints)
    digest = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()
    streaming = on_chunk is not None

    request_body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning": {"exclude": True},
    }
    if streaming:
        request_body["stream"] = True
        request_body["stream_options"] = {"include_usage": True}
    body = json.dumps(request_body).encode("utf-8")

    text = finish = served = None
    usage: dict = {}
    last_error = None

    for attempt in range(retries + 1):
        if attempt:
            if on_retry is not None:
                on_retry(attempt, last_error)
            time.sleep(backoff * attempt)

        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/saeedrafieyan/mlate",
                     "X-Title": "MLATE V3"})
        emitted = False

        def watched(piece):
            nonlocal emitted
            emitted = True
            on_chunk(piece)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if streaming:
                    text, finish, usage, served = _consume_stream(
                        response, watched)
                else:
                    text, finish, usage, served = _read_whole(response)
        except urllib.error.HTTPError as e:
            last_error = _explain_http_error(e, model)
            if e.code in RETRY_CODES and attempt < retries and not emitted:
                continue
            raise RuntimeError(last_error) from None
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = f"Could not reach OpenRouter: {e}"
            if attempt < retries and not emitted:
                continue
            raise RuntimeError(last_error) from None

        if text.strip():
            break

        last_error = (f"{model} returned an empty completion "
                      f"(finish reason: {finish or 'none reported'})")
        if attempt < retries and not emitted:
            continue
        raise RuntimeError(
            f"{last_error}. If the reason is a length limit, raise the token "
            f"budget; otherwise choose another model.")

    return {
        "protocol": text,
        "model": served or model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": digest,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "streamed": streaming,
        "attempts": attempt + 1,
        "truncated": finish == "length",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def nearest_formulations(values: dict, corpus: pd.DataFrame,
                         columns, n: int = 3) -> pd.DataFrame:
    numeric = [c for c in columns.predictors
               if c in corpus.columns
               and pd.api.types.is_numeric_dtype(corpus[c])]
    sub = corpus[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    span = (sub.max() - sub.min()).replace(0, 1.0)
    target = pd.Series({c: float(values.get(c, 0.0)) for c in numeric})
    d = (((sub - target) / span) ** 2).sum(axis=1).pow(0.5)

    keep = d.nsmallest(n).index
    show = corpus.loc[keep].copy()
    cols = [c for c in (list(columns.biomaterials) + list(cfg.CELL_COLS)
                        + list(columns.print_params) + cfg.TARGETS)
            if c in show.columns]
    out = show[cols]
    return out.loc[:, (out != 0).any(axis=0)]
