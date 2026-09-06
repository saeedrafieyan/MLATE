"""
LLM-assisted protocol generation
================================

Turns an optimised formulation into a bench-ready fabrication protocol.

Provider
--------
Requests go to OpenRouter, which exposes many vendors' models behind one
OpenAI-compatible endpoint. The previous release called Google Gemini directly,
which tied the module to one vendor's SDK, one account and one model family. A
single endpoint means a user can reach Anthropic, OpenAI, Google, Meta and open
models with one key, and it means this module does not have to be rewritten
when a vendor changes its client library. The request body is plain JSON over
HTTPS, so there is no provider SDK to depend on at all.

What makes the output trustworthy, or not
-----------------------------------------
A language model asked for a laboratory protocol will produce one whether or not
it has grounds to. Four things here are aimed at that, and all of them are in
the prompt rather than in post-processing, because a fabricated catalogue number
cannot be detected after the fact:

  grounding      the closest real formulations from the corpus are supplied, so
                 the model has published precedent in front of it rather than
                 only its own priors
  uncertainty    predictions are passed with their class probabilities and
                 labelled as predictions, so the protocol cannot present a
                 modelled value as a measured one
  refusal path   the model is told to report what is missing or implausible
                 instead of filling the gap, and is given a section to put it in
  no invention   supplier names, catalogue numbers and citations are forbidden
                 outright, since those are what a model confabulates most
                 readily and what a reader is least able to check

Determinism and record-keeping
------------------------------
Temperature and top-p are low and explicit, and every generation returns the
model id, the parameters, the prompt version and a hash of the exact prompt
sent. A protocol that cannot be traced to the instruction that produced it is
not reproducible, and the referee asked specifically for the model and the
generation parameters to be stated.

The API key is read from the environment or a git-ignored .env file, or passed
in by the caller. It is never written to a source file and never logged.
"""

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
# 6144 was too tight and truncated the output of every verbose model tested,
# losing the Deviations and Cautions section - the one section whose absence is
# least visible and most consequential, since a protocol that simply stops
# looks complete. Every model in the curated list has a context window of at
# least 128k, so the ceiling costs nothing when it is not reached.
DEFAULT_MAX_TOKENS = 16384

# A short, curated list rather than OpenRouter's full catalogue of several
# hundred. These span vendors and price points so that a user without a paid
# account can still generate a protocol.
#
# The catalogue turns over quickly - five of the eleven identifiers written here
# for the previous revision had been withdrawn within weeks, and a withdrawn
# identifier fails at generation time with a message about the model not being
# available. `available_models` therefore checks this list against the live
# catalogue before it is shown, and `fetch_models` returns the catalogue itself
# for a user wanting something not listed. The list is a starting point, not the
# authority.
#
# Tiers: "free" costs nothing and is aggressively rate-limited, so a request may
# be refused and should simply be retried or another model chosen; "cheap" is
# under about $1 per million output tokens; "paid" is the strongest available.
SUGGESTED_MODELS = [
    # Free. `openrouter/free` is OpenRouter's own routed endpoint rather than a
    # single model, which is why it heads the list and is the default: a named
    # free model is shared and frequently refuses under load, and the routed
    # endpoint moves to whichever is answering.
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

    # Under roughly $1 per million output tokens.
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

    # Strongest available. These reserve the full token budget against the
    # account balance before generating, so a nearly empty account is refused
    # here while the cheap models still run.
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

# A free model, so the feature works on a new account with no credit. The
# paid models reserve `max_tokens` against the balance before generating and
# refuse outright when it will not cover them, which is what a user with an
# empty account meets first.
DEFAULT_MODEL = "openrouter/free"


def api_key(explicit: str | None = None) -> str | None:
    """Key from the caller, the environment, or a git-ignored .env file."""
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
    """Describe a key without printing it."""
    if not key:
        return "not set"
    return f"set ({len(key)} chars, ending {key[-4:]})"


# ── prompt assembly ──────────────────────────────────────────────────────────

@dataclass
class Formulation:
    """Everything the prompt needs about one optimised candidate."""
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
    # to_dict, not itertuples: the column names carry units and per-cent signs
    # ("Alginate (%w/v)"), which are not valid Python identifiers, so
    # itertuples silently renames them to _0, _1, _2 and the precedent reaches
    # the model as a list of anonymous numbers.
    lines = []
    for i, row in enumerate(nb.to_dict("records"), 1):
        bits = [f"{k}: {v:g}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in row.items()
                if pd.notna(v) and str(v) not in ("0", "0.0", "")]
        lines.append(f"{i}. " + "; ".join(bits))
    return "\n".join(lines)


def build_prompt(f: Formulation, n_records: int = 2646,
                 user_constraints: str = "") -> tuple[str, str]:
    """Return (system, user) messages from the versioned template."""
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

    # A confident prediction and a hedged one warrant different protocols; the
    # model is told which it is looking at rather than left to assume. Both
    # targets are checked, because they are independent: printability is
    # routinely predicted at above 90% for one class while cell response is
    # split across two, and a note that reported only the first would describe
    # the formulation as well characterised when half of it is not.
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
        # No distribution was supplied, so nothing is known about the spread.
        # Claiming confidence here would be an assertion about a quantity that
        # was never computed.
        confidence = ("Class probabilities were not supplied for these "
                      "predictions, so their confidence is unknown. Do not "
                      "describe any predicted outcome as likely or reliable.")

    extrap = "The proposed formulation lies within the range of the corpus."
    if f.extrapolation and f.extrapolation.get("n_out_of_range"):
        # The observed bounds, not merely the names of the offending
        # parameters. Given names alone a model states the range anyway and
        # invents the numbers: one test run asserted an alginate range of
        # "0.5-3 % w/v" for a corpus whose actual range is 0.25-20. An
        # invented bound in a Deviations section is worse than no bound,
        # because that section is where a reader looks to calibrate trust.
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


# ── generation ───────────────────────────────────────────────────────────────

def fetch_models(key: str | None = None, timeout: int = 20) -> list[dict]:
    """Live OpenRouter catalogue, for users wanting a model not in the list."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                 headers={"Accept": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("data", [])


def available_models(key: str | None = None,
                     timeout: int = 20) -> tuple[list[tuple], bool]:
    """
    The curated list, filtered to what the catalogue currently offers.

    Returns (models, verified). `verified` is False when the catalogue could
    not be reached, in which case the full curated list is returned unfiltered
    and the caller should say that availability is unconfirmed rather than
    imply it was checked.

    This exists because a withdrawn identifier is indistinguishable from a
    working one until a request is made, and the request then fails after the
    user has already run an optimisation. Checking a single catalogue call in
    advance moves that failure to a point where it costs nothing.
    """
    try:
        live = {m["id"] for m in fetch_models(key, timeout)}
    except Exception:
        return list(SUGGESTED_MODELS), False
    if not live:
        return list(SUGGESTED_MODELS), False
    return [m for m in SUGGESTED_MODELS if m[0] in live], True


def _explain_http_error(e, model: str) -> str:
    """
    Turn OpenRouter's response into something a user can act on.

    The raw body is a nested JSON envelope several hundred characters long, and
    the three failures a user actually meets - a rate-limited free model, an
    empty account, a withdrawn identifier - all read the same in it.
    """
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
    """
    Read a server-sent-event stream and return (text, finish_reason, usage,
    model).

    Chunks are handed to `on_chunk` as they arrive, so an interface can render
    the protocol while it is being written. That matters more here than in most
    applications: the free models a user without credit will reach take between
    thirty seconds and two and a half minutes to finish, and a blank screen for
    that long is indistinguishable from a hang.

    Lines beginning with a colon are stream comments used as keep-alives and
    carry no data.
    """
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
    """The same tuple as `_consume_stream`, from one non-streamed response."""
    payload = json.loads(response.read().decode("utf-8"))
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    # A reasoning model can return its answer under "reasoning" with "content"
    # empty, and a truncated or refused completion returns content null as
    # well. Reading the field blindly propagates None into the caller and fails
    # somewhere unrelated, so the case is resolved here.
    text = message.get("content") or message.get("reasoning") or ""
    return (text, choice.get("finish_reason"), payload.get("usage") or {},
            payload.get("model"))


# Failures worth trying again rather than reporting. Free models are shared and
# refuse under load, and a provider occasionally closes a connection having
# sent nothing; both clear on their own within seconds. An insufficient balance
# or a withdrawn identifier never clears, and retrying those only delays the
# explanation.
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
    """
    Generate a protocol and return it with everything needed to reproduce it.

    The returned record carries the model id, all generation parameters, the
    prompt version and a SHA-256 of the exact prompt sent, so that a protocol
    appearing in a paper or a supplement can be traced to the instruction that
    produced it.

    Pass `on_chunk` to stream: it is called with each fragment as it arrives,
    and the complete text is still returned in the record. `on_retry(attempt,
    reason)` is called before each retry, for an interface that would rather
    say what is happening than appear stalled.
    """
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
        # Let a reasoning model think, but return the protocol rather than the
        # thinking. Without this the trace arrives inside the message content
        # and is indistinguishable from the protocol to anyone reading it: one
        # model under test returned its own rule-compliance checklist as part
        # of the document. Providers that do not support the field ignore it.
        "reasoning": {"exclude": True},
    }
    if streaming:
        request_body["stream"] = True
        # Token counts arrive in a final packet rather than with each chunk.
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
            # Once a fragment has reached the caller's screen a retry would
            # append a second protocol to the first, so the attempt is
            # committed from the first chunk onward.
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

        # An empty completion with nothing reported as wrong. Seen
        # intermittently on the free tier and it clears on a retry, so it is
        # treated as a transient fault rather than shown to the user as one.
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
    """
    The closest real formulations in the corpus, for grounding the prompt.

    Distance is computed on the scaled numeric predictors so that a material at
    30 %w/v and a pressure at 300 kPa contribute comparably. Only the non-zero
    components of each neighbour are shown, since a row listing 130 zeros is not
    precedent a reader or a model can use.
    """
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
