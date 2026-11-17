"""Live model discovery for the candidate picker (candidate-picker spec, Option A).

Lists what each call path can actually reach today -- Bedrock Runtime (foundation models +
inference profiles), Bedrock Mantle (/openai/v1/models) and the LiteLLM proxy -- and keeps a
small registry of models that are NOT in the built-in catalog, so a run (and a later re-test,
after a restart) knows how to call them:

* ``model_id``  -- what Bedrock / the proxy is called with,
* ``endpoint``  -- runtime | mantle | litellm (where it was listed),
* ``family``    -- "openai" (/openai/v1 Chat Completions) or "converse" (every other provider),
* ``efforts``   -- reasoning efforts the model accepted when probed (None = not checked yet).

Catalog models are always matched first, so their curated prices and efforts win. A discovered
model has no price, so its cost is left out of ranking, exactly like any un-priced model.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from .config import DEFAULT_SETTINGS, MIGRATION_MATRIX, MODEL_CATALOG, base_model, estimate_cost

ALL_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]
CACHE_SECONDS = 600
# Bedrock model ids that list as text models but aren't chat models a migration replays against.
_NOT_CHAT = re.compile(r"embed|rerank|sonic|voxtral|titan-image|canvas|reel|stable-|marengo|pegasus|guard", re.I)


class DiscoveredModel(BaseModel):
    name: str                        # candidate name used in runs (unique, no '@')
    model_id: str                    # id sent to Bedrock / the proxy
    endpoint: str                    # runtime | mantle | litellm
    provider: str = ""
    display: str = ""
    family: str = "openai"           # openai | converse
    efforts: Optional[List[str]] = None   # probed; None = unknown
    effort_applies: bool = True


# --------------------------------------------------------------------------- #
# Registry (persisted next to the runs so re-tests survive a restart)
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_REGISTRY: Dict[str, DiscoveredModel] = {}
_LOADED = False


def _registry_path() -> Path:
    from .store import runs_dir
    return runs_dir().parent / "discovered_models.json"


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        data = json.loads(_registry_path().read_text(encoding="utf-8"))
        for d in data.get("models", []):
            m = DiscoveredModel.model_validate(d)
            _REGISTRY[m.name] = m
    except (OSError, ValueError):
        pass


def _save() -> None:
    try:
        p = _registry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"models": [m.model_dump() for m in _REGISTRY.values()]}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # the registry is a convenience; a write failure must never break a request


def get(name: Optional[str]) -> Optional[DiscoveredModel]:
    """The discovered (non-catalog) model behind a candidate name or label, if any."""
    b = base_model(name)
    if not b or b in MODEL_CATALOG:
        return None
    with _LOCK:
        _load()
        return _REGISTRY.get(b)


def register(models: List[DiscoveredModel]) -> None:
    with _LOCK:
        _load()
        for m in models:
            old = _REGISTRY.get(m.name)
            if old and old.efforts is not None and m.efforts is None and old.model_id == m.model_id:
                m = m.model_copy(update={"efforts": old.efforts})  # keep earlier probe results
            _REGISTRY[m.name] = m
        _save()


def set_efforts(name: str, efforts: List[str]) -> None:
    with _LOCK:
        _load()
        m = _REGISTRY.get(name)
        if m:
            _REGISTRY[name] = m.model_copy(update={"efforts": efforts})
            _save()


def supported_efforts(name: Optional[str]) -> Optional[List[str]]:
    """Efforts a discovered model accepts (probed), ["medium"] when effort doesn't apply,
    or None when unknown (not validated -- same as before discovery existed)."""
    m = get(name)
    if m is None:
        return None
    if not m.effort_applies or m.family != "openai":
        return ["medium"]
    return list(m.efforts) if m.efforts else None


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Dict[str, Any]] = {}


def _short_name(model_id: str) -> str:
    """'us.openai.gpt-6-luna' -> 'gpt-6-luna'; 'us.anthropic.claude-x-v1:0' -> 'claude-x-v1:0'."""
    s = re.sub(r"^(us|eu|apac|global|us-gov)\.", "", model_id)
    s = s.split(".", 1)[1] if "." in s.split("-", 1)[0] else s
    return s.replace("@", "-")


def _catalog_by_id(endpoint: str) -> Dict[str, str]:
    """Id the endpoint lists for each catalog model -> catalog name."""
    out: Dict[str, str] = {}
    for name, e in MODEL_CATALOG.items():
        ref = {"mantle": e.mantle_model_id, "litellm": e.litellm_model_group}.get(endpoint, e.bedrock_model_id)
        out[ref] = name
    return out


def _family_for(provider: str, model_id: str) -> str:
    return "openai" if (provider.lower() == "openai" or ".openai." in f".{model_id}") else "converse"


def _list_runtime(region: str) -> List[DiscoveredModel]:
    import boto3  # lazy
    client = boto3.client("bedrock", region_name=region)
    fms = client.list_foundation_models().get("modelSummaries", [])
    profiles = set()
    token = None
    while True:
        kw = {"maxResults": 1000, **({"nextToken": token} if token else {})}
        resp = client.list_inference_profiles(**kw)
        profiles.update(p["inferenceProfileId"] for p in resp.get("inferenceProfileSummaries", []))
        token = resp.get("nextToken")
        if not token:
            break
    geo = region.split("-", 1)[0]          # us-east-1 -> us
    out: List[DiscoveredModel] = []
    for m in fms:
        mid = m.get("modelId", "")
        if (m.get("modelLifecycle") or {}).get("status", "ACTIVE") != "ACTIVE":
            continue
        if "TEXT" not in (m.get("inputModalities") or []) or "TEXT" not in (m.get("outputModalities") or []):
            continue
        if "EMBEDDING" in (m.get("outputModalities") or []) or _NOT_CHAT.search(mid) or _NOT_CHAT.search(
                m.get("modelName", "")):
            continue
        types = m.get("inferenceTypesSupported") or []
        if "ON_DEMAND" in types:
            ref = mid
        elif f"{geo}.{mid}" in profiles:
            ref = f"{geo}.{mid}"
        elif f"global.{mid}" in profiles:
            ref = f"global.{mid}"
        else:
            continue                        # provisioned-only: can't be called on demand
        prov = m.get("providerName", "")
        fam = _family_for(prov, ref)
        out.append(DiscoveredModel(name=_short_name(ref), model_id=ref, endpoint="runtime", provider=prov,
                                   display=m.get("modelName", ""), family=fam,
                                   effort_applies=fam == "openai"))   # Converse sends no effort
    return out


def _list_mantle(region: str, lister: Optional[Callable[[], List[str]]]) -> List[DiscoveredModel]:
    if lister is None:
        from .adapters import make_bedrock_sigv4_model_lister
        lister = make_bedrock_sigv4_model_lister(region, endpoint="mantle")
    out = []
    for mid in lister():
        if _NOT_CHAT.search(mid):
            continue
        prov = mid.split(".", 1)[0] if "." in mid else ""
        # Every Mantle model is called as Chat Completions through the Mantle transport.
        # "-mantle" keeps the name distinct from the same model listed on Runtime: the
        # registry is keyed by name and decides which endpoint a run calls.
        out.append(DiscoveredModel(name=_short_name(mid) + "-mantle", model_id=mid, endpoint="mantle",
                                   provider=_PROVIDERS.get(prov, prov.capitalize()),
                                   display=mid, family="openai"))
    return out


_PROVIDERS = {"openai": "OpenAI", "anthropic": "Anthropic", "deepseek": "DeepSeek", "google": "Google",
              "minimax": "MiniMax", "mistral": "Mistral AI", "moonshotai": "Moonshot AI", "nvidia": "NVIDIA",
              "qwen": "Qwen", "writer": "Writer", "xai": "xAI", "zai": "Z.AI", "meta": "Meta", "amazon": "Amazon"}


def _list_litellm() -> List[DiscoveredModel]:
    from .adapters import discover_litellm_models
    from .config import resolve_litellm_key
    s = DEFAULT_SETTINGS
    if not s.litellm_base:
        raise RuntimeError("no LiteLLM base URL configured")
    out = []
    for alias in discover_litellm_models(s.litellm_base, resolve_litellm_key(s)):
        prov = "Bedrock passthrough" if alias.startswith("bedrock/") else "Proxy model group"
        # Through the proxy every alias is called as Chat Completions; effort support is probed.
        out.append(DiscoveredModel(name=alias.replace("@", "-"), model_id=alias, endpoint="litellm",
                                   provider=prov, display=alias, family="openai"))
    return out


def _uniquify(models: List[DiscoveredModel]) -> List[DiscoveredModel]:
    seen: Dict[str, int] = {}
    out = []
    for m in models:
        n = m.name
        if n in MODEL_CATALOG or n in seen:
            seen[n] = seen.get(n, 0) + 1
            n = f"{m.name}-{m.endpoint}" if seen[n] == 1 else f"{m.name}-{m.endpoint}{seen[n]}"
        seen.setdefault(n, 0)
        out.append(m.model_copy(update={"name": n}))
    return out


def _price(name: str) -> Optional[Dict[str, float]]:
    p = DEFAULT_SETTINGS.price_table.get(name)
    if not p or estimate_cost(name, 1000, 1000) is None:
        return None
    return {"input_per_1m": round(p.input_per_1k * 1000, 4), "output_per_1m": round(p.output_per_1k * 1000, 4)}


def available(endpoint: str, region: str = "us-east-1", legacy_model: Optional[str] = None,
              refresh: bool = False, mantle_lister: Optional[Callable[[], List[str]]] = None,
              runtime_lister: Optional[Callable[[], List[DiscoveredModel]]] = None) -> Dict[str, Any]:
    """Models the given endpoint offers right now, catalog models matched first.

    Never raises: an unreachable endpoint returns the catalog models plus a `reason`."""
    ep = {"bedrock_runtime": "runtime", "bedrock_mantle": "mantle"}.get(endpoint, endpoint)
    if ep not in ("runtime", "mantle", "litellm"):
        raise ValueError(f"unknown endpoint {endpoint!r}")
    key = f"{ep}|{region}"
    hit = _CACHE.get(key)
    if hit and not refresh and time.time() - hit["at"] < CACHE_SECONDS:
        listed, reason, at = hit["listed"], hit["reason"], hit["at"]
    else:
        reason = None
        try:
            if ep == "runtime":
                listed = runtime_lister() if runtime_lister else _list_runtime(region)
            elif ep == "mantle":
                listed = _list_mantle(region, mantle_lister)
            else:
                listed = _list_litellm()
        except Exception as e:  # noqa: BLE001 -- show the catalog and say why the live list is missing
            listed, reason = [], f"could not list models: {str(e)[:200]}"
        at = time.time()
        if not reason:
            _CACHE[key] = {"listed": listed, "reason": None, "at": at}

    by_id = _catalog_by_id(ep)
    suggested = set()
    norm = (legacy_model or "").lower().split("/")[-1]
    for e in MIGRATION_MATRIX:
        if norm and e.legacy_model.lower() == norm:
            suggested = set(e.candidates)
    items: List[Dict[str, Any]] = []
    seen_catalog = set()
    extras: List[DiscoveredModel] = []
    for m in listed:
        cat = by_id.get(m.model_id)
        if cat:
            seen_catalog.add(cat)
            continue
        extras.append(m)
    extras = _uniquify(extras)
    register(extras)
    # Catalog models first. Off-list catalog models still show on Bedrock paths (they're
    # curated and callable); on LiteLLM only what the proxy serves is selectable.
    for name, e in MODEL_CATALOG.items():
        if ep == "litellm" and name not in seen_catalog:
            continue
        items.append({"name": name, "model_id": e.mantle_model_id if ep == "mantle" else e.bedrock_model_id,
                      "provider": "Anthropic" if e.family == "anthropic" else "OpenAI", "display": name,
                      "catalog": True, "listed": name in seen_catalog or not listed,
                      "suggested": name in suggested, "like_for_like": e.like_for_like,
                      "price": _price(name), "efforts": list(e.reasoning_efforts),
                      "effort_applies": e.effort_applies, "efforts_checked": True})
    for m in sorted(extras, key=lambda x: (x.provider.lower(), x.name)):
        reg = get(m.name) or m
        items.append({"name": m.name, "model_id": m.model_id, "provider": m.provider, "display": m.display,
                      "catalog": False, "listed": True, "suggested": False, "like_for_like": False,
                      "price": None,
                      "efforts": (["medium"] if reg.family != "openai" else reg.efforts),
                      "effort_applies": reg.family == "openai",
                      "efforts_checked": reg.family != "openai" or reg.efforts is not None})
    return {"endpoint": ep, "region": region, "count": len(items), "listed_count": len(listed),
            "reason": reason, "fetched_at": at, "models": items}


# --------------------------------------------------------------------------- #
# Effort probe
# --------------------------------------------------------------------------- #
def _rejected(err: str) -> bool:
    e = err.lower()
    return ("400" in e or "validation" in e) and any(k in e for k in (
        "unsupported value", "not supported", "reasoning_effort", "reasoning effort", "invalid value"))


def probe_efforts(name: str, transport: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Send one tiny request per reasoning effort and record which ones the model accepts."""
    m = get(name)
    if m is None:
        raise KeyError(name)
    if m.family != "openai":
        return {"name": name, "efforts": ["medium"], "effort_applies": False, "errors": {}}
    accepted, errors = [], {}
    for eff in ALL_EFFORTS:
        payload = {"model": m.model_id, "input": [{"role": "user", "content": [{"text": "Reply with OK."}]}],
                   "reasoning": {"effort": eff}, "max_completion_tokens": 64}
        try:
            transport(payload)
            accepted.append(eff)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if not _rejected(msg):
                errors[eff] = msg[:200]
    if accepted:
        set_efforts(name, accepted)
    return {"name": name, "efforts": accepted or None, "effort_applies": True, "errors": errors}
