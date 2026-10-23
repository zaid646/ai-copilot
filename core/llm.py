"""Provider-agnostic LLM access layer.

Chat providers (`CHAT_PROVIDER`):
  - "openai_compat" (default for remote): any OpenAI-compatible server, e.g.
    vLLM on a Vast.ai GPU serving Qwen3-30B-A3B.
  - "lmstudio": local LM Studio (quota-free).
  - "gemini": Google Gemini via REST (free-tier `AQ.` key uses `X-goog-api-key`).

Embedding providers (`EMBED_PROVIDER`): the same three, 768-dim. Nomic models get
`search_document`/`search_query` task prefixes regardless of provider.

The rest of the codebase depends only on this module's interface:
`chat`, `chat_full`, `chat_json`, `embed_texts`, `get_neo4j_embedder`.
Swapping providers means editing only this file.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core import cache
from core.config import settings
from core.logging import get_logger

log = get_logger("core.llm")

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_MODEL = settings.llm_model
_EMBED_DIM = 768
_EMBED_BATCH = 100

_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _active_chat_model() -> str:
    if settings.chat_provider == "lmstudio":
        return settings.lmstudio_chat_model
    if settings.chat_provider == "openai_compat":
        return settings.openai_chat_model
    return settings.llm_model


def _strip_thinking(text: str) -> str:
    """Remove reasoning traces emitted by thinking models (e.g. Qwen3)."""
    return _THINKING_RE.sub("", text).strip()

# Embeddings can run locally in LM Studio (quota-free) or on Gemini free tier.
# `kind` selects the nomic task prefix: "document" for ingestion, "query" for
# retrieval. Nomic-embed models are trained with these prefixes; using them
# materially improves retrieval. Gemini ignores prefixes.
_NOMIC_PREFIX = {"document": "search_document: ", "query": "search_query: "}


def _active_embed_model() -> str:
    if settings.embed_provider == "lmstudio":
        return settings.lmstudio_embed_model
    if settings.embed_provider == "openai_compat":
        return settings.openai_embed_model
    return settings.embed_model


def _prepare_embed_input(text: str, kind: str) -> str:
    if "nomic" in _active_embed_model().lower():
        return _NOMIC_PREFIX.get(kind, "") + text
    return text


class LLMError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    return {"Content-Type": "application/json", "X-goog-api-key": settings.gemini_api_key}


class RetryableError(LLMError):
    """Transient upstream failures (429/5xx) worth retrying."""


@retry(
    retry=retry_if_exception_type(RetryableError),
    wait=wait_exponential(multiplier=2, min=2, max=40),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = httpx.post(url, headers=_headers(), json=payload, timeout=120.0)
    except httpx.HTTPError as exc:
        raise RetryableError(f"network error: {exc}") from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        raise RetryableError(f"upstream {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise LLMError(f"{resp.status_code}: {resp.text[:400]}")
    return resp.json()


def _extract_text(data: dict[str, Any]) -> tuple[str, dict[str, int]]:
    candidates = data.get("candidates") or []
    if not candidates:
        fb = data.get("promptFeedback", {})
        raise LLMError(f"no candidates returned (promptFeedback={fb})")
    cand = candidates[0]
    finish = cand.get("finishReason", "STOP")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts if p.get("text") and not p.get("thought"))
    usage = data.get("usageMetadata", {}) or {}
    meta = {
        "prompt_tokens": int(usage.get("promptTokenCount", 0)),
        "completion_tokens": int(usage.get("candidatesTokenCount", 0)),
        "total_tokens": int(usage.get("totalTokenCount", 0)),
    }
    if not text and finish not in ("STOP",):
        raise LLMError(f"generation finished without text (finishReason={finish})")
    return text, meta


@retry(
    retry=retry_if_exception_type(RetryableError),
    wait=wait_exponential(multiplier=2, min=1, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _chat_lmstudio(
    system: str,
    user: str,
    temperature: float,
    think: bool,
    seed: int,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> tuple[str, dict[str, int]]:
    """OpenAI-compatible /v1/chat/completions served by LM Studio (local).

    `think=False` appends Qwen's `/no_think` soft switch to the user turn, which
    cuts latency ~3x on the 9B model by suppressing reasoning tokens.
    """
    usr = user if think else f"{user}\n/no_think"
    body: dict[str, Any] = {
        "model": model or settings.lmstudio_chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": usr},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens or settings.lmstudio_chat_max_tokens,
        "seed": seed,
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            },
        }
    try:
        resp = httpx.post(
            f"{base_url or settings.lmstudio_base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.lmstudio_api_key}",
            },
            json=body,
            timeout=600.0,
        )
    except httpx.HTTPError as exc:
        raise RetryableError(f"lmstudio network error: {exc}") from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        raise RetryableError(f"lmstudio {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise LLMError(f"lmstudio {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"lmstudio returned no choices: {data}")
    msg = choices[0].get("message") or {}
    text = _strip_thinking(msg.get("content") or "")
    usage = data.get("usage", {}) or {}
    meta = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return text, meta


@retry(
    retry=retry_if_exception_type(RetryableError),
    wait=wait_exponential(multiplier=2, min=1, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _chat_openai_compat(
    system: str,
    user: str,
    temperature: float,
    think: bool,
    json_mode: bool,
    seed: int,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> tuple[str, dict[str, int]]:
    """OpenAI-compatible /v1/chat/completions (e.g. vLLM on a Vast.ai GPU).

    Thinking is controlled via vLLM's `chat_template_kwargs`; `json_mode` adds
    `response_format={"type": "json_object"}` for guaranteed JSON output.
    """
    if not base_url:
        raise LLMError("OPENAI_CHAT_BASE_URL is not set for openai_compat provider")
    body: dict[str, Any] = {
        "model": model or settings.openai_chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens or settings.openai_chat_max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            },
        }
    elif json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.openai_api_key}",
            },
            json=body,
            timeout=600.0,
        )
    except httpx.HTTPError as exc:
        raise RetryableError(f"openai_compat network error: {exc}") from exc
    if resp.status_code in (429, 500, 502, 503, 504):
        raise RetryableError(f"openai_compat {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
        raise LLMError(f"openai_compat {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"openai_compat returned no choices: {data}")
    msg = choices[0].get("message") or {}
    text = _strip_thinking(msg.get("content") or "")
    usage = data.get("usage", {}) or {}
    meta = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return text, meta


def chat_full(
    system: str,
    user: str,
    temperature: float | None = None,
    namespace: str = "chat",
    think: bool | None = None,
    json_mode: bool = False,
    seed: int | None = None,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return {'text', 'usage'} for a single-turn completion (disk-cached).

    `think` toggles reasoning traces; `json_mode` requests a JSON object from
    OpenAI-compatible servers. Both are part of the cache key.

    `provider`/`base_url` override the configured provider for this call only
    (e.g. voice agent prefers vLLM GPU with LM Studio fallback).
    """
    temp = settings.llm_temperature if temperature is None else temperature
    if think is None:
        disabled = (
            settings.openai_disable_thinking
            if settings.chat_provider == "openai_compat"
            else settings.lmstudio_disable_thinking
        )
        use_think = not disabled
    else:
        use_think = think
    provider = provider or settings.chat_provider
    base_url = base_url or {
        "lmstudio": settings.lmstudio_base_url,
        "openai_compat": settings.openai_chat_base_url,
    }.get(provider)
    model = model or {
        "lmstudio": settings.lmstudio_chat_model,
        "openai_compat": settings.openai_chat_model,
    }.get(provider, _active_chat_model())
    use_seed = settings.llm_seed if seed is None else seed
    schema_key = json.dumps(schema, sort_keys=True, separators=(",", ":")) if schema else ""
    key = cache.key_for(
        model,
        temp,
        system,
        (
            f"{user}\x00think={int(use_think)}\x00json={int(json_mode)}\x00"
            f"seed={use_seed}\x00schema={schema_key}\x00max_tokens={max_tokens or 0}"
            f"\x00provider={provider}\x00base_url={base_url or ""}"
        ),
        namespace,
    )

    if use_cache:
        hit = cache.get(key)
        if hit is not None:
            hit["usage"]["cached"] = True
            return hit

    if provider == "lmstudio":
        text, usage = _chat_lmstudio(
            system,
            user,
            temp,
            use_think,
            use_seed,
            base_url=base_url,
            model=model,
            schema=schema,
            max_tokens=max_tokens,
        )
    elif provider == "openai_compat":
        text, usage = _chat_openai_compat(
            system,
            user,
            temp,
            use_think,
            json_mode,
            use_seed,
            base_url=base_url,
            model=model,
            schema=schema,
            max_tokens=max_tokens,
        )
    else:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": settings.llm_max_output_tokens,
                "seed": use_seed,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        data = _post(f"{_BASE}/models/{_MODEL}:generateContent", payload)
        text, usage = _extract_text(data)
    usage["cached"] = False
    result = {"text": text, "usage": usage}
    if use_cache:
        cache.set(key, model, namespace, result)
    return result


def chat(
    system: str,
    user: str,
    temperature: float | None = None,
    namespace: str = "chat",
    think: bool | None = None,
    seed: int | None = None,
) -> str:
    return chat_full(
        system,
        user,
        temperature=temperature,
        namespace=namespace,
        think=think,
        seed=seed,
    )["text"]


def chat_json(
    system: str,
    user: str,
    temperature: float | None = None,
    think: bool | None = None,
    seed: int | None = None,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    text = chat_full(
        system,
        user + "\n\nRespond with a single valid JSON object and absolutely nothing else.",
        temperature=temperature,
        namespace="json",
        think=think,
        json_mode=True,
        seed=seed,
        model=model,
        schema=schema,
        max_tokens=max_tokens,
        use_cache=use_cache,
        provider=provider,
        base_url=base_url,
    )["text"]
    return _extract_json(text)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return valid JSON: {exc}\nraw: {text[:400]}") from exc


def _embed_gemini(inputs: list[str], model: str, dim: int) -> list[list[float]]:
    reqs = [
        {
            "model": f"models/{model}",
            "content": {"parts": [{"text": t}]},
            "outputDimensionality": dim,
        }
        for t in inputs
    ]
    data = _post(f"{_BASE}/models/{model}:batchEmbedContents", {"requests": reqs})
    embeddings = data.get("embeddings") or []
    if len(embeddings) != len(inputs):
        raise LLMError(f"expected {len(inputs)} embeddings, got {len(embeddings)}")
    return [emb["values"] for emb in embeddings]


def _embed_lmstudio(inputs: list[str], model: str, dim: int) -> list[list[float]]:
    """OpenAI-compatible /v1/embeddings served by LM Studio (local, no quota)."""
    resp = httpx.post(
        f"{settings.lmstudio_base_url}/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.lmstudio_api_key}",
        },
        json={"model": model, "input": inputs},
        timeout=600.0,
    )
    if resp.status_code >= 400:
        raise LLMError(f"lmstudio {resp.status_code}: {resp.text[:300]}")
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    vecs = [d["embedding"] for d in data]
    if len(vecs) != len(inputs):
        raise LLMError(f"lmstudio: expected {len(inputs)} embeddings, got {len(vecs)}")
    return vecs


def _embed_openai_compat(inputs: list[str], model: str, dim: int) -> list[list[float]]:
    """OpenAI-compatible /v1/embeddings (e.g. vLLM serving nomic-embed on Vast.ai)."""
    if not settings.openai_embed_base_url:
        raise LLMError("OPENAI_EMBED_BASE_URL is not set")
    resp = httpx.post(
        f"{settings.openai_embed_base_url}/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        },
        json={"model": model, "input": inputs},
        timeout=600.0,
    )
    if resp.status_code >= 400:
        raise LLMError(f"openai_compat {resp.status_code}: {resp.text[:300]}")
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    vecs = [d["embedding"] for d in data]
    if len(vecs) != len(inputs):
        raise LLMError(f"openai_compat: expected {len(inputs)} embeddings, got {len(vecs)}")
    return vecs


def embed_texts(texts: list[str], dim: int = _EMBED_DIM, kind: str = "document") -> list[list[float]]:
    """Batch-embed texts (768-dim) via the configured provider.

    `kind` is "document" (ingestion) or "query" (retrieval); it selects the
    nomic task prefix when running on LM Studio. Disk-cached per text so
    repeated ingestion/eval runs never re-hit the API.
    """
    if not texts:
        return []
    model = _active_embed_model()
    prepared = [_prepare_embed_input(t, kind) for t in texts]
    keys = [cache.key_for(model, 0.0, "", t, namespace="embed") for t in prepared]
    out: list[list[float] | None] = [None] * len(texts)

    miss_idx: list[int] = []
    for i, k in enumerate(keys):
        hit = cache.get(k)
        if hit is not None:
            out[i] = hit["vector"]
        else:
            miss_idx.append(i)

    if settings.embed_provider == "lmstudio":
        batch = settings.lmstudio_embed_batch
    elif settings.embed_provider == "openai_compat":
        batch = settings.openai_embed_batch
    else:
        batch = _EMBED_BATCH
    for start in range(0, len(miss_idx), batch):
        idxs = miss_idx[start : start + batch]
        chunk = [prepared[i] for i in idxs]
        if settings.embed_provider == "lmstudio":
            vecs = _embed_lmstudio(chunk, model, dim)
        elif settings.embed_provider == "openai_compat":
            vecs = _embed_openai_compat(chunk, model, dim)
        else:
            vecs = _embed_gemini(chunk, model, dim)
        for pos, vec in zip(idxs, vecs):
            out[pos] = vec
            cache.set(keys[pos], model, "embed", {"vector": vec})

    if any(v is None for v in out):  # pragma: no cover - defensive
        raise LLMError("embedding cache/API produced a hole")
    return [v for v in out if v is not None]


def get_neo4j_embedder() -> Any:
    """Neo4j GraphRAG-compatible embedder backed by the REST client (768-dim)."""
    from neo4j_graphrag.embeddings.base import Embedder

    class _RestEmbedder(Embedder):
        def embed_query(self, text: str, **_: Any) -> list[float]:
            return embed_texts([text], kind="query")[0]

    return _RestEmbedder()
