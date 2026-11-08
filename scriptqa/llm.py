"""One provider-agnostic LLM interface: complete_json(system, prompt, schema) -> dict.

Swap the model with env vars only:
  LLM_PROVIDER = gemini | openai_compat
  LLM_MODEL    = e.g. gemini-3.5-flash-lite, llama-3.3-70b-versatile, gpt-4o-mini ...
  LLM_FALLBACK_MODEL (optional), LLM_BASE_URL + LLM_API_KEY (openai_compat), GEMINI_API_KEY (gemini)

Every response is cached on disk keyed by (provider, model, system, prompt, schema), so re-runs are
free and byte-identical. Every call is logged (tokens, seconds, cache hit) for the cost report.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, cache_dir: Path | None = None):
        load_env()
        self.provider = provider or os.environ.get("LLM_PROVIDER", "gemini")
        self.model = model or os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        self.fallback = os.environ.get("LLM_FALLBACK_MODEL") or None
        self.cache_dir = cache_dir or Path(os.environ.get("LLM_CACHE_DIR") or ROOT / ".cache" / "llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict] = []
        self._client = None

    # ---------------------------------------------------------------- public
    def complete_json(self, system: str, prompt: str, schema: type[BaseModel], tag: str = "") -> dict:
        fail = [t for t in os.environ.get("LLM_FAIL_TAGS", "").split(",") if t]
        if any(tag.startswith(t) for t in fail):  # test switch: simulate an unavailable model for these steps
            raise LLMError(f"{tag}: simulated failure (LLM_FAIL_TAGS)")
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        key = hashlib.sha256(
            json.dumps([self.provider, self.model, system, prompt, schema_json]).encode()
        ).hexdigest()
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            self.calls.append({"tag": tag, "model": data.get("model", self.model), "cache": True,
                               "in_tokens": data.get("in_tokens", 0), "out_tokens": data.get("out_tokens", 0),
                               "seconds": 0.0})
            return data["result"]

        last_err = None
        for model in [self.model] + ([self.fallback] if self.fallback and self.fallback != self.model else []):
            for attempt in range(4):
                t0 = time.time()
                try:
                    text, tin, tout = self._call(model, system, prompt, schema)
                    result = schema.model_validate_json(_strip_fences(text)).model_dump()
                    rec = {"model": model, "in_tokens": tin, "out_tokens": tout, "result": result}
                    cpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                    self.calls.append({"tag": tag, "model": model, "cache": False, "in_tokens": tin,
                                       "out_tokens": tout, "seconds": round(time.time() - t0, 2)})
                    return result
                except ValidationError as e:  # model returned JSON that breaks the schema: retry once more
                    last_err = e
                except LLMError:
                    raise  # configuration problem (missing key, unknown provider): retrying won't help
                except Exception as e:  # rate limit / transient
                    last_err = e
                    msg = str(e)
                    wait = 20 * (attempt + 1) if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower()) else 3 * (attempt + 1)
                    time.sleep(wait)
        raise LLMError(f"{tag}: all attempts failed: {last_err}")

    def usage(self) -> dict:
        live = [c for c in self.calls if not c["cache"]]
        return {
            "provider": self.provider, "model": self.model, "fallback": self.fallback,
            "calls_total": len(self.calls), "calls_live": len(live),
            "calls_cached": len(self.calls) - len(live),
            "in_tokens": sum(c["in_tokens"] or 0 for c in self.calls),
            "out_tokens": sum(c["out_tokens"] or 0 for c in self.calls),
            "live_seconds": round(sum(c["seconds"] for c in live), 1),
            "models_used": sorted({c["model"] for c in self.calls}),
        }

    # ---------------------------------------------------------------- adapters
    def _call(self, model: str, system: str, prompt: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        if self.provider == "gemini":
            return self._gemini(model, system, prompt, schema)
        if self.provider == "openai_compat":
            return self._openai_compat(model, system, prompt, schema)
        raise LLMError(f"unknown LLM_PROVIDER {self.provider}")

    def _gemini(self, model, system, prompt, schema):
        from google import genai
        from google.genai import types
        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY not set (put it in .env)")
            self._client = genai.Client(api_key=key)
        r = self._client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0,
                response_mime_type="application/json", response_schema=schema,
                **({"max_output_tokens": int(os.environ["LLM_MAX_TOKENS"])} if os.environ.get("LLM_MAX_TOKENS") else {}),
            ),
        )
        fr = getattr(r.candidates[0], "finish_reason", None) if r.candidates else None
        if fr is not None and "MAX_TOKENS" in str(fr):
            raise LLMError(f"output truncated at the model's token limit ({os.environ.get('LLM_MAX_TOKENS', 'default')}); "
                           "raise LLM_MAX_TOKENS or split the call")
        u = r.usage_metadata
        return r.text or "", (u.prompt_token_count or 0) if u else 0, (u.candidates_token_count or 0) if u else 0

    def _openai_compat(self, model, system, prompt, schema):
        """Any OpenAI-compatible /chat/completions endpoint (Groq, OpenRouter, Together, vLLM, Ollama,
        Gemini's own OpenAI endpoint, or Kuku's gateway)."""
        import httpx
        base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("LLM_API_KEY", "")
        schema_txt = json.dumps(schema.model_json_schema())
        body = {
            "model": model, "temperature": 0,
            "response_format": {"type": "json_object"},
            **({"max_tokens": int(os.environ["LLM_MAX_TOKENS"])} if os.environ.get("LLM_MAX_TOKENS") else {}),
            "messages": [
                {"role": "system", "content": system + "\n\nReturn ONLY a JSON object matching this JSON Schema:\n" + schema_txt},
                {"role": "user", "content": prompt},
            ],
        }
        r = httpx.post(f"{base}/chat/completions", json=body, timeout=180,
                       headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 400 and "response_format" in r.text:
            body.pop("response_format", None)  # some gateways reject JSON mode: retry once without it
            r = httpx.post(f"{base}/chat/completions", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {key}"})
        if r.status_code in (401, 403, 404):
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")  # configuration: key or model name
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")  # 429 / 5xx: transient, retried with backoff
        j = r.json()
        if j["choices"][0].get("finish_reason") == "length":
            raise LLMError("output truncated at the model's token limit; raise LLM_MAX_TOKENS or split the call")
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()
