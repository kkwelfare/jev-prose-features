"""On-demand advisory Jev prose-feature classification.

Pure parsing/validation helpers live here; the Hermes adapter performs an
explicit call only when its tool is invoked.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import logging
import uuid

_log = logging.getLogger("hermes_plugins.jev_lifecycle.prose")
_log.setLevel(logging.INFO)
if not any(getattr(h, "_jev_lifecycle_console", False) for h in _log.handlers):
    _console = logging.StreamHandler()
    _console.setLevel(logging.INFO)
    _console.setFormatter(logging.Formatter("%(message)s"))
    setattr(_console, "_jev_lifecycle_console", True)
    _log.addHandler(_console)

def _lifecycle(event, request_id, **fields):
    try:
        import re
        safe = {k: v for k, v in fields.items() if k in {"provider", "attempt", "status", "failure", "destination"}}
        for key, value in list(safe.items()):
            if isinstance(value, str): safe[key] = re.sub(r"[^A-Za-z0-9_.:+-]", "_", value)[:96]
        clean_id = re.sub(r"[^A-Za-z0-9_.:+-]", "_", str(request_id))[:80]
        _log.info("jev.lifecycle event=%s route=jev.docs.prose request_id=%s metadata=%s", event, clean_id, json.dumps(safe, sort_keys=True))
    except Exception:
        pass

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_spec = importlib.util.spec_from_file_location("_jev_prose_features_core", Path(__file__).with_name("core.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load Jev prose feature core")
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

PLUGIN_ID = "jev-prose-features"


def _schema() -> dict:
    return {
        "name": "jev_classify_prose_features",
        "description": (
            "Classify whether supplied prose reads naturally to its intended reader in context. Returns the original eight advisory feature labels and confidence unchanged, plus per-segment review_candidates at the provisional 0.5 threshold. Candidates are review topics, not deletion or rewrite suggestions; author_style_impression alone never creates one. No text is rewritten, corrected, submitted, or blocked. Avoid private/sensitive text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": _core.MAX_TEXT_CHARS},
                "segments": {
                    "type": "array", "maxItems": _core.MAX_SEGMENTS,
                    "items": {"type": "object", "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "text": {"type": "string", "maxLength": _core.MAX_SEGMENT_CHARS},
                        "previous_sentence": {"type": "string", "maxLength": _core.MAX_SEGMENT_CHARS},
                        "next_sentence": {"type": "string", "maxLength": _core.MAX_SEGMENT_CHARS},
                    }, "required": ["id", "text"], "additionalProperties": False},
                },
                "purpose": {"type": "string", "maxLength": 240},
                "context": {"type": "string", "maxLength": 500},
                "reader": {"type": "string", "maxLength": 240},
            },
            "additionalProperties": False,
        },
    }


def _runtime_key(env_name):
    value = os.environ.get(env_name, "")
    if value.strip(): return value.strip()
    hermes_home = os.environ.get("HERMES_HOME", "")
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv(hermes_home=hermes_home or None)
    except Exception:
        # Standalone/test hosts may not ship the Hermes helper. Read only the
        # requested key from the standard dotenv path; never log its contents.
        candidates = [Path(hermes_home) / ".env"] if hermes_home else []
        if hermes_home and Path(hermes_home).parent.name == "profiles":
            candidates.append(Path(hermes_home).parent.parent / ".env")
        for path in candidates:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("export "): line = line[7:].lstrip()
                    if not line or line.startswith("#") or "=" not in line: continue
                    name, candidate = line.split("=", 1)
                    if name.strip() != env_name: continue
                    candidate = candidate.strip()
                    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
                        candidate = candidate[1:-1]
                    if candidate.strip(): return candidate.strip()
            except OSError:
                continue
    return os.environ.get(env_name, "").strip()


def _provider_config(ctx):
    get = getattr(ctx, "get_config", lambda _k, d=None: d)
    def cfg(k, d):
        try: return get(k, d)
        except Exception: return d
    configured_name = cfg("primary_provider", None)
    if configured_name is None:
        configured_name = os.environ.get("JEV_PROSE_FEATURES_PROVIDER")
    if configured_name is None:
        # Select one already-authorized Jev decisions route before sending;
        # this is primary selection, not a retry or provider fallback.
        try:
            configured_name = "typesafe" if _runtime_key("TYPESAFE_API_KEY") else None
        except RuntimeError:
            configured_name = None
        if configured_name is None:
            try:
                configured_name = "openrouter" if _runtime_key("OPENROUTER_API_KEY") else "typesafe"
            except RuntimeError:
                configured_name = "typesafe"
    name = str(configured_name).lower()
    defaults = {
        "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13", "OPENROUTER_API_KEY"),
    }
    if name not in defaults: return {"name": name, "endpoint": "", "model": "", "env": ""}, _core.MAX_TIMEOUT_SECONDS
    endpoint, model, env = defaults[name]
    configured_endpoint = cfg("primary_endpoint", None)
    # Existing route settings default to TypeSafe. When only primary_provider
    # is switched, use the matching provider defaults rather than its old URL.
    if configured_endpoint and not (name == "openrouter" and configured_endpoint == defaults["typesafe"][0]):
        endpoint = configured_endpoint
    model = cfg("primary_model", model)
    env = cfg("primary_api_key_env", env)
    spec = {"name": name, "endpoint": endpoint, "model": model, "env": env}
    timeout = cfg("request_timeout_seconds", _core.MAX_TIMEOUT_SECONDS)
    try: timeout = min(float(timeout), _core.MAX_TIMEOUT_SECONDS)
    except (TypeError, ValueError): timeout = _core.MAX_TIMEOUT_SECONDS
    return spec, max(0.5, timeout)


def _post_systemone(spec, body, timeout):
    key = _runtime_key(spec["env"])
    if not key: raise RuntimeError("credential_unavailable")
    request = Request(spec["endpoint"], data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
                               "HTTP-Referer": "https://github.com/NousResearch/hermes-agent", "X-Title": "Hermes Jev Prose Features"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(1_000_000)
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        raise RuntimeError(f"http_error_{exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("network_error") from None
    if not 200 <= int(status) < 300: raise RuntimeError(f"http_error_{status}")
    try: return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError): raise RuntimeError("malformed_response") from None


def _post_openrouter(spec, body, timeout):
    key = _runtime_key(spec["env"])
    if not key: raise RuntimeError("credential_unavailable")
    request = Request(spec["endpoint"], data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
                               "HTTP-Referer": "https://github.com/NousResearch/hermes-agent", "X-Title": "Hermes Jev Prose Features"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(1_000_000)
            status = getattr(response, "status", 200)
    except HTTPError as exc: raise RuntimeError(f"http_error_{exc.code}") from None
    except (URLError, TimeoutError, OSError): raise RuntimeError("network_error") from None
    if not 200 <= int(status) < 300: raise RuntimeError(f"http_error_{status}")
    try: return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError): raise RuntimeError("malformed_response") from None


def _send(ctx, bridge, body, timeout):
    primary, timeout = _provider_config(ctx)
    if primary["name"] == "typesafe": return _post_systemone(primary, body, timeout)
    if primary["name"] == "openrouter": return _post_openrouter(primary, body, timeout)
    raise RuntimeError("unsupported_provider")

class ProseFeatureTool:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, args, **_kwargs):
        try:
            clean = _core.prepare_input(args)
        except _core.InputError as exc:
            return json.dumps({"status": "unknown", "reason": str(exc), "taxonomy_version": _core.TAXONOMY_VERSION,
                               "advisory_only": True, "segments": []}, ensure_ascii=False)
        request_id = None
        try:
            primary, timeout = _provider_config(self.ctx)
            if primary["name"] not in {"typesafe", "openrouter"}:
                raise _core.InputError("unsupported_provider")
            # Calls are sequential, capped at two total, and never retried/fallbacked.
            merged = {"model": "unknown", "features": {}}
            protected = [row for row in clean["segments"] if not row["classification_text"].strip()]
            eligible = [row for row in clean["segments"] if row["classification_text"].strip()]
            if not eligible:
                raise _core.InputError("no_unprotected_prose")
            groups = [eligible[i:i + 4] for i in range(0, len(eligible), 4)]
            for attempt, group in enumerate(groups, 1):
                batch = {**clean, "segments": group}
                request = _core.build_request(batch, model=primary["model"])
                question_map = request.pop("_question_map")
                request_id = "jev-prose-" + uuid.uuid4().hex[:20]
                _lifecycle("provider_start", request_id, provider=primary["name"], attempt=attempt)
                try:
                    raw_response = _send(self.ctx, None, request, timeout)
                    _lifecycle("transport_response", request_id, provider=primary["name"], attempt=attempt, status="returned")
                except Exception as error:
                    _lifecycle("provider_failure", request_id, provider=primary["name"], attempt=attempt, failure=type(error).__name__)
                    raise
                parsed = _core.parse_response(raw_response)
                request["_question_map"] = question_map
                validated = _core.validate_response(batch, request, parsed)
                _lifecycle("validated_response", request_id, status="valid")
                merged["model"] = validated["model"]
                merged["features"].update(validated["features"])
            for row in protected:
                merged["features"][row["id"]] = {}
            result = _core.bind_response(clean, merged)
        except Exception as exc:
            # Error details and provider payloads are intentionally excluded.
            safe_reason = str(exc) if isinstance(exc, RuntimeError) else ""
            if not (safe_reason in {"credential_unavailable", "network_error", "malformed_response", "unsupported_provider"}
                    or safe_reason.startswith("http_error_") and safe_reason[11:].isdigit()):
                safe_reason = ""
            reason = getattr(exc, "reason", None) or (str(exc) if isinstance(exc, _core.InputError) else safe_reason or "provider_unavailable")
            result = _core.unknown_result(clean, reason)
        output = json.dumps(result, ensure_ascii=False, sort_keys=True)
        if request_id is not None:
            _lifecycle("delivered", request_id, destination="jev_classify_prose_features_tool_result", status=result.get("status", "unknown"))
        return output


def register(ctx):
    ctx.register_tool(
        name="jev_classify_prose_features",
        toolset="jev_prose_features",
        schema=_schema(),
        handler=ProseFeatureTool(ctx),
        description="Reader-naturalness oriented, context-aware advisory classification; preserves all eight raw labels/confidence and emits confidence-filtered review topics, never rewrites or blocks.",
    )


__all__ = ["PLUGIN_ID", "ProseFeatureTool", "register"]
