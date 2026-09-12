"""Authenticated serving identity protocol for strict evaluation.

The configured HTTPS server must measure its loaded representation and bind every
completion to a deployment identity. A standard alias-only OpenAI server is blocked.
This is operator trust over authenticated TLS, not hardware attestation.
"""

from __future__ import annotations

import os
import secrets
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from admin.artifacts import StageError, canonical, same_domain
from hermes.evidence_json import evidence_object


def validate_completion_usage(usage: Any, *, max_tokens: int) -> None:
    """Check measured OpenAI usage before normalization, for each response.

    max_tokens is an OUTPUT limit. Prompts, cache subsets and cumulative episode
    usage never consume that per-response allowance.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise StageError("invalid frozen output token limit")
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")
    ):
        raise StageError("serving response lacks measured nonnegative integer token usage")
    if usage["completion_tokens"] > max_tokens:
        raise StageError("serving exceeded frozen output token budget")
    if "total_tokens" in usage and (
        type(usage["total_tokens"]) is not int
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise StageError("serving total token usage contradicts prompt/output counts")
    details = usage.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict) or any(
            k in details and (type(details[k]) is not int or details[k] < 0)
            for k in ("cached_tokens", "cache_write_tokens")
        ):
            raise StageError("serving cache usage must contain nonnegative integer counts")
        if sum(details.get(k, 0) for k in ("cached_tokens", "cache_write_tokens")) > usage["prompt_tokens"]:
            raise StageError("serving cache usage exceeds prompt tokens")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise StageError("trusted serving cannot redirect credentials")


class TrustedServing:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_id: str,
        origin: dict[str, str],
        sampling: dict[str, Any],
        budget: dict[str, Any],
    ):
        required = {"url", "api_key_env", "alias", "deployment_id", "engine", "precision", "device", "environment"}
        if set(config) != required or any(not isinstance(v, str) or not v for v in config.values()):
            raise StageError("incomplete trusted serving configuration")
        parsed = urlparse(config["url"])
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise StageError("trusted serving requires a configured HTTPS endpoint")
        if not os.environ.get(config["api_key_env"]):
            raise StageError("trusted serving credential is absent")
        self.config, self.sampling, self.budget = config, sampling, budget
        self.expected = {
            "schema": "spark-serving-identity-v1",
            "model_id": model_id,
            "representation": "merged-sft",
            "deployment_id": config["deployment_id"],
            **{k: config[k] for k in ("engine", "precision", "device", "environment")},
            "mode": origin["mode"],
            "namespace": origin["namespace"],
        }
        self.observations: list[dict[str, Any]] = []

    def _request(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config["url"].rstrip("/") + route,
            data=canonical(payload),
            headers={
                "Authorization": "Bearer " + os.environ[self.config["api_key_env"]],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
                if response.url != request.full_url:
                    raise StageError("serving identity endpoint redirected")
                return evidence_object(response.read())
        except Exception as exc:
            # Do not include URLs, headers or server response bodies in diagnostics.
            raise StageError("trusted serving request failed or identity is unverifiable") from exc

    def _check(self, record: dict[str, Any], nonce: str) -> None:
        if record != {**self.expected, "nonce": nonce}:
            raise StageError("trusted serving identity differs from exact expected representation/deployment")
        same_domain(self.expected, record)
        self.observations.append(record)

    def attest(self) -> None:
        nonce = secrets.token_hex(24)
        response = self._request("/identity", {"nonce": nonce})
        self._check(response, nonce)

    def completion(
        self,
        seed: int,
        *,
        usage_observer: Callable[[Any], None] | None = None,
        finish_observer: Callable[[Any], None] | None = None,
    ):
        self.attest()

        def complete(messages, *, tools=None):
            nonce = secrets.token_hex(24)
            request = {
                "model": self.config["alias"],
                "messages": messages,
                **self.sampling,
                "max_tokens": self.budget["max_tokens"],
                "seed": seed,
                "spark_identity": {"nonce": nonce, "deployment_id": self.config["deployment_id"]},
            }
            if tools:
                request["tools"] = tools
            response = self._request("/chat/completions", request)
            self._check(response.get("spark_identity", {}), nonce)
            usage = response.get("usage", {})
            if usage_observer is not None:
                usage_observer(usage)
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise StageError("serving response lacks one measured completion")
            choice = choices[0]
            if finish_observer is not None:
                # Preserve missing/unknown status too: partial text that parses as
                # a final answer cannot establish complete provider execution.
                finish_observer(choice.get("finish_reason"))
            validate_completion_usage(usage, max_tokens=self.budget["max_tokens"])
            message = choice["message"]
            raw = dict(usage)
            if message.get("reasoning_content"):
                raw["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                raw["tool_calls"] = [
                    {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                    for c in message["tool_calls"]
                ]
            return message.get("content") or "", raw

        return complete
