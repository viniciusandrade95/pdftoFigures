"""Utility client for interacting with HTTP based LLM APIs.

The client centralises the HTTP logic that was previously handled by the
examples and provides a single entrypoint with sensible defaults such as
timeouts, retries and secure API key handling via environment variables.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
from requests import Session
from requests.exceptions import RequestException


class LLMClient:
    """Thin wrapper around an HTTP LLM endpoint with retry support."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str = "LLM_API_KEY",
        default_endpoint: str = "v1/completions",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        session: Optional[Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.default_endpoint = default_endpoint
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_factor = max(0.0, backoff_factor)
        self.session = session or requests.Session()

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Set the '{api_key_env}' environment variable before "
                "creating the client."
            )
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send the supplied prompt to the remote LLM endpoint.

        Parameters
        ----------
        prompt:
            The textual prompt that should be analysed by the LLM.
        model:
            Optional model identifier. The key is added to the JSON payload when
            provided.
        endpoint:
            Override the default API endpoint path.
        extra_payload:
            Additional fields merged into the request payload. This enables
            callers to pass through options such as temperature, max_tokens, etc.
        """

        payload: Dict[str, Any] = {"prompt": prompt}
        if model:
            payload["model"] = model
        if extra_payload:
            payload.update(extra_payload)

        return self._post(endpoint or self.default_endpoint, payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except RequestException as exc:
                if attempt >= self.max_retries:
                    raise
                self._sleep(attempt)

        # The loop either returns or raises; this is defensive programming.
        raise RuntimeError("LLM request retry loop exited unexpectedly")

    def _sleep(self, attempt: int) -> None:
        if self.backoff_factor <= 0:
            return
        delay = self.backoff_factor * (2 ** (attempt - 1))
        time.sleep(delay)

