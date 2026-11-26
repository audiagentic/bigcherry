"""The model name a running server answers to: the first id of GET /v1/models.

A managed native start serves its model under the name the YAML gives it and nothing else (the old implicit "default"
alias is gone), so clients that were told no model ask the server which one it serves.
"""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=16)
def served_model(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as response:
        listing = json.loads(response.read())
    models = listing.get("data") or []
    if not models or not isinstance(models[0].get("id"), str) or not models[0]["id"]:
        raise RuntimeError(f"{url}/v1/models lists no model")
    return models[0]["id"]
