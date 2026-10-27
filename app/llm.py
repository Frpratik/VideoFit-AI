"""One place that talks to the LLM for structured outputs."""

import time
from typing import TypeVar

from pydantic import BaseModel

from app import config

T = TypeVar("T", bound=BaseModel)
ATTEMPTS = 2  # the model occasionally answers in plain text instead of calling the output tool
RETRY_DELAY = 3.0  # seconds; gives a short rate limit (HTTP 429) a moment to clear

# Running total of tokens used by this process (handy for judging free-tier budgets).
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def reset_usage() -> None:
    USAGE.update(calls=0, input_tokens=0, output_tokens=0)


def track(handler) -> None:
    """Add the tokens a UsageMetadataCallbackHandler saw to the running total."""
    for usage in handler.usage_metadata.values():
        USAGE["calls"] += 1
        USAGE["input_tokens"] += usage.get("input_tokens", 0)
        USAGE["output_tokens"] += usage.get("output_tokens", 0)


def _invoke_once(schema: type[T], system: str, user: str, temperature: float) -> T:
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_groq import ChatGroq

    handler = UsageMetadataCallbackHandler()
    llm = ChatGroq(model=config.LLM_MODEL, temperature=temperature).with_structured_output(schema)
    try:
        return llm.invoke([("system", system), ("human", user)], config={"callbacks": [handler]})
    finally:
        track(handler)


def call_structured(schema: type[T], system: str, user: str, temperature: float = 0.0) -> T:
    """Ask the model for output matching `schema`. Retries once; raises if both attempts fail."""
    config.require_groq_key()
    for attempt in range(ATTEMPTS):
        try:
            return _invoke_once(schema, system, user, temperature)
        except Exception:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(RETRY_DELAY)
