"""Shared OpenAI client helper."""
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client built from environment variables."""
    return OpenAI()
