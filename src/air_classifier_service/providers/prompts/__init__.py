"""Prompt bundle for the LLM tiers.

The package holds a data file, so it must stay importable from a wheel — every
consumer goes through :mod:`air_classifier_service.providers.prompts.loader` rather than
reading ``system.md`` off the filesystem.
"""

from __future__ import annotations

from air_classifier_service.providers.prompts.loader import (
    CLOUD_MIN_CACHEABLE_TOKENS,
    estimated_system_tokens,
    repair_instruction,
    system_prompt,
    system_prompt_version,
    user_content,
)

__all__ = [
    "CLOUD_MIN_CACHEABLE_TOKENS",
    "estimated_system_tokens",
    "repair_instruction",
    "system_prompt",
    "system_prompt_version",
    "user_content",
]
