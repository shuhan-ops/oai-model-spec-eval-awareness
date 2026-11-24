"""Types and helpers shared by the experiment's modules."""

from __future__ import annotations

from dataclasses import dataclass

from behaviors.implementations.models.openrouter import OpenRouterAPI
from behaviors.types import (
    AssistantMessage,
    SamplingParams,
    SystemMessage,
    ToolMessage,
    ToolParam,
    UserMessage,
)
from fxtr.entity_defns.dataclass_entity import DataclassEntity

ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


@dataclass(frozen=True)
class SamplingParamsEntity(DataclassEntity, fxtr_type="local.spec_eval_awareness.SamplingParams.v1"):
    """Sampling settings stored as an entity, so a step can take them as an input."""

    params: SamplingParams


@dataclass(frozen=True)
class SpecPrompt(DataclassEntity, fxtr_type="local.spec_eval_awareness.SpecPrompt.v1"):
    """One prompt from openai/model_spec_dataset, with its rubric and Model Spec location."""

    prompt_id: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolParam, ...]
    rubric: str
    focus_id: str
    section_id: str
    top_level_section: str
    sections: tuple[str, ...]
    skipped_by_openai_harness: bool



def make_api(model_name: str) -> OpenRouterAPI:
    """The adapter for a model named `openrouter/VENDOR/MODEL`. The caller closes it."""
    provider, _, model_id = model_name.partition("/")
    if provider == "openrouter":
        return OpenRouterAPI(model_id)
    raise ValueError(f"Name a model as openrouter/VENDOR/MODEL, not {model_name!r}")
