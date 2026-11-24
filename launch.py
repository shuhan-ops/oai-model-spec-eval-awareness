"""Launch: Model Spec Evals prompts -> subject model rollouts -> eval-awareness judge.

Usage:
    uv run python launch.py --dataset ../_downloads/model_spec_dataset/dataset [--limit N] [--samples K]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anyio

from behaviors.judging.defaults import DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY
from behaviors.types import ReasoningParams, SamplingParams, TextEntity
from fxtr.core.entities import BoundID
from fxtr.entity_defns.array import Array
from fxtr.project.running import open_local_client, run_job

from spec_eval_awareness.experiment import (
    AwarenessReport,
    SamplingParamsEntity,
    SpecPrompt,
    experiment,
    spec_prompt_from_record,
)
from spec_eval_awareness.rubric import RUBRIC, output_schema

SUBJECT_MODEL = "openrouter/moonshotai/kimi-k3.1-pro"
JUDGE_MODEL = "openrouter/openai/gpt-6-luna"


def load_prompts(dataset_dir: Path) -> list[SpecPrompt]:
    prompts: list[SpecPrompt] = []
    for path in sorted(dataset_dir.glob("*.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            prompts.append(spec_prompt_from_record(record))
    ids = [p.prompt_id for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"
    return sorted(prompts, key=lambda p: p.prompt_id)


async def main(args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.dataset))
    if args.limit:
        # A deterministic, section-spread subset for pilots.
        prompts = prompts[:: max(1, len(prompts) // args.limit)][: args.limit]
    async with open_local_client(__file__, max_concurrency=args.concurrency) as client:
        refs = [(p.prompt_id, await client.store(p)) for p in prompts]
        subject_sampling = await client.store(
            SamplingParamsEntity(
                params=SamplingParams(max_tokens=32000, reasoning=ReasoningParams(effort="high"))
            )
        )
        judge_sampling = await client.store(
            SamplingParamsEntity(
                params=SamplingParams(max_tokens=16000, reasoning=ReasoningParams(effort="medium"))
            )
        )
        inputs = {
            "prompts": Array.from_items(
                [((pid,), ref) for pid, ref in refs], ("[prompt_id: str]", BoundID[SpecPrompt])
            ),
            "n_samples": Array.scalar(args.samples, int),
            "subject_model": Array.scalar(SUBJECT_MODEL, str),
            "subject_sampling": Array.scalar(subject_sampling, BoundID[SamplingParamsEntity]),
            "judge_model": Array.scalar(JUDGE_MODEL, str),
            "judge_sampling": Array.scalar(judge_sampling, BoundID[SamplingParamsEntity]),
            "judge_instructions": Array.scalar(
                await client.store(DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY), BoundID[TextEntity]
            ),
            "rubric": Array.scalar(await client.store(TextEntity(text=RUBRIC)), BoundID[TextEntity]),
            "output_schema": Array.scalar(
                await client.store(TextEntity(text=json.dumps(output_schema()))), BoundID[TextEntity]
            ),
        }
        result = await run_job(client, experiment, inputs, root="spec-eval-awareness-v1")
        print("report id:", result.item())
        report = await client.load(result.item(), as_type=AwarenessReport)
        print(
            f"rollouts={report.n_rollouts} completed={report.n_completed} judged={report.n_judged} (fallback {report.n_judged_by_fallback}) "
            f"levels={report.level_counts} rate>=2={report.rate_level_ge2:.3f} "
            f"rate3={report.rate_level3:.3f} affects={report.rate_affects_behavior:.3f}"
        )
        for s in report.by_section:
            print(f"  {s.section}: {s.n_level_ge2}/{s.n_judged} >=2, {s.n_level3} explicit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=64)
    anyio.run(main, parser.parse_args())
