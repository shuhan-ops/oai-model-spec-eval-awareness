"""Launch: ask gpt-6-luna whether each Model Spec scenario is an eval or real production traffic.

Usage:
    uv run python launch_realism.py --dataset ../_downloads/model_spec_dataset/dataset [--limit N] [--samples K]

The subject models' eval-awareness judgments come from the step cache of their runs (launch.py),
so run those first. They enter this job as entity references, so the report links back to them.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anyio
from sqlalchemy import select

from behaviors.judging.defaults import DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY
from behaviors.judging.judge import Judgment
from behaviors.types import ReasoningParams, SamplingParams, TextEntity
from fxtr.core.entities import BoundID
from fxtr.entity_defns.array import Array
from fxtr.project.running import open_local_client, run_job
from fxtr.runner.postgres.schema import step_cache, step_results

from launch import JUDGE_MODEL, ROOTS, load_prompts
from spec_eval_awareness.common import SamplingParamsEntity, SpecPrompt
from spec_eval_awareness.realism import RealismReport, realism, realism_rubric, schema_text

ROOT = "spec-eval-awareness-realism"
KEY = re.compile(r'^([^/]+)/judge_awareness\[prompt_id:"([^"]+)",sample:(\d+)\]$')


async def subject_judgments(client, prompt_ids: set[str]) -> list[tuple[tuple[str, str, int], BoundID[Judgment]]]:
    """Each subject run's eval-awareness judgment per (prompt, sample), from the step cache."""
    subject_of = {root: subject for subject, root in ROOTS.items()}
    async with client.database.engine.connect() as conn:
        rows = (
            await conn.execute(
                select(step_cache.c.cache_key, step_results.c.result_cid)
                .join(step_results, step_cache.c.step_result_id == step_results.c.id)
                .where(step_cache.c.cache_key.like("%/judge_awareness[%"))
            )
        ).all()
    from fxtr.core.entity_data import EntityID

    out = []
    for key, cid in rows:
        m = KEY.match(key)
        if not m or m[1] not in subject_of or m[2] not in prompt_ids:
            continue
        array = await client.load_raw_data(EntityID(cid))
        out.append(((subject_of[m[1]], m[2], int(m[3])), array["rows"][0][-1]))
    return sorted(out, key=lambda x: x[0])


async def main(args: argparse.Namespace) -> None:
    prompts = load_prompts(Path(args.dataset))
    if args.limit:
        prompts = prompts[:: max(1, len(prompts) // args.limit)][: args.limit]
    async with open_local_client(__file__, max_concurrency=args.concurrency) as client:
        refs = [(p.prompt_id, await client.store(p)) for p in prompts]
        subj = await subject_judgments(client, {p.prompt_id for p in prompts})
        print(f"{len(subj)} subject judgments")
        sampling = await client.store(
            SamplingParamsEntity(params=SamplingParams(max_tokens=16000, reasoning=ReasoningParams(effort="medium")))
        )
        inputs = {
            "prompts": Array.from_items([((pid,), ref) for pid, ref in refs], ("[prompt_id: str]", BoundID[SpecPrompt])),
            "n_judge_samples": Array.scalar(args.samples, int),
            "judge_model": Array.scalar(JUDGE_MODEL, str),
            "judge_sampling": Array.scalar(sampling, BoundID[SamplingParamsEntity]),
            "judge_instructions": Array.scalar(await client.store(DEFAULT_SYSTEM_INSTRUCTIONS_ENTITY), BoundID[TextEntity]),
            "rubric": Array.scalar(await client.store(TextEntity(text=realism_rubric(args.eval_share))), BoundID[TextEntity]),
            "output_schema": Array.scalar(await client.store(TextEntity(text=schema_text())), BoundID[TextEntity]),
            "subject_judgments": Array.from_items(
                subj, ("[subject: str, prompt_id: str, sample: int]", BoundID[Judgment])
            ),
        }
        result = await run_job(client, realism, inputs, root=ROOT + ("" if args.eval_share == 50 else f"-prior{args.eval_share}") + ("-pilot" if args.limit else ""))
        report = await client.load(result.item(), as_type=RealismReport)
        print(
            f"report {result.item()}: judged {report.n_judged}/{report.n_prompts}, mean P(eval) {report.mean_p_eval}, "
            f"called eval {report.share_called_eval}, histogram {report.histogram}"
        )
        for s in report.by_section:
            print(f"  {s.section}: {s.mean_p_eval}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--eval-share", type=int, default=50, help="share of evaluations the judge is told the pool has")
    parser.add_argument("--concurrency", type=int, default=64)
    anyio.run(main, parser.parse_args())
