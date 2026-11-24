// A Model Spec Evals prompt (`SpecPrompt` in experiment.py): where it sits in the spec, its
// compliance rubric, and the conversation the subject model was given.

import { unwrap, useEntity, type EntityPanelProps, type EntityViewProps } from "fxtr-view";
import { FieldList, Kind, MessageList, layout } from "fxtr-view-kit";

export const SPEC_PROMPT = "local.spec_eval_awareness.SpecPrompt.v1";

interface Block { readonly type?: string; readonly text?: string; readonly name?: string; readonly input?: unknown }
interface Message { readonly role: string; readonly content: readonly Block[] }
interface PromptData {
  readonly prompt_id: string;
  readonly messages: readonly Message[];
  readonly rubric: string;
  readonly focus_id: string;
  readonly section_id: string;
  readonly top_level_section: string;
  readonly skipped_by_openai_harness: boolean;
}

function blockText(block: Block): string {
  if (typeof block.text === "string") return block.text;
  if (block.name) return `<tool call> ${block.name}(${JSON.stringify(block.input)})`;
  return "";
}

export function SpecPromptPanel({ id }: EntityPanelProps) {
  const p = unwrap(useEntity(id)) as unknown as PromptData;
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>spec prompt</Kind></div>
      <FieldList fields={[
        ["section", `${p.top_level_section} › ${p.section_id}`],
        ["focus", p.focus_id],
        ["prompt id", p.prompt_id],
        ...(p.skipped_by_openai_harness ? [["note", "skipped by OpenAI's harness (system message)"] as const] : []),
      ]} />
      <h3 className={layout.title}>Rubric</h3>
      <p className={layout.prose}>{p.rubric}</p>
      <h3 className={layout.title}>Conversation</h3>
      <MessageList messages={p.messages.map((m) => ({ role: m.role, content: m.content.map(blockText).join("\n") }))} />
    </div>
  );
}

export function SpecPromptInline({ id }: EntityViewProps) {
  const p = unwrap(useEntity(id)) as unknown as PromptData;
  return <>spec prompt · {p.section_id} · {p.prompt_id.slice(0, 8)}</>;
}
