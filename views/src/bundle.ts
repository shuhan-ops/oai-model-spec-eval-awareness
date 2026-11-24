// The bundle's entry: what this project contributes to the viewer.

import { EntityInlineLink, EntityPanel, InvocationPanel, renderer, type ViewBundle } from "fxtr-view";
import { behaviorsRenderers } from "fxtr-view-behaviors";
import {
  AWARENESS_REPORT, AwarenessReportInline, AwarenessReportPanel, EXPERIMENT_WORKFLOW, ExperimentOverview,
  FLAGGED_ROLLOUT, FlaggedRolloutInline,
} from "./report";
import { SPEC_PROMPT, SpecPromptInline, SpecPromptPanel } from "./prompt";
import { COMPLIANCE_GRADE, ComplianceGradeInline, ComplianceGradePanel } from "./grade";

const bundle: ViewBundle = {
  renderers: [
    ...behaviorsRenderers,
    renderer(InvocationPanel, EXPERIMENT_WORKFLOW, "spec-eval-awareness.Overview", ExperimentOverview),
    renderer(EntityPanel, AWARENESS_REPORT, "spec-eval-awareness.Report", AwarenessReportPanel),
    renderer(EntityInlineLink, AWARENESS_REPORT, "spec-eval-awareness.Report", AwarenessReportInline),
    renderer(EntityInlineLink, FLAGGED_ROLLOUT, "spec-eval-awareness.Flagged", FlaggedRolloutInline),
    renderer(EntityPanel, SPEC_PROMPT, "spec-eval-awareness.Prompt", SpecPromptPanel),
    renderer(EntityInlineLink, SPEC_PROMPT, "spec-eval-awareness.Prompt", SpecPromptInline),
    renderer(EntityPanel, COMPLIANCE_GRADE, "spec-eval-awareness.Grade", ComplianceGradePanel),
    renderer(EntityInlineLink, COMPLIANCE_GRADE, "spec-eval-awareness.Grade", ComplianceGradeInline),
  ],
};

export default bundle;
