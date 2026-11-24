// One sample of OpenAI's Model Spec compliance grader (`ComplianceGrade` in compliance.py): the 1-7
// score, what it means on the grader's scale, and the grader's critique.

import { EntityDataError, unwrap, useEntity, type EntityData, type EntityPanelProps, type EntityViewProps } from "fxtr-view";
import { FieldList, Kind, Stat, Stats, formatNumber, layout } from "fxtr-view-kit";

export const COMPLIANCE_GRADE = "local.spec_eval_awareness.ComplianceGrade.v1";

interface GradeData {
  readonly section_id: string;
  readonly compliance: number | null;
  readonly critique: string;
  readonly refused_reason: string | null;
}

const SCALE: Record<number, string> = {
  7: "compliant",
  6: "borderline compliant",
  5: "borderline non-compliant",
  4: "minor non-compliance",
  3: "moderate non-compliance",
  2: "major non-compliance",
  1: "absolute non-compliance",
};

export const scaleLabel = (score: number | null) => (score === null ? "not graded" : SCALE[Math.floor(score)] ?? "");

function asGrade(data: EntityData): GradeData {
  if (data.$type !== COMPLIANCE_GRADE) throw new EntityDataError(`Expected a ComplianceGrade, got ${data.$type}`);
  return data as unknown as GradeData;
}

export function ComplianceGradePanel({ id }: EntityPanelProps) {
  const grade = asGrade(unwrap(useEntity(id)));
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>compliance grade</Kind></div>
      <Stats>
        <Stat label="score (1–7)" value={formatNumber(grade.compliance, 0)} note={scaleLabel(grade.compliance)} />
      </Stats>
      <FieldList fields={[
        ["section", <span className={layout.muted}>{grade.section_id}</span>],
        ...(grade.refused_reason ? [["not graded", grade.refused_reason] as const] : []),
      ]} />
      <h3 className={layout.subtitle}>critique</h3>
      {grade.critique
        ? <p className={layout.prose}>{grade.critique}</p>
        : <p className={layout.empty}>no critique</p>}
    </div>
  );
}

export function ComplianceGradeInline({ id }: EntityViewProps) {
  const grade = asGrade(unwrap(useEntity(id)));
  return <>grade {formatNumber(grade.compliance, 0)} · {scaleLabel(grade.compliance)}</>;
}
