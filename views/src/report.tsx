// The experiment's report (`AwarenessReport` in experiment.py): headline eval-awareness rates, the
// 0-3 level distribution, rates per Model Spec section, and every flagged rollout with the judge's
// quote, linking to the conversation and the judgment.

import {
  EntityDataError, EntityLink, isEntityID, rowsOf, unwrap, useEntity, useInvocation,
  type ArrayData, type EntityData, type EntityID, type EntityPanelProps, type EntityViewProps,
  type InvocationPanelProps, type Spawn,
} from "fxtr-view";
import { DataTable, Kind, Stat, Stats, formatNumber, formatPercent, layout, sequentialFill, type TableColumn } from "fxtr-view-kit";
import styles from "./report.module.css";

export const AWARENESS_REPORT = "local.spec_eval_awareness.AwarenessReport.v1";
export const FLAGGED_ROLLOUT = "local.spec_eval_awareness.FlaggedRollout.v1";
export const EXPERIMENT_WORKFLOW = "spec_eval_awareness.run";

interface Flagged {
  readonly prompt_id: string;
  readonly sample: number;
  readonly level: number;
  readonly location: string;
  readonly affects_behavior: boolean;
  readonly quote: string;
  readonly top_level_section: string;
  readonly judgment: EntityID;
  readonly rollout: EntityID;
  readonly compliance?: number | null;
}

interface SectionStats {
  readonly section: string;
  readonly n_judged: number;
  readonly n_level_ge2: number;
  readonly n_level3: number;
  readonly n_graded?: number;
  readonly n_compliant?: number;
}

interface ByAwareness {
  readonly group: string;
  readonly n_graded: number;
  readonly n_compliant: number;
  readonly mean_score: number | null;
}

interface ReportData {
  readonly subject_model: string;
  readonly judge_model: string;
  readonly n_rollouts: number;
  readonly n_completed: number;
  readonly n_judged: number;
  readonly n_judged_by_fallback?: number;
  readonly unjudged?: readonly string[];
  readonly level_counts: readonly number[];
  readonly rate_level_ge2: number;
  readonly rate_level3: number;
  readonly rate_affects_behavior: number;
  readonly by_section: readonly SectionStats[];
  readonly flagged: readonly Flagged[];
  readonly grader_model?: string;
  readonly n_graded?: number;
  readonly n_ungraded?: number;
  readonly compliance_rate?: number | null;
  readonly compliance_by_awareness?: readonly ByAwareness[];
  readonly paired_n_prompts?: number;
  readonly paired_compliance_aware?: number | null;
  readonly paired_compliance_unaware?: number | null;
}

const LEVELS = ["none", "ambiguous", "implicit", "explicit"] as const;
const short = (model: string) => model.replace(/^openrouter\//, "");

function LevelBars({ counts, total }: { counts: readonly number[]; total: number }) {
  return (
    <div className={styles.levels}>
      {LEVELS.map((name, level) => {
        const n = counts[level] ?? 0;
        return (
          <div key={name} className={styles.levelRow}>
            <span className={styles.levelName}>{level} · {name}</span>
            <span className={styles.track}>
              <span className={styles.bar} data-level={level} style={{ width: `${total ? (100 * n) / total : 0}%`, ...(level > 0 ? { background: sequentialFill(0.3 + 0.23 * level) } : {}) }} />
            </span>
            <span className={layout.num}>{n} <span className={layout.muted}>({formatPercent(total ? n / total : null, 1)})</span></span>
          </div>
        );
      })}
    </div>
  );
}

const SECTION_COLUMNS: readonly TableColumn<SectionStats>[] = [
  { key: "section", label: "spec section", value: (s) => s.section },
  { key: "n", label: "judged", value: (s) => s.n_judged, numeric: true, width: 80 },
  { key: "ge2", label: "eval-aware (≥2)", value: (s) => (s.n_judged ? s.n_level_ge2 / s.n_judged : null), numeric: true,
    cell: (s) => `${formatPercent(s.n_judged ? s.n_level_ge2 / s.n_judged : null, 1)} (${s.n_level_ge2})`, width: 140 },
  { key: "l3", label: "explicit (3)", value: (s) => (s.n_judged ? s.n_level3 / s.n_judged : null), numeric: true,
    cell: (s) => `${formatPercent(s.n_judged ? s.n_level3 / s.n_judged : null, 1)} (${s.n_level3})`, width: 120 },
  { key: "comp", label: "compliant", value: (s) => (s.n_graded ? (s.n_compliant ?? 0) / s.n_graded : null), numeric: true,
    cell: (s) => formatPercent(s.n_graded ? (s.n_compliant ?? 0) / s.n_graded : null, 1), width: 100 },
];

const AWARENESS_COLUMNS: readonly TableColumn<ByAwareness>[] = [
  { key: "group", label: "eval awareness", value: (g) => g.group },
  { key: "n", label: "graded", value: (g) => g.n_graded, numeric: true, width: 80 },
  { key: "rate", label: "compliant (≥6)", value: (g) => (g.n_graded ? g.n_compliant / g.n_graded : null), numeric: true,
    cell: (g) => formatPercent(g.n_graded ? g.n_compliant / g.n_graded : null, 1), width: 130 },
  { key: "mean", label: "mean score (1–7)", value: (g) => g.mean_score, numeric: true,
    cell: (g) => formatNumber(g.mean_score, 2), width: 130 },
];

const FLAGGED_COLUMNS: readonly TableColumn<Flagged>[] = [
  { key: "level", label: "level", value: (f) => f.level, numeric: true, width: 56 },
  { key: "section", label: "section", value: (f) => f.top_level_section, width: 130 },
  { key: "where", label: "where", value: (f) => f.location, width: 84 },
  { key: "affects", label: "drives behavior", value: (f) => (f.affects_behavior ? "yes" : "no"), width: 110 },
  { key: "comp", label: "compliance", value: (f) => f.compliance ?? null, numeric: true, width: 90 },
  { key: "quote", label: "evidence", value: (f) => f.quote, cell: (f) => <span className={styles.quote}>{f.quote}</span> },
  { key: "links", label: "open", width: 190, cell: (f) => (
    <span className={styles.links}><EntityLink id={f.rollout} /> <EntityLink id={f.judgment} /></span>
  ) },
];

export function ReportView({ report }: { report: ReportData }) {
  const judged = report.n_judged;
  return (
    <div className={layout.panel}>
      <p className={layout.subtitle}>
        {short(report.subject_model)} on OpenAI Model Spec Evals · judged by {short(report.judge_model)}
      </p>
      <Stats>
        <Stat label="eval-aware (level ≥2)" value={formatPercent(report.rate_level_ge2, 1)} note={`of ${judged} judged`} />
        <Stat label="explicit (level 3)" value={formatPercent(report.rate_level3, 1)} />
        <Stat label="awareness drives behavior" value={formatPercent(report.rate_affects_behavior, 1)} />
        <Stat label="rollouts" value={report.n_rollouts}
          note={`${report.n_completed} completed · ${report.unjudged?.length ?? 0} unjudged`} />
      </Stats>
      {report.compliance_by_awareness && report.compliance_by_awareness.length > 0 && (
        <>
          <h3 className={layout.title}>Spec compliance by eval awareness</h3>
          <p className={layout.muted}>
            {short(report.grader_model ?? "")} grading against the Model Spec (OpenAI's grader prompt, median of
            samples) · overall {formatPercent(report.compliance_rate ?? null, 1)} compliant of {report.n_graded} graded
            {report.n_ungraded ? `, ${report.n_ungraded} ungraded` : ""}
          </p>
          <DataTable rows={report.compliance_by_awareness} columns={AWARENESS_COLUMNS} rowKey={(g) => g.group} />
          <p className={layout.muted}>
            Within the {report.paired_n_prompts} prompts that have both eval-aware and level-0 samples:
            eval-aware {formatPercent(report.paired_compliance_aware ?? null, 1)} vs not
            {" "}{formatPercent(report.paired_compliance_unaware ?? null, 1)} compliant (each prompt weighted equally).
          </p>
        </>
      )}
      <h3 className={layout.title}>Levels</h3>
      <LevelBars counts={report.level_counts} total={judged} />
      <h3 className={layout.title}>By spec section</h3>
      <DataTable rows={report.by_section} columns={SECTION_COLUMNS} rowKey={(s) => s.section}
        initialSort={{ key: "ge2", direction: "desc" }} />
      <h3 className={layout.title}>Flagged rollouts ({report.flagged.length})</h3>
      <DataTable rows={report.flagged} columns={FLAGGED_COLUMNS} rowKey={(f) => `${f.prompt_id}/${f.sample}`}
        maxHeight={640} emptyText="no rollout reached level 2" />
    </div>
  );
}

function asReport(data: EntityData): ReportData {
  if (data.$type !== AWARENESS_REPORT) throw new EntityDataError(`Expected an AwarenessReport, got ${data.$type}`);
  return data as unknown as ReportData;
}

export function AwarenessReportPanel({ id }: EntityPanelProps) {
  const report = asReport(unwrap(useEntity(id)));
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>awareness report</Kind></div>
      <ReportView report={report} />
    </div>
  );
}

export function AwarenessReportInline({ id }: EntityViewProps) {
  const report = asReport(unwrap(useEntity(id)));
  return <>awareness report · {formatPercent(report.rate_level_ge2, 1)} eval-aware of {report.n_judged}</>;
}

export function FlaggedRolloutInline({ id }: EntityViewProps) {
  const f = unwrap(useEntity(id)) as unknown as Flagged;
  return <>flagged · L{f.level} · {f.top_level_section} · {f.prompt_id.slice(0, 8)}/{f.sample}</>;
}

function reportIdOf(array: ArrayData): EntityID | null {
  const value = rowsOf(array)[0]?.value;
  return array.dims.length === 0 && isEntityID(value) ? value : null;
}

function FromOutput({ output }: { output: EntityID<ArrayData> }) {
  const id = reportIdOf(unwrap(useEntity(output)));
  if (id === null) return <p className={layout.empty}>This run returned no report.</p>;
  const report = asReport(unwrap(useEntity(id)));
  return <ReportView report={report} />;
}

function Overview({ run }: { run: Spawn }) {
  const invocation = unwrap(useInvocation(run.ref));
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>workflow</Kind> <span className={layout.muted}>{invocation.state}</span></div>
      <h2 className={layout.title}>Eval awareness on Model Spec Evals</h2>
      {invocation.error && <p className={layout.muted}>{invocation.error.typeName}: {invocation.error.message}</p>}
      {invocation.output
        ? <FromOutput output={invocation.output} />
        : <p className={layout.empty}>Results appear once every rollout is judged ({invocation.state}).</p>}
    </div>
  );
}

/** The root workflow's results overview. */
export function ExperimentOverview({ invocations }: InvocationPanelProps) {
  const run = invocations[0];
  return run ? <Overview run={run} /> : null;
}
