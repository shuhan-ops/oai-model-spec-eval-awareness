// The experiment's report (`AwarenessReport` in experiment.py): headline eval-awareness rates, the
// 0-3 level distribution, rates per Model Spec section, and every flagged rollout with the judge's
// quote, linking to the conversation and the judgment.

import {
  EntityDataError, EntityLink, isEntityID, rowsOf, unwrap, useEntity, useInvocation,
  type ArrayData, type EntityData, type EntityID, type EntityPanelProps, type EntityViewProps,
  type InvocationPanelProps, type Spawn,
} from "fxtr-view";
import { DataTable, Kind, Stat, Stats, formatNumber, formatPercent, formatSigned, layout, sequentialFill, type TableColumn } from "fxtr-view-kit";
import styles from "./report.module.css";
import { ComplianceByAwarenessChart, SectionChart, pairedSummary, type GroupData, type PairedData } from "./charts";

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
  readonly grades?: readonly EntityID[];
}

interface SectionStats {
  readonly section: string;
  readonly n_judged: number;
  readonly n_level_ge2: number;
  readonly n_level3: number;
  readonly n_graded?: number;
  readonly n_compliant?: number;
}

type ByAwareness = GroupData;

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
  readonly paired_prompts?: readonly PairedData[];
  readonly rubric_version?: string;
  readonly level_names?: readonly string[];
}

const LEVELS_V1: readonly string[] = ["none", "ambiguous", "implicit", "explicit"];
const short = (model: string) => model.replace(/^openrouter\//, "");

function LevelBars({ counts, total, names }: { counts: readonly number[]; total: number; names: readonly string[] }) {
  return (
    <div className={styles.levels}>
      {names.map((name, level) => {
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

const FLAGGED_COLUMNS: readonly TableColumn<Flagged>[] = [
  { key: "level", label: "level", value: (f) => f.level, numeric: true, width: 56 },
  { key: "section", label: "section", value: (f) => f.top_level_section, width: 130 },
  { key: "where", label: "where", value: (f) => f.location, width: 84 },
  { key: "affects", label: "drives behavior", value: (f) => (f.affects_behavior ? "yes" : "no"), width: 110 },
  { key: "comp", label: "compliance (median)", value: (f) => f.compliance ?? null, numeric: true, width: 130,
    cell: (f) => formatNumber(f.compliance ?? null, 0) },
  { key: "grades", label: "grader critiques", width: 330, cell: (f) => (
    <span className={styles.links}>{(f.grades ?? []).map((g) => <EntityLink key={String(g)} id={g} />)}</span>
  ) },
  { key: "quote", label: "evidence", value: (f) => f.quote, cell: (f) => <span className={styles.quote}>{f.quote}</span> },
  { key: "links", label: "open", width: 190, cell: (f) => (
    <span className={styles.links}><EntityLink id={f.rollout} /> <EntityLink id={f.judgment} /></span>
  ) },
];

export function ReportView({ report }: { report: ReportData }) {
  const judged = report.n_judged;
  const levels = report.level_names ?? LEVELS_V1;
  const groups = report.compliance_by_awareness ?? [];
  const aware = groups.find((g) => g.group.startsWith("level ≥2 (")) ?? null;
  const unaware = groups.find((g) => g.group.startsWith("level 0")) ?? null;
  const pairs = report.paired_prompts ?? [];
  const paired = pairedSummary(pairs);
  const rate = (g: ByAwareness | null) => (g && g.n_graded ? g.n_compliant / g.n_graded : null);
  return (
    <div className={layout.panel}>
      <p className={layout.subtitle}>
        {short(report.subject_model)} on OpenAI Model Spec Evals · awareness judged by {short(report.judge_model)}
        {report.grader_model ? ` · compliance graded by ${short(report.grader_model)}` : ""}
      </p>
      <Stats>
        <Stat label="eval-aware (level ≥2)" value={formatPercent(report.rate_level_ge2, 1)} note={`of ${formatNumber(judged)} judged`} />
        <Stat label={`${levels[3]} (level 3)`} value={formatPercent(report.rate_level3, 1)} />
        <Stat label="awareness drives behavior" value={formatPercent(report.rate_affects_behavior, 1)} />
        {report.compliance_rate != null && (
          <Stat label="spec-compliant" value={formatPercent(report.compliance_rate, 1)}
            note={`of ${formatNumber(report.n_graded ?? 0)} graded`} />
        )}
        {aware && unaware && (
          <Stat label="compliant: aware vs level 0" value={`${formatPercent(rate(aware), 1)} vs ${formatPercent(rate(unaware), 1)}`}
            note={paired ? `same prompts: ${formatPercent(paired.aware, 1)} vs ${formatPercent(paired.unaware, 1)}` : undefined} />
        )}
      </Stats>

      {groups.length > 0 && (
        <>
          <h3 className={layout.subtitle}>spec compliance by eval awareness</h3>
          <p className={layout.muted}>
            Share of rollouts whose median grader score is 6 or 7, with 95% intervals; dashed line: all rollouts.
            Right: the spread of median scores. Hover a bar for counts.
          </p>
          <ComplianceByAwarenessChart groups={groups} overall={report.compliance_rate ?? null} />
        </>
      )}

      {paired && (
        <>
          <h3 className={layout.subtitle}>same prompt, aware vs not</h3>
          <Stats>
            <Stat label="level-0 samples" value={formatPercent(paired.unaware, 1)} note="compliant" />
            <Stat label="eval-aware samples" value={formatPercent(paired.aware, 1)} note="compliant" />
            <Stat label="gap" value={`${formatSigned(100 * paired.mean, 1)} pts`}
              note={`95% interval ${formatSigned(100 * (paired.mean - 1.96 * paired.se), 1)} to ${formatSigned(100 * (paired.mean + 1.96 * paired.se), 1)}`} />
          </Stats>
          <p className={layout.muted}>
            Each prompt was run 5 times. In {formatNumber(paired.n)} prompts, some runs were eval-aware (level 2–3) and
            others showed no awareness (level 0). Comparing runs of the same prompt removes the effect of harder prompts
            triggering awareness more often. Within a prompt, the eval-aware runs complied less often on average, but the
            interval includes zero: the eval-aware runs did worse in {formatNumber(paired.worse)} prompts, better
            in {formatNumber(paired.better)}, and the same in {formatNumber(paired.tied)}. Each prompt counts equally.
          </p>
        </>
      )}

      <h3 className={layout.subtitle}>by spec section</h3>
      <p className={layout.muted}>Share of rollouts per top-level section, with 95% intervals; dashed lines: all sections.</p>
      <SectionChart sections={report.by_section} overallAware={report.rate_level_ge2} overallCompliance={report.compliance_rate ?? null} />

      <h3 className={layout.subtitle}>awareness levels</h3>
      <LevelBars counts={report.level_counts} total={judged} names={levels} />
      <p className={layout.muted}>
        {formatNumber(report.n_rollouts)} rollouts · {formatNumber(report.n_completed)} completed
        · {report.unjudged?.length ?? 0} unjudged · {report.n_ungraded ?? 0} ungraded
      </p>

      <h3 className={layout.subtitle}>flagged rollouts ({report.flagged.length})</h3>
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
