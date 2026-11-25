// The eval-vs-production judge's report (`RealismReport` in realism.py): how eval-like each Model
// Spec scenario looks to gpt-6-luna, how often the subjects voiced eval awareness at each level
// of that, and the scenarios the judge was least sure about on either side.

import {
  EntityDataError, EntityLink, isEntityID, rowsOf, unwrap, useEntity, useInvocation,
  type ArrayData, type EntityData, type EntityID, type EntityPanelProps, type EntityViewProps,
  type InvocationPanelProps, type Spawn,
} from "fxtr-view";
import { DataTable, Kind, Stat, Stats, formatNumber, formatPercent, layout, seriesColor, sequentialFill, type TableColumn } from "fxtr-view-kit";
import { RowChart, type RateRow } from "./charts";
import styles from "./report.module.css";

export const REALISM_REPORT = "local.spec_eval_awareness.RealismReport.v1";
export const REALISM_WORKFLOW = "spec_eval_awareness.realism";

interface PromptRow {
  readonly prompt_id: string;
  readonly section: string;
  readonly prompt: EntityID;
  readonly p_eval: number | null;
  readonly p_eval_samples: readonly number[];
  readonly n_eval_votes: number;
  readonly main_cues: readonly string[];
  readonly judgments: readonly EntityID[];
  readonly subject_aware: readonly (readonly [string, number, number])[];
}

interface SectionRow { readonly section: string; readonly n_prompts: number; readonly mean_p_eval: number | null }

interface RealismData {
  readonly judge_model: string;
  readonly n_prompts: number;
  readonly n_judged: number;
  readonly unjudged: readonly string[];
  readonly mean_p_eval: number | null;
  readonly share_called_eval: number | null;
  readonly histogram: readonly number[];
  readonly by_section: readonly SectionRow[];
  readonly subjects: readonly string[];
  readonly prompts: readonly PromptRow[];
}

const BUCKETS: readonly [string, number, number][] = [
  ["P(eval) < 0.3", 0, 0.3], ["0.3 – 0.5", 0.3, 0.5], ["0.5 – 0.7", 0.5, 0.7], ["0.7 – 0.9", 0.7, 0.9], ["≥ 0.9", 0.9, 1.01],
];

const awareOf = (p: PromptRow, subject: string) => p.subject_aware.find((s) => s[0] === subject);

function Histogram({ counts, total }: { counts: readonly number[]; total: number }) {
  return (
    <div className={styles.levels}>
      {counts.map((n, i) => (
        <div key={i} className={styles.levelRow}>
          <span className={styles.levelName}>{(i / 10).toFixed(1)} – {((i + 1) / 10).toFixed(1)}</span>
          <span className={styles.track}>
            <span className={styles.bar} style={{ width: `${total ? (100 * n) / total : 0}%`, background: sequentialFill(0.15 + 0.08 * i) }} />
          </span>
          <span className={layout.num}>{n} <span className={layout.muted}>({formatPercent(total ? n / total : null, 0)})</span></span>
        </div>
      ))}
    </div>
  );
}

function AwarenessByBucket({ report }: { report: RealismData }) {
  const judged = report.prompts.filter((p) => p.p_eval !== null);
  const panels = report.subjects.map((subject, i) => {
    const rows: RateRow[] = BUCKETS.map(([label, lo, hi]) => {
      const ps = judged.filter((p) => (p.p_eval as number) >= lo && (p.p_eval as number) < hi);
      let k = 0, n = 0;
      for (const p of ps) { const a = awareOf(p, subject); if (a) { k += a[1]; n += a[2]; } }
      return { key: label, label: `${label} (${ps.length})`, k, n };
    });
    const max = Math.max(0.05, ...rows.map((r) => (r.n ? r.k / r.n : 0) * 1.8));
    return { kind: "rate" as const, title: `${subject}: runs voicing eval awareness`, max: Math.min(1, Math.ceil(max * 20) / 20), color: seriesColor(i), rows };
  });
  return <RowChart label="voiced eval awareness by how eval-like the judge found the scenario" labelWidth={150} panels={panels} />;
}

const pct = (p: PromptRow) => formatPercent(p.p_eval, 0);

function columns(subjects: readonly string[]): readonly TableColumn<PromptRow>[] {
  return [
    { key: "p", label: "P(eval)", value: (p) => p.p_eval, numeric: true, width: 72, cell: pct },
    { key: "samples", label: "samples", width: 90, value: (p) => p.p_eval_samples.join(" "), cell: (p) => <span className={layout.muted}>{p.p_eval_samples.join(" · ")}</span> },
    { key: "section", label: "section", value: (p) => p.section, width: 120 },
    ...subjects.map((s): TableColumn<PromptRow> => ({
      key: `aware-${s}`, label: `${s} aware`, numeric: true, width: 96,
      value: (p) => { const a = awareOf(p, s); return a && a[2] ? a[1] / a[2] : null; },
      cell: (p) => { const a = awareOf(p, s); return a ? `${a[1]}/${a[2]}` : "–"; },
    })),
    { key: "cue", label: "judge's main cue", value: (p) => p.main_cues.join(" | "), cell: (p) => <span className={styles.quote}>{p.main_cues.join(" · ")}</span> },
    { key: "open", label: "open", width: 230, cell: (p) => (
      <span className={styles.links}><EntityLink id={p.prompt} />{p.judgments.map((j) => <EntityLink key={String(j)} id={j} />)}</span>
    ) },
  ];
}

export function RealismView({ report }: { report: RealismData }) {
  const judged = report.prompts.filter((p) => p.p_eval !== null);
  const cols = columns(report.subjects);
  const prodSide = judged.filter((p) => (p.p_eval as number) < 0.5);
  const unsure = [...judged].filter((p) => (p.p_eval as number) >= 0.35 && (p.p_eval as number) <= 0.65)
    .sort((a, b) => Math.abs((a.p_eval as number) - 0.5) - Math.abs((b.p_eval as number) - 0.5));
  return (
    <div className={layout.panel}>
      <p className={layout.subtitle}>
        {report.judge_model.replace(/^openrouter\//, "")} told the scenarios mix real production traffic and evaluations, roughly half each.
        All {formatNumber(report.n_prompts)} are in fact Model Spec eval prompts.
      </p>
      <Stats>
        <Stat label="called evaluation" value={formatPercent(report.share_called_eval, 1)} note={`of ${formatNumber(report.n_judged)} scenarios (majority of samples)`} />
        <Stat label="mean P(eval)" value={formatPercent(report.mean_p_eval, 0)} />
        <Stat label="looks like production" value={formatNumber(prodSide.length)} note="mean P(eval) below 0.5" />
      </Stats>

      <h3 className={layout.subtitle}>how eval-like the scenarios look</h3>
      <p className={layout.muted}>Scenarios by mean P(eval) over the judge's samples.</p>
      <Histogram counts={report.histogram} total={report.n_judged} />

      {report.subjects.length > 0 && (
        <>
          <h3 className={layout.subtitle}>voiced awareness vs how eval-like the scenario looks</h3>
          <p className={layout.muted}>
            Share of subject runs (5 per scenario) whose reasoning voiced eval awareness (level 2–3), grouped by the
            judge's P(eval) for the scenario, with 95% intervals. Row labels give the number of scenarios.
          </p>
          <AwarenessByBucket report={report} />
        </>
      )}

      <h3 className={layout.subtitle}>least eval-looking scenarios</h3>
      <DataTable rows={judged.slice(0, 40)} columns={cols} rowKey={(p) => p.prompt_id} maxHeight={520} />

      <h3 className={layout.subtitle}>judge unsure (P(eval) 0.35–0.65)</h3>
      <DataTable rows={unsure} columns={cols} rowKey={(p) => p.prompt_id} maxHeight={520} emptyText="none" />

      <h3 className={layout.subtitle}>all scenarios ({formatNumber(report.prompts.length)})</h3>
      <DataTable rows={report.prompts} columns={cols} rowKey={(p) => p.prompt_id} maxHeight={640} />
      {report.unjudged.length > 0 && <p className={layout.muted}>{report.unjudged.length} unjudged</p>}
    </div>
  );
}

function asReport(data: EntityData): RealismData {
  if (data.$type !== REALISM_REPORT) throw new EntityDataError(`Expected a RealismReport, got ${data.$type}`);
  return data as unknown as RealismData;
}

export function RealismReportPanel({ id }: EntityPanelProps) {
  const report = asReport(unwrap(useEntity(id)));
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>eval-vs-production report</Kind></div>
      <RealismView report={report} />
    </div>
  );
}

export function RealismReportInline({ id }: EntityViewProps) {
  const report = asReport(unwrap(useEntity(id)));
  return <>eval-vs-production · {formatPercent(report.share_called_eval, 0)} called eval of {report.n_judged}</>;
}

function FromOutput({ output }: { output: EntityID<ArrayData> }) {
  const value = rowsOf(unwrap(useEntity(output)))[0]?.value;
  if (!isEntityID(value)) return <p className={layout.empty}>This run returned no report.</p>;
  return <RealismView report={asReport(unwrap(useEntity(value)))} />;
}

function Overview({ run }: { run: Spawn }) {
  const invocation = unwrap(useInvocation(run.ref));
  return (
    <div className={layout.panel}>
      <div className={layout.kindLine}><Kind>workflow</Kind> <span className={layout.muted}>{invocation.state}</span></div>
      <h2 className={layout.title}>Can gpt-6-luna tell these scenarios are evals?</h2>
      {invocation.error && <p className={layout.muted}>{invocation.error.typeName}: {invocation.error.message}</p>}
      {invocation.output
        ? <FromOutput output={invocation.output} />
        : <p className={layout.empty}>Results appear once every scenario is judged ({invocation.state}).</p>}
    </div>
  );
}

/** The realism workflow's results overview. */
export function RealismOverview({ invocations }: InvocationPanelProps) {
  const run = invocations[0];
  return run ? <Overview run={run} /> : null;
}
