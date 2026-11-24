// The overview's charts: compliance by eval-awareness group (rate with a 95% interval, and the
// distribution of 1-7 scores), the within-prompt comparison as one dumbbell per prompt, and eval
// awareness beside compliance per spec section. Drawn as SVG at the frame's measured width.

import { useState, type ReactNode } from "react";
import type { EntityID } from "fxtr-view";
import {
  ChartFrame, ScaleLegend, divergingFill, formatNumber, formatPercent, layout, seriesColor,
  type ChartTooltip,
} from "fxtr-view-kit";
import styles from "./charts.module.css";

const Z = 1.96;

/** The 95% Wilson score interval of k successes in n trials. */
export function wilson(k: number, n: number): [number, number] | null {
  if (n === 0) return null;
  const p = k / n;
  const d = 1 + (Z * Z) / n;
  const centre = (p + (Z * Z) / (2 * n)) / d;
  const half = (Z * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n))) / d;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function ticks(max: number): number[] {
  const step = max > 0.5 ? 0.25 : max > 0.2 ? 0.05 : max > 0.08 ? 0.02 : 0.01;
  const out: number[] = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(Math.round(t * 1000) / 1000);
  return out;
}

const tip = (title: ReactNode, ...lines: ReactNode[]) => (
  <>
    <strong>{title}</strong>
    {lines.map((line, i) => <div key={i}>{line}</div>)}
  </>
);

// ---------------------------------------------------------------------------------------------
// Row charts: one row per category, a label column, then panels sharing the rows.
// ---------------------------------------------------------------------------------------------

export interface RateRow {
  readonly key: string;
  readonly label: string;
  readonly k: number;
  readonly n: number;
  /** Counts of scores 1..7, for a distribution panel. */
  readonly scores?: readonly number[];
  /** Set to draw a rule above this row (a new group starts). */
  readonly ruleAbove?: boolean;
  readonly extra?: ReactNode;
}

type Panel =
  | { readonly kind: "rate"; readonly title: string; readonly max: number; readonly color: string; readonly reference?: number; readonly rows: readonly RateRow[] }
  | { readonly kind: "scores"; readonly title: string; readonly rows: readonly RateRow[] };

const ROW = 24;
const BAR = 12;
const TOP = 18;
const AXIS = 22;
const GAP = 20;

function RowChart({ label, panels, labelWidth = 200 }: { label: string; panels: readonly Panel[]; labelWidth?: number }) {
  const [tooltip, setTooltip] = useState<ChartTooltip | null>(null);
  const rows = panels[0]?.rows ?? [];
  const height = TOP + rows.length * ROW + AXIS;
  return (
    <ChartFrame label={label} tooltip={tooltip} fallbackWidth={720}>
      {(width) => {
        const lw = Math.min(labelWidth, Math.max(110, width * 0.3));
        const pw = Math.max(80, (width - lw - GAP * panels.length) / panels.length);
        const hover = (x: number, y: number, content: ReactNode) => ({
          onPointerEnter: () => setTooltip({ x, y, content }),
          onPointerLeave: () => setTooltip(null),
          onFocus: () => setTooltip({ x, y, content }),
          onBlur: () => setTooltip(null),
        });
        return (
          <svg width={width} height={height} className={styles.svg}>
            {rows.map((row, i) => (
              <g key={row.key}>
                {row.ruleAbove && <line x1={0} x2={width} y1={TOP + i * ROW} y2={TOP + i * ROW} className={styles.rule} />}
                <text x={0} y={TOP + i * ROW + ROW / 2} dominantBaseline="middle" className={styles.rowLabel}>
                  {row.label} <tspan className={styles.dim}>· {formatNumber(row.n)}</tspan>
                </text>
              </g>
            ))}
            {panels.map((panel, p) => {
              const x0 = lw + GAP + p * (pw + GAP);
              return (
                <g key={p} transform={`translate(${x0},0)`}>
                  <text x={0} y={10} className={styles.panelTitle}>{panel.title}</text>
                  {panel.kind === "rate"
                    ? <RatePanel panel={panel} width={pw} x0={x0} hover={hover} />
                    : <ScoresPanel panel={panel} width={pw} x0={x0} hover={hover} />}
                </g>
              );
            })}
          </svg>
        );
      }}
    </ChartFrame>
  );
}

type Hover = (x: number, y: number, content: ReactNode) => object;

function RatePanel({ panel, width, x0, hover }: { panel: Extract<Panel, { kind: "rate" }>; width: number; x0: number; hover: Hover }) {
  const x = (v: number) => (v / panel.max) * width;
  const bottom = TOP + panel.rows.length * ROW;
  return (
    <>
      <g className={layout.grid}>
        {ticks(panel.max).map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={TOP} y2={bottom} />)}
      </g>
      {panel.rows.map((row, i) => {
        const rate = row.n ? row.k / row.n : 0;
        const ci = wilson(row.k, row.n);
        const yc = TOP + i * ROW + ROW / 2;
        const content = tip(
          row.label,
          `${formatPercent(row.n ? rate : null, 1)} (${formatNumber(row.k)} of ${formatNumber(row.n)})`,
          ci ? `95% interval ${formatPercent(ci[0], 1)}–${formatPercent(ci[1], 1)}` : "",
          row.extra ?? "",
        );
        return (
          <g key={row.key} tabIndex={0} className={styles.hit} {...hover(x0 + x(rate), yc - ROW / 2, content)}>
            <rect x={0} y={yc - ROW / 2} width={width} height={ROW} className={styles.hitArea} />
            <rect x={0} y={yc - BAR / 2} width={Math.max(0, x(rate))} height={BAR} rx={2} style={{ fill: panel.color }} />
            {ci && (
              <g className={styles.whisker}>
                <line x1={x(ci[0])} x2={x(ci[1])} y1={yc} y2={yc} />
                <line x1={x(ci[0])} x2={x(ci[0])} y1={yc - 4} y2={yc + 4} />
                <line x1={x(ci[1])} x2={x(ci[1])} y1={yc - 4} y2={yc + 4} />
              </g>
            )}
            <text x={Math.min(width - 2, x(ci ? ci[1] : rate) + 4)} y={yc} dominantBaseline="middle"
              textAnchor={x(ci ? ci[1] : rate) + 40 > width ? "end" : "start"} className={layout.chartLabel}>
              {formatPercent(row.n ? rate : null, 1)}
            </text>
          </g>
        );
      })}
      {panel.reference !== undefined && (
        <line x1={x(panel.reference)} x2={x(panel.reference)} y1={TOP - 2} y2={bottom} className={styles.reference} />
      )}
      <g className={layout.axis} transform={`translate(0,${bottom})`}>
        <line x1={0} x2={width} />
        {ticks(panel.max).map((t) => (
          <g key={t} transform={`translate(${x(t)},0)`}>
            <line y2={4} />
            <text y={14} textAnchor="middle">{formatPercent(t)}</text>
          </g>
        ))}
      </g>
    </>
  );
}

const scoreFill = (s: number) => divergingFill((s - 4) / 3);

function ScoresPanel({ panel, width, x0, hover }: { panel: Extract<Panel, { kind: "scores" }>; width: number; x0: number; hover: Hover }) {
  const bottom = TOP + panel.rows.length * ROW;
  return (
    <>
      {panel.rows.map((row, i) => {
        const counts = row.scores ?? [];
        const total = counts.reduce((a, b) => a + b, 0);
        const yc = TOP + i * ROW + ROW / 2;
        let acc = 0;
        return (
          <g key={row.key}>
            {counts.map((c, s) => {
              const x = total ? (acc / total) * width : 0;
              const w = total ? (c / total) * width : 0;
              acc += c;
              if (c === 0) return null;
              const content = tip(`${row.label}: score ${s + 1}`, `${formatNumber(c)} rollouts (${formatPercent(c / total, 1)})`);
              return (
                <rect key={s} x={x} y={yc - BAR / 2} width={Math.max(0, w - (w > 2 ? 1 : 0))} height={BAR}
                  style={{ fill: scoreFill(s + 1) }} tabIndex={0} className={styles.seg}
                  {...hover(x0 + x + w / 2, yc - ROW / 2, content)} />
              );
            })}
          </g>
        );
      })}
      <g className={layout.axis} transform={`translate(0,${bottom})`}>
        <line x1={0} x2={width} />
        {[0, 0.5, 1].map((t) => (
          <g key={t} transform={`translate(${t * width},0)`}>
            <line y2={4} />
            <text y={14} textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"}>{formatPercent(t)}</text>
          </g>
        ))}
      </g>
    </>
  );
}

// ---------------------------------------------------------------------------------------------
// Compliance by awareness group
// ---------------------------------------------------------------------------------------------

export interface GroupData {
  readonly group: string;
  readonly n_graded: number;
  readonly n_compliant: number;
  readonly mean_score: number | null;
  readonly score_counts?: readonly number[];
}

export function ComplianceByAwarenessChart({ groups, overall }: { groups: readonly GroupData[]; overall: number | null }) {
  const rows: RateRow[] = groups.map((g, i) => ({
    key: g.group,
    label: g.group,
    k: g.n_compliant,
    n: g.n_graded,
    ...(g.score_counts ? { scores: g.score_counts } : {}),
    ruleAbove: i > 0 && g.group.startsWith("level ≥2") && !groups[i - 1]!.group.startsWith("level ≥2"),
    extra: `mean score ${formatNumber(g.mean_score, 2)}`,
  }));
  const hasScores = groups.some((g) => (g.score_counts?.length ?? 0) > 0);
  const panels: Panel[] = [
    { kind: "rate", title: "compliant (median score ≥6)", max: 1, color: seriesColor(0), rows, ...(overall !== null ? { reference: overall } : {}) },
    ...(hasScores ? [{ kind: "scores" as const, title: "median score, 1 to 7", rows }] : []),
  ];
  return (
    <>
      {hasScores && <ScaleLegend scale="diverging" low="1 absolute non-compliance" mid="4" high="7 compliant" />}
      <RowChart label="spec compliance by eval-awareness group" panels={panels} labelWidth={230} />
    </>
  );
}

// ---------------------------------------------------------------------------------------------
// Within-prompt comparison
// ---------------------------------------------------------------------------------------------

export interface PairedData {
  readonly prompt_id: string;
  readonly section: string;
  readonly prompt: EntityID;
  readonly n_aware: number;
  readonly k_aware: number;
  readonly n_unaware: number;
  readonly k_unaware: number;
}

export function pairedSummary(pairs: readonly PairedData[]) {
  const diffs = pairs.map((p) => p.k_aware / p.n_aware - p.k_unaware / p.n_unaware);
  const n = diffs.length;
  if (n === 0) return null;
  const mean = diffs.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(diffs.reduce((a, d) => a + (d - mean) ** 2, 0) / (n - 1)) : 0;
  const aware = pairs.reduce((a, p) => a + p.k_aware / p.n_aware, 0) / n;
  const unaware = pairs.reduce((a, p) => a + p.k_unaware / p.n_unaware, 0) / n;
  const worse = diffs.filter((d) => d < 0).length;
  const better = diffs.filter((d) => d > 0).length;
  return { n, mean, se: sd / Math.sqrt(n), aware, unaware, worse, better, tied: n - worse - better };
}

// ---------------------------------------------------------------------------------------------
// By spec section
// ---------------------------------------------------------------------------------------------

export interface SectionData {
  readonly section: string;
  readonly n_judged: number;
  readonly n_level_ge2: number;
  readonly n_level3: number;
  readonly n_graded?: number;
  readonly n_compliant?: number;
}

export function SectionChart({ sections, overallAware, overallCompliance }: {
  sections: readonly SectionData[]; overallAware: number; overallCompliance: number | null;
}) {
  const sorted = [...sections].sort((a, b) => b.n_level_ge2 / (b.n_judged || 1) - a.n_level_ge2 / (a.n_judged || 1));
  const aware: RateRow[] = sorted.map((s) => ({
    key: s.section, label: s.section, k: s.n_level_ge2, n: s.n_judged,
    extra: `explicit (level 3): ${formatNumber(s.n_level3)}`,
  }));
  const compliant: RateRow[] = sorted.map((s) => ({
    key: s.section, label: s.section, k: s.n_compliant ?? 0, n: s.n_graded ?? 0,
  }));
  const maxAware = Math.max(0.05, ...aware.map((r) => (wilson(r.k, r.n)?.[1] ?? 0)));
  const awareMax = Math.ceil(maxAware * 50) / 50;
  return (
    <RowChart label="eval awareness and compliance by spec section" labelWidth={150} panels={[
      { kind: "rate", title: "eval-aware (level ≥2)", max: awareMax, color: seriesColor(1), rows: aware, reference: overallAware },
      { kind: "rate", title: "compliant (median score ≥6)", max: 1, color: seriesColor(0), rows: compliant,
        ...(overallCompliance !== null ? { reference: overallCompliance } : {}) },
    ]} />
  );
}
