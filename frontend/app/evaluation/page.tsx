"use client";

import * as React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import { fetchBenchmark, type BenchmarkRow } from "@/lib/api";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { PageHeader, SectionLabel, StatCard } from "@/components/page-primitives";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const LABELS: Record<string, string> = {
  base_zero_shot: "Base · zero-shot",
  base_few_shot: "Base · few-shot",
  fine_tuned: "Fine-tuned",
};

const chartConfig = {
  pearson: { label: "Pearson r", color: "var(--chart-1)" },
  spearman: { label: "Spearman ρ", color: "var(--chart-3)" },
} satisfies ChartConfig;

function pct(from: number, to: number) {
  if (!from) return "—";
  const d = ((to - from) / Math.abs(from)) * 100;
  return `${d > 0 ? "+" : ""}${d.toFixed(1)}%`;
}

export default function EvaluationPage() {
  const [rows, setRows] = React.useState<BenchmarkRow[] | null>(null);
  const [error, setError] = React.useState(false);

  React.useEffect(() => {
    fetchBenchmark()
      .then(setRows)
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <div>
        <PageHeader title="Evaluation" subtitle="Held-out test set: base vs fine-tuned." />
        <div className="rounded-lg border border-border bg-card p-6">
          <SectionLabel>No benchmark results yet</SectionLabel>
          <p className="text-sm text-muted-foreground">
            Run the Phase 4 eval after training to populate this page. It writes{" "}
            <code className="font-metric text-xs">docs/benchmark_results.json</code>, which this
            endpoint serves.
          </p>
          <pre className="font-metric mt-3 overflow-x-auto rounded-md bg-muted p-3 text-xs">
            cd training{"\n"}python eval_base_vs_finetuned.py --adapter_dir ../outputs/adapter
          </pre>
        </div>
      </div>
    );
  }

  if (!rows) {
    return (
      <div>
        <PageHeader title="Evaluation" subtitle="Held-out test set: base vs fine-tuned." />
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      </div>
    );
  }

  const ft = rows.find((r) => r.approach === "fine_tuned");
  const base = rows.find((r) => r.approach === "base_zero_shot");
  const chartData = rows.map((r) => ({
    approach: LABELS[r.approach] ?? r.approach,
    pearson: r.pearson,
    spearman: r.spearman,
  }));

  return (
    <div>
      <PageHeader
        title="Evaluation"
        subtitle={
          ft && base
            ? `Fine-tuned QLoRA adapter vs base zero-shot, on ${ft.n} held-out test examples.`
            : "Held-out test set: base vs fine-tuned."
        }
      />

      {ft && base && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Pearson r ↑"
            value={ft.pearson.toFixed(3)}
            delta={pct(base.pearson, ft.pearson)}
            deltaGood={ft.pearson >= base.pearson}
            sub={`base ${base.pearson.toFixed(3)}`}
          />
          <StatCard
            label="MAE ↓"
            value={ft.mae.toFixed(1)}
            delta={pct(base.mae, ft.mae)}
            deltaGood={ft.mae <= base.mae}
            sub={`base ${base.mae.toFixed(1)}`}
          />
          <StatCard
            label="Rationale 1–5 ↑"
            value={
              ft.rationale_quality_1_5 !== null ? ft.rationale_quality_1_5.toFixed(2) : "—"
            }
            delta={
              ft.rationale_quality_1_5 !== null && base.rationale_quality_1_5 !== null
                ? pct(base.rationale_quality_1_5, ft.rationale_quality_1_5)
                : undefined
            }
            deltaGood={
              ft.rationale_quality_1_5 !== null && base.rationale_quality_1_5 !== null
                ? ft.rationale_quality_1_5 >= base.rationale_quality_1_5
                : undefined
            }
            sub={
              base.rationale_quality_1_5 !== null
                ? `base ${base.rationale_quality_1_5.toFixed(2)}`
                : "judge off"
            }
          />
          <StatCard
            label="Latency ↓"
            value={`${ft.mean_latency_sec.toFixed(2)}s`}
            delta={pct(base.mean_latency_sec, ft.mean_latency_sec)}
            deltaGood={ft.mean_latency_sec <= base.mean_latency_sec}
            sub={`base ${base.mean_latency_sec.toFixed(2)}s`}
          />
        </div>
      )}

      <div className="mt-4 rounded-lg border border-border bg-card p-4">
        <SectionLabel>Correlation with the labelled score</SectionLabel>
        <ChartContainer config={chartConfig} className="h-56 w-full">
          <BarChart data={chartData}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="approach" tickLine={false} axisLine={false} fontSize={11} />
            <YAxis domain={[0, 1]} tickLine={false} axisLine={false} fontSize={11} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="pearson" fill="var(--color-pearson)" radius={3} />
            <Bar dataKey="spearman" fill="var(--color-spearman)" radius={3} />
          </BarChart>
        </ChartContainer>
      </div>

      <div className="mt-4 rounded-lg border border-border bg-card p-4">
        <SectionLabel>Full results</SectionLabel>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Approach</TableHead>
                <TableHead className="text-right">N</TableHead>
                <TableHead className="text-right">Pearson r</TableHead>
                <TableHead className="text-right">Spearman ρ</TableHead>
                <TableHead className="text-right">MAE</TableHead>
                <TableHead className="text-right">Rationale</TableHead>
                <TableHead className="text-right">Latency</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.approach}>
                  <TableCell className="font-medium">{LABELS[r.approach] ?? r.approach}</TableCell>
                  <TableCell className="font-metric text-right">{r.n}</TableCell>
                  <TableCell className="font-metric text-right">{r.pearson}</TableCell>
                  <TableCell className="font-metric text-right">{r.spearman}</TableCell>
                  <TableCell className="font-metric text-right">{r.mae}</TableCell>
                  <TableCell className="font-metric text-right">
                    {r.rationale_quality_1_5 ?? "—"}
                  </TableCell>
                  <TableCell className="font-metric text-right">{r.mean_latency_sec}s</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}
