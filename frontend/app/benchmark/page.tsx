"use client";

import * as React from "react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { fetchBenchmark, type BenchmarkRow } from "@/lib/api";

const APPROACH_LABELS: Record<string, string> = {
  base_zero_shot: "Base (zero-shot)",
  base_few_shot: "Base (few-shot)",
  fine_tuned: "Fine-tuned",
};

const chartConfig = {
  pearson: { label: "Pearson r", color: "var(--chart-1)" },
  spearman: { label: "Spearman ρ", color: "var(--chart-2)" },
  mean_latency_sec: { label: "Latency (s)", color: "var(--chart-3)" },
} satisfies ChartConfig;

export default function BenchmarkPage() {
  const [rows, setRows] = React.useState<BenchmarkRow[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    fetchBenchmark()
      .then(setRows)
      .catch(() => setError("no-data"));
  }, []);

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Benchmark Dashboard</h1>
        <Card>
          <CardHeader>
            <CardTitle>No benchmark results yet</CardTitle>
            <CardDescription>
              Run the Phase 4 eval script after training to populate this page.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="rounded-md bg-muted p-3 text-xs overflow-x-auto">
              cd training{"\n"}python eval_base_vs_finetuned.py
            </pre>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!rows) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const chartData = rows.map((r) => ({
    approach: APPROACH_LABELS[r.approach] ?? r.approach,
    pearson: r.pearson,
    spearman: r.spearman,
    mean_latency_sec: r.mean_latency_sec,
  }));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Benchmark Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Held-out test set: base zero-shot vs base few-shot vs the fine-tuned adapter.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Correlation with human/labeled score</CardTitle>
            <CardDescription>Higher is better</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-64 w-full">
              <BarChart data={chartData}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="approach" tickLine={false} axisLine={false} fontSize={11} />
                <YAxis domain={[0, 1]} tickLine={false} axisLine={false} fontSize={11} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="pearson" fill="var(--color-pearson)" radius={4} />
                <Bar dataKey="spearman" fill="var(--color-spearman)" radius={4} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Mean latency per request</CardTitle>
            <CardDescription>Lower is better — small fine-tuned model vs prompted base</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-64 w-full">
              <BarChart data={chartData}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="approach" tickLine={false} axisLine={false} fontSize={11} />
                <YAxis tickLine={false} axisLine={false} fontSize={11} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="mean_latency_sec" fill="var(--color-mean_latency_sec)" radius={4} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Full results</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Approach</TableHead>
                <TableHead className="text-right">N</TableHead>
                <TableHead className="text-right">Pearson r</TableHead>
                <TableHead className="text-right">Spearman ρ</TableHead>
                <TableHead className="text-right">MAE</TableHead>
                <TableHead className="text-right">Rationale quality (1-5)</TableHead>
                <TableHead className="text-right">Latency (s)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={r.approach}>
                  <TableCell className="font-medium">
                    {APPROACH_LABELS[r.approach] ?? r.approach}
                  </TableCell>
                  <TableCell className="text-right">{r.n}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.pearson}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.spearman}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.mae}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.rationale_quality_1_5}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.mean_latency_sec}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
