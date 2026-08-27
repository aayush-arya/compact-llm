"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

import {
  fetchBenchmark,
  fetchDatasetStats,
  fetchHistory,
  type BenchmarkRow,
  type DatasetStats,
  type HistoryItem,
} from "@/lib/api";
import { buttonVariants } from "@/components/ui/button";
import { PageHeader, SectionLabel, StatCard } from "@/components/page-primitives";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function truncate(s: string, n = 48) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

export default function OverviewPage() {
  const [dataset, setDataset] = React.useState<DatasetStats | null>(null);
  const [benchmark, setBenchmark] = React.useState<BenchmarkRow[] | null>(null);
  const [benchmarkMissing, setBenchmarkMissing] = React.useState(false);
  const [history, setHistory] = React.useState<HistoryItem[] | null>(null);
  const [historyTotal, setHistoryTotal] = React.useState(0);

  React.useEffect(() => {
    fetchDatasetStats().then(setDataset).catch(() => setDataset(null));
    fetchBenchmark()
      .then(setBenchmark)
      .catch(() => setBenchmarkMissing(true));
    fetchHistory(1, 6)
      .then((p) => {
        setHistory(p.items);
        setHistoryTotal(p.total);
      })
      .catch(() => setHistory([]));
  }, []);

  const ft = benchmark?.find((r) => r.approach === "fine_tuned");
  const base = benchmark?.find((r) => r.approach === "base_zero_shot");

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle={today}
        action={
          <Link href="/score" className={cn(buttonVariants({ size: "sm" }))}>
            Score a match <ArrowRight className="ml-1.5 h-4 w-4" />
          </Link>
        }
      />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Dataset"
          value={dataset ? dataset.total.toLocaleString() : <Skeleton className="h-6 w-16" />}
          sub={
            dataset
              ? `${dataset.splits.map((s) => s.count).join(" / ")} split`
              : "loading"
          }
        />
        <StatCard label="Base model" value="gemma-3-4b-it" sub="QLoRA r16 · 4-bit" />
        <StatCard
          label="Benchmark"
          value={
            ft ? ft.pearson.toFixed(3) : benchmarkMissing ? "not run" : <Skeleton className="h-6 w-16" />
          }
          sub={ft ? "Pearson r, fine-tuned" : "run eval after training"}
        />
        <StatCard
          label="Requests served"
          value={history ? historyTotal.toLocaleString() : <Skeleton className="h-6 w-12" />}
          sub="score + compare calls"
        />
      </div>

      <div className="mt-4 rounded-lg border border-border bg-card p-4">
        <SectionLabel>The delta the fine-tune bought</SectionLabel>
        {ft && base ? (
          <div className="grid gap-4 sm:grid-cols-3">
            <Delta name="Pearson r" base={base.pearson} ft={ft.pearson} higherBetter />
            <Delta name="MAE" base={base.mae} ft={ft.mae} higherBetter={false} />
            <Delta
              name="Latency (s)"
              base={base.mean_latency_sec}
              ft={ft.mean_latency_sec}
              higherBetter={false}
            />
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            The adapter hasn&apos;t been trained and evaluated yet. Once{" "}
            <code className="font-metric text-xs">eval_base_vs_finetuned.py</code> runs on the
            held-out test split, the base-vs-fine-tuned deltas land here and on the{" "}
            <Link href="/evaluation" className="text-primary underline-offset-2 hover:underline">
              Evaluation
            </Link>{" "}
            page.
          </p>
        )}
      </div>

      <div className="mt-4 rounded-lg border border-border bg-card p-4">
        <div className="mb-2 flex items-center justify-between">
          <SectionLabel>Recent activity</SectionLabel>
          <Link
            href="/history"
            className="text-[11px] text-muted-foreground hover:text-foreground"
          >
            View all →
          </Link>
        </div>
        {history === null ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        ) : history.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            No requests yet — try the{" "}
            <Link href="/score" className="text-primary hover:underline">
              Score
            </Link>{" "}
            page.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Resume</TableHead>
                  <TableHead>Job Description</TableHead>
                  <TableHead className="text-right">FT</TableHead>
                  <TableHead className="text-right">Base</TableHead>
                  <TableHead className="text-right">When</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="text-xs text-muted-foreground">
                      {truncate(row.resume_text)}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {truncate(row.jd_text)}
                    </TableCell>
                    <TableCell className="font-metric text-right">{row.finetuned_score}</TableCell>
                    <TableCell className="font-metric text-right text-muted-foreground">
                      {row.base_score ?? "—"}
                    </TableCell>
                    <TableCell className="font-metric text-right text-[11px] text-muted-foreground">
                      {new Date(row.created_at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  );
}

function Delta({
  name,
  base,
  ft,
  higherBetter,
}: {
  name: string;
  base: number;
  ft: number;
  higherBetter: boolean;
}) {
  const good = higherBetter ? ft >= base : ft <= base;
  return (
    <div>
      <p className="label-caps">{name}</p>
      <div className="font-metric mt-1 flex items-baseline gap-2">
        <span className="text-xl font-semibold">{ft}</span>
        <span className="text-xs text-muted-foreground line-through decoration-muted-foreground/40">
          {base}
        </span>
      </div>
      <p className={`mt-0.5 text-[11px] ${good ? "text-accent" : "text-destructive"}`}>
        {good ? "improved" : "regressed"} vs base
      </p>
    </div>
  );
}
