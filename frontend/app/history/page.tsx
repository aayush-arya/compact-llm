"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { fetchHistory, type HistoryItem } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/page-primitives";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 15;

function truncate(text: string, n = 60) {
  return text.length > n ? text.slice(0, n) + "…" : text;
}

type Filter = "all" | "comparison" | "score-only";

export default function HistoryPage() {
  const [page, setPage] = React.useState(1);
  const [data, setData] = React.useState<{
    page: number;
    items: HistoryItem[];
    total: number;
  } | null>(null);
  const [sortDesc, setSortDesc] = React.useState(true);
  const [filter, setFilter] = React.useState<Filter>("all");

  React.useEffect(() => {
    let cancelled = false;
    fetchHistory(page, PAGE_SIZE)
      .then((res) => !cancelled && setData({ page, items: res.items, total: res.total }))
      .catch(() => !cancelled && setData({ page, items: [], total: 0 }));
    return () => {
      cancelled = true;
    };
  }, [page]);

  const loading = !data || data.page !== page;
  const items = loading ? null : data.items;
  const total = data?.total ?? 0;

  const rows = React.useMemo(() => {
    if (!items) return [];
    let r = items;
    if (filter === "comparison") r = r.filter((x) => x.is_comparison);
    if (filter === "score-only") r = r.filter((x) => !x.is_comparison);
    return [...r].sort((a, b) =>
      sortDesc ? b.finetuned_score - a.finetuned_score : a.finetuned_score - b.finetuned_score
    );
  }, [items, filter, sortDesc]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div>
      <PageHeader
        title="History"
        subtitle="Every score and comparison request, persisted to the database."
        action={
          <div className="flex gap-1.5">
            {(["all", "comparison", "score-only"] as const).map((f) => (
              <Button
                key={f}
                size="sm"
                variant={filter === f ? "default" : "outline"}
                onClick={() => setFilter(f)}
              >
                {f === "all" ? "All" : f === "comparison" ? "Compare" : "Score"}
              </Button>
            ))}
          </div>
        }
      />

      <div className="rounded-lg border border-border bg-card p-4">
        {items === null ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <p className="py-10 text-center text-sm text-muted-foreground">
            Nothing to show — run a score or comparison first.
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Resume</TableHead>
                    <TableHead>Job Description</TableHead>
                    <TableHead
                      className="cursor-pointer select-none text-right"
                      onClick={() => setSortDesc((s) => !s)}
                    >
                      FT score {sortDesc ? "↓" : "↑"}
                    </TableHead>
                    <TableHead className="text-right">Base</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead className="text-right">When</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row) => (
                    <TableRow key={row.id}>
                      <TableCell className="max-w-[220px] text-xs text-muted-foreground">
                        {truncate(row.resume_text)}
                      </TableCell>
                      <TableCell className="max-w-[220px] text-xs text-muted-foreground">
                        {truncate(row.jd_text)}
                      </TableCell>
                      <TableCell className="font-metric text-right font-medium">
                        {row.finetuned_score}
                      </TableCell>
                      <TableCell className="font-metric text-right text-muted-foreground">
                        {row.base_score ?? "—"}
                      </TableCell>
                      <TableCell>
                        <span
                          className={cn(
                            "label-caps",
                            row.is_comparison ? "text-primary" : "text-muted-foreground"
                          )}
                        >
                          {row.is_comparison ? "compare" : "score"}
                        </span>
                      </TableCell>
                      <TableCell className="font-metric text-right text-[11px] text-muted-foreground">
                        {new Date(row.created_at).toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            <div className="flex items-center justify-between pt-4">
              <span className="font-metric text-[11px] text-muted-foreground">
                page {page}/{totalPages} · {total} total
              </span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                >
                  <ChevronLeft className="h-4 w-4" /> Prev
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                >
                  Next <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
