"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { fetchHistory, type HistoryItem } from "@/lib/api";

const PAGE_SIZE = 15;

function truncate(text: string, n = 60) {
  return text.length > n ? text.slice(0, n) + "…" : text;
}

export default function HistoryPage() {
  const [page, setPage] = React.useState(1);
  const [items, setItems] = React.useState<HistoryItem[] | null>(null);
  const [total, setTotal] = React.useState(0);
  const [sortDesc, setSortDesc] = React.useState(true);
  const [filter, setFilter] = React.useState<"all" | "comparison" | "score-only">("all");

  React.useEffect(() => {
    setItems(null);
    fetchHistory(page, PAGE_SIZE)
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
      })
      .catch(() => setItems([]));
  }, [page]);

  const filtered = React.useMemo(() => {
    if (!items) return [];
    let rows = items;
    if (filter === "comparison") rows = rows.filter((r) => r.is_comparison);
    if (filter === "score-only") rows = rows.filter((r) => !r.is_comparison);
    return [...rows].sort((a, b) =>
      sortDesc ? b.finetuned_score - a.finetuned_score : a.finetuned_score - b.finetuned_score
    );
  }, [items, sortDesc, filter]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">History</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Past scoring requests, newest first.
        </p>
      </div>

      <div className="flex items-center gap-2">
        {(["all", "comparison", "score-only"] as const).map((f) => (
          <Button
            key={f}
            size="sm"
            variant={filter === f ? "default" : "outline"}
            onClick={() => setFilter(f)}
          >
            {f === "all" ? "All" : f === "comparison" ? "Comparisons" : "Score only"}
          </Button>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Requests</CardTitle>
        </CardHeader>
        <CardContent>
          {items === null ? (
            <div className="flex flex-col gap-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : items.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No requests yet — try the Scorer or Compare page.
            </p>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Resume</TableHead>
                    <TableHead>Job Description</TableHead>
                    <TableHead
                      className="cursor-pointer select-none"
                      onClick={() => setSortDesc((s) => !s)}
                    >
                      Fine-tuned score {sortDesc ? "↓" : "↑"}
                    </TableHead>
                    <TableHead>Base score</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>When</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((row) => (
                    <TableRow key={row.id}>
                      <TableCell className="max-w-[220px] text-xs text-muted-foreground">
                        {truncate(row.resume_text)}
                      </TableCell>
                      <TableCell className="max-w-[220px] text-xs text-muted-foreground">
                        {truncate(row.jd_text)}
                      </TableCell>
                      <TableCell className="font-medium tabular-nums">
                        {row.finetuned_score}
                      </TableCell>
                      <TableCell className="tabular-nums">
                        {row.base_score ?? "—"}
                      </TableCell>
                      <TableCell>
                        <Badge variant={row.is_comparison ? "default" : "secondary"}>
                          {row.is_comparison ? "compare" : "score"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {new Date(row.created_at).toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <div className="flex items-center justify-between pt-4">
                <span className="text-xs text-muted-foreground">
                  Page {page} of {totalPages} ({total} total)
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
        </CardContent>
      </Card>
    </div>
  );
}
