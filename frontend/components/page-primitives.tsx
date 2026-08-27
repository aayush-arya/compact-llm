import { TrendingDown, TrendingUp } from "lucide-react";

import { cn } from "@/lib/utils";

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function StatCard({
  label,
  value,
  sub,
  delta,
  deltaGood,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  /** e.g. "+19.1%" — rendered with an arrow */
  delta?: string;
  /** whether the delta is a good outcome (green) or bad (red); omit for neutral */
  deltaGood?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <p className="label-caps">{label}</p>
      <div className="font-metric mt-1.5 text-xl font-semibold">{value}</div>
      {(sub || delta) && (
        <div className="mt-1 flex items-center gap-2 text-[11px]">
          {delta && (
            <span
              className={cn(
                "font-metric inline-flex items-center gap-0.5",
                deltaGood === undefined
                  ? "text-muted-foreground"
                  : deltaGood
                  ? "text-accent"
                  : "text-destructive"
              )}
            >
              {deltaGood === undefined ? null : deltaGood ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              {delta}
            </span>
          )}
          {sub && <span className="text-muted-foreground">{sub}</span>}
        </div>
      )}
    </div>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="label-caps mb-2">{children}</p>;
}
