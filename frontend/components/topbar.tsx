"use client";

import * as React from "react";
import { usePathname } from "next/navigation";

import { fetchHealth, type HealthResponse } from "@/lib/api";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const TITLES: Record<string, string> = {
  "/": "Overview",
  "/score": "Score",
  "/playground": "Playground",
  "/evaluation": "Evaluation",
  "/datasets": "Datasets",
  "/history": "History",
};

function backendLabel(backend: string) {
  if (backend === "mock") return "mock backend";
  if (backend === "transformers") return "transformers · GPU";
  if (backend === "ollama") return "ollama · GGUF";
  return backend;
}

export function Topbar() {
  const pathname = usePathname();
  const title =
    TITLES[pathname] ??
    Object.entries(TITLES).find(([href]) => href !== "/" && pathname.startsWith(href))?.[1] ??
    "CompactLLM";

  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [offline, setOffline] = React.useState(false);

  React.useEffect(() => {
    let alive = true;
    const tick = () =>
      fetchHealth()
        .then((h) => alive && (setHealth(h), setOffline(false)))
        .catch(() => alive && setOffline(true));
    tick();
    const id = setInterval(tick, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <header className="sticky top-0 z-20 flex h-11 items-center justify-between border-b border-border bg-card/80 px-4 backdrop-blur">
      <div className="flex items-center gap-2 text-sm">
        <span className="font-medium md:hidden">CompactLLM</span>
        <span className="hidden text-muted-foreground md:inline">CompactLLM</span>
        <span className="hidden text-muted-foreground md:inline">/</span>
        <span className="font-medium">{title}</span>
      </div>

      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              offline
                ? "bg-destructive"
                : health
                ? "animate-pulse bg-accent"
                : "bg-muted-foreground"
            )}
          />
          <span className="font-metric">
            {offline ? "backend offline" : health ? backendLabel(health.model_backend) : "…"}
          </span>
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
