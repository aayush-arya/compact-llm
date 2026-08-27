"use client";

import * as React from "react";
import { Loader2, Play } from "lucide-react";
import { toast } from "sonner";

import { EXAMPLES } from "@/lib/examples";
import { compareScores, type CompareResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PageHeader, SectionLabel } from "@/components/page-primitives";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

export default function PlaygroundPage() {
  const [resume, setResume] = React.useState("");
  const [jobDescription, setJobDescription] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<CompareResponse | null>(null);

  const canSubmit =
    resume.trim().length > 0 && jobDescription.trim().length > 0 && !loading;

  function loadExample() {
    const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    setResume(ex.resume);
    setJobDescription(ex.jd);
    setResult(null);
  }

  async function handleCompare() {
    setLoading(true);
    setResult(null);
    try {
      setResult(await compareScores(resume, jobDescription));
    } catch (err) {
      toast.error("Comparison failed — is the backend running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Playground"
        subtitle="Same resume and JD through the base model and the QLoRA fine-tune, off one loaded set of weights."
        action={
          <Button variant="outline" size="sm" onClick={loadExample} disabled={loading}>
            Load example
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <ModelChip
          slot="Model A — base"
          name="gemma-3-4b-it"
          detail="zero-shot, no adapter"
        />
        <ModelChip
          slot="Model B — fine-tuned"
          name="gemma-3-4b-it + adapter"
          detail="QLoRA r16 · resume-jd-relevance"
          highlight
        />
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Field label="Resume" value={resume} onChange={setResume} placeholder="Paste resume text…" />
        <Field
          label="Job Description"
          value={jobDescription}
          onChange={setJobDescription}
          placeholder="Paste job description text…"
        />
      </div>

      <div className="mt-4">
        <Button onClick={handleCompare} disabled={!canSubmit}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Comparing…
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" /> Run comparison
            </>
          )}
        </Button>
      </div>

      {loading && (
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {result && !loading && <Results result={result} />}
    </div>
  );
}

function ModelChip({
  slot,
  name,
  detail,
  highlight,
}: {
  slot: string;
  name: string;
  detail: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-3",
        highlight ? "border-primary/50" : "border-border"
      )}
    >
      <p className="label-caps">{slot}</p>
      <p className="font-metric mt-1 text-sm">{name}</p>
      <p className="text-[11px] text-muted-foreground">{detail}</p>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="label-caps">{label}</span>
      <Textarea
        placeholder={placeholder}
        className="min-h-[220px] resize-y font-mono text-xs leading-relaxed"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function Results({ result }: { result: CompareResponse }) {
  const { base, finetuned, score_delta } = result;
  const latencyDelta =
    base.latency_ms > 0
      ? ((finetuned.latency_ms - base.latency_ms) / base.latency_ms) * 100
      : 0;

  return (
    <div className="mt-6 flex flex-col gap-4">
      <div className="grid gap-4 md:grid-cols-2">
        <OutputCard title="Base" badge="zero-shot" result={base} />
        <OutputCard title="Fine-tuned" badge="QLoRA adapter" result={finetuned} highlight />
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <SectionLabel>Fine-tuned vs base</SectionLabel>
        <div className="grid gap-3 sm:grid-cols-3">
          <Metric
            name="Score"
            base={base.score}
            ft={finetuned.score}
            delta={`${score_delta > 0 ? "+" : ""}${score_delta}`}
          />
          <Metric
            name="Latency"
            base={`${(base.latency_ms / 1000).toFixed(2)}s`}
            ft={`${(finetuned.latency_ms / 1000).toFixed(2)}s`}
            delta={`${latencyDelta > 0 ? "+" : ""}${latencyDelta.toFixed(0)}%`}
            deltaGood={latencyDelta < 0}
          />
          <Metric
            name="Decisiveness"
            base={`${Math.abs(base.score - 50)}`}
            ft={`${Math.abs(finetuned.score - 50)}`}
            hint="distance from a hedged 50"
          />
        </div>
        <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
          The fine-tune commits — it pushes strong matches high and weak matches
          low, where the base model hedges toward the middle of the scale. A
          negative score delta on a poor match means the fine-tune correctly
          rated it lower.
        </p>
      </div>
    </div>
  );
}

function OutputCard({
  title,
  badge,
  result,
  highlight,
}: {
  title: string;
  badge: string;
  result: { score: number; rationale: string; latency_ms: number };
  highlight?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-4",
        highlight ? "border-primary/50" : "border-border"
      )}
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm font-medium">
          {title}
          <span className="label-caps">{badge}</span>
        </span>
        <span className="font-metric text-2xl font-semibold">{result.score}</span>
      </div>
      <Progress value={result.score} className="mt-2 h-1.5" />
      <p className="mt-3 text-sm leading-relaxed">{result.rationale}</p>
      <p className="font-metric mt-3 text-[11px] text-muted-foreground">
        {(result.latency_ms / 1000).toFixed(2)}s
      </p>
    </div>
  );
}

function Metric({
  name,
  base,
  ft,
  delta,
  deltaGood,
  hint,
}: {
  name: string;
  base: React.ReactNode;
  ft: React.ReactNode;
  delta?: string;
  deltaGood?: boolean;
  hint?: string;
}) {
  return (
    <div>
      <p className="label-caps">{name}</p>
      <div className="font-metric mt-1 flex items-baseline gap-2">
        <span className="text-lg font-semibold">{ft}</span>
        <span className="text-xs text-muted-foreground line-through decoration-muted-foreground/40">
          {base}
        </span>
        {delta && (
          <span
            className={cn(
              "text-xs",
              deltaGood === undefined
                ? "text-muted-foreground"
                : deltaGood
                ? "text-accent"
                : "text-destructive"
            )}
          >
            {delta}
          </span>
        )}
      </div>
      {hint && <p className="mt-0.5 text-[10px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
