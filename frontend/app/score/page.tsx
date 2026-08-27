"use client";

import * as React from "react";
import { Loader2, Play } from "lucide-react";
import { toast } from "sonner";

import { EXAMPLES } from "@/lib/examples";
import { streamScore } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/page-primitives";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";

export default function ScorePage() {
  const [resume, setResume] = React.useState("");
  const [jobDescription, setJobDescription] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const [rawText, setRawText] = React.useState("");
  const [score, setScore] = React.useState<number | null>(null);
  const [rationale, setRationale] = React.useState<string | null>(null);
  const [latencyMs, setLatencyMs] = React.useState<number | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);

  const canSubmit =
    resume.trim().length > 0 && jobDescription.trim().length > 0 && !streaming;

  function loadExample() {
    const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    setResume(ex.resume);
    setJobDescription(ex.jd);
    setScore(null);
    setRationale(null);
    setRawText("");
    setLatencyMs(null);
  }

  async function handleScore() {
    setStreaming(true);
    setRawText("");
    setScore(null);
    setRationale(null);
    setLatencyMs(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamScore(
        resume,
        jobDescription,
        (event) => {
          if (event.type === "chunk") {
            setRawText((prev) => prev + event.text);
          } else {
            setScore(event.score);
            setRationale(event.rationale);
            setLatencyMs(event.latency_ms);
          }
        },
        controller.signal
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        toast.error("Scoring failed — is the backend running?");
        console.error(err);
      }
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Score"
        subtitle="The fine-tuned model streams a 0–100 fit score and a two-sentence rationale."
        action={
          <Button variant="outline" size="sm" onClick={loadExample} disabled={streaming}>
            Load example
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <div className="flex flex-col gap-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Resume" value={resume} onChange={setResume} placeholder="Paste resume text…" />
            <Field
              label="Job Description"
              value={jobDescription}
              onChange={setJobDescription}
              placeholder="Paste job description text…"
            />
          </div>
          <div>
            <Button onClick={handleScore} disabled={!canSubmit}>
              {streaming ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scoring…
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" /> Score match
                </>
              )}
            </Button>
          </div>
        </div>

        <ResultPanel
          streaming={streaming}
          score={score}
          rationale={rationale}
          rawText={rawText}
          latencyMs={latencyMs}
        />
      </div>
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
        className="min-h-[280px] resize-y font-mono text-xs leading-relaxed"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function ResultPanel({
  streaming,
  score,
  rationale,
  rawText,
  latencyMs,
}: {
  streaming: boolean;
  score: number | null;
  rationale: string | null;
  rawText: string;
  latencyMs: number | null;
}) {
  const idle = !streaming && score === null && !rawText;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-baseline justify-between">
        <span className="label-caps">Fit score</span>
        {latencyMs !== null && (
          <span className="font-metric text-[11px] text-muted-foreground">
            {(latencyMs / 1000).toFixed(2)}s
          </span>
        )}
      </div>

      <div className="font-metric mt-1 text-4xl font-semibold tracking-tight">
        {score !== null ? score : idle ? "—" : <span className="text-muted-foreground">··</span>}
        <span className="ml-1 text-lg text-muted-foreground">/100</span>
      </div>

      <Progress value={score ?? 0} className="mt-3 h-1.5" />

      <div className="mt-4 min-h-[120px] text-sm leading-relaxed">
        {idle ? (
          <p className="text-muted-foreground">
            Results stream here as the model generates them.
          </p>
        ) : (
          <p className="whitespace-pre-wrap">
            {rationale ?? rawText}
            {streaming && <span className="animate-pulse">▍</span>}
          </p>
        )}
      </div>
    </div>
  );
}
