"use client";

import * as React from "react";
import { ArrowRight, Loader2, TrendingDown, TrendingUp } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { compareScores, type CompareResponse } from "@/lib/api";

function ModelCard({
  title,
  badge,
  score,
  rationale,
  latencyMs,
  highlight,
}: {
  title: string;
  badge: string;
  score: number;
  rationale: string;
  latencyMs: number;
  highlight?: boolean;
}) {
  return (
    <Card className={highlight ? "border-primary shadow-md" : undefined}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            {title}
            <Badge variant={highlight ? "default" : "secondary"}>{badge}</Badge>
          </span>
          <span className="text-3xl font-bold tabular-nums">{score}</span>
        </CardTitle>
        <Progress value={score} className="mt-2" />
        <CardDescription>{(latencyMs / 1000).toFixed(2)}s</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm leading-relaxed">{rationale}</p>
      </CardContent>
    </Card>
  );
}

export default function ComparePage() {
  const [resume, setResume] = React.useState("");
  const [jobDescription, setJobDescription] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<CompareResponse | null>(null);

  const canSubmit = resume.trim().length > 0 && jobDescription.trim().length > 0 && !loading;

  async function handleCompare() {
    setLoading(true);
    setResult(null);
    try {
      const res = await compareScores(resume, jobDescription);
      setResult(res);
    } catch (err) {
      toast.error("Comparison failed — is the backend running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Base vs. Fine-Tuned</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Same input, run through both the vanilla base model and the QLoRA fine-tuned
          adapter — off the same loaded weights. This is the delta the fine-tune bought you.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2">
          <Label htmlFor="c-resume">Resume</Label>
          <Textarea
            id="c-resume"
            placeholder="Paste resume text..."
            className="min-h-[200px] font-mono text-xs"
            value={resume}
            onChange={(e) => setResume(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="c-jd">Job Description</Label>
          <Textarea
            id="c-jd"
            placeholder="Paste job description text..."
            className="min-h-[200px] font-mono text-xs"
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
        </div>
      </div>

      <div>
        <Button onClick={handleCompare} disabled={!canSubmit}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Comparing...
            </>
          ) : (
            <>
              Compare models <ArrowRight className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>
      </div>

      {loading && (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-56 w-full" />
          <Skeleton className="h-56 w-full" />
        </div>
      )}

      {result && (
        <>
          <div className="flex items-center justify-center gap-3 rounded-lg border bg-muted/40 py-4">
            <span className="text-sm text-muted-foreground">Score delta (fine-tuned − base)</span>
            <span
              className={`flex items-center gap-1 text-2xl font-bold tabular-nums ${
                result.score_delta > 0
                  ? "text-emerald-500"
                  : result.score_delta < 0
                  ? "text-red-500"
                  : "text-muted-foreground"
              }`}
            >
              {result.score_delta > 0 ? (
                <TrendingUp className="h-5 w-5" />
              ) : result.score_delta < 0 ? (
                <TrendingDown className="h-5 w-5" />
              ) : null}
              {result.score_delta > 0 ? "+" : ""}
              {result.score_delta}
            </span>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <ModelCard
              title="Base model"
              badge="zero-shot"
              score={result.base.score}
              rationale={result.base.rationale}
              latencyMs={result.base.latency_ms}
            />
            <ModelCard
              title="Fine-tuned"
              badge="QLoRA adapter"
              score={result.finetuned.score}
              rationale={result.finetuned.rationale}
              latencyMs={result.finetuned.latency_ms}
              highlight
            />
          </div>
        </>
      )}
    </div>
  );
}
