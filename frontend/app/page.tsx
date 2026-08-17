"use client";

import * as React from "react";
import { Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { streamScore } from "@/lib/api";

export default function ScorerPage() {
  const [resume, setResume] = React.useState("");
  const [jobDescription, setJobDescription] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const [rawText, setRawText] = React.useState("");
  const [score, setScore] = React.useState<number | null>(null);
  const [rationale, setRationale] = React.useState<string | null>(null);
  const [latencyMs, setLatencyMs] = React.useState<number | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);

  const canSubmit = resume.trim().length > 0 && jobDescription.trim().length > 0 && !streaming;

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
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Live Scorer</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Paste a resume and a job description. The fine-tuned model streams a 0-100 score
          and a short rationale as it generates.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2">
          <Label htmlFor="resume">Resume</Label>
          <Textarea
            id="resume"
            placeholder="Paste resume text..."
            className="min-h-[240px] font-mono text-xs"
            value={resume}
            onChange={(e) => setResume(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="jd">Job Description</Label>
          <Textarea
            id="jd"
            placeholder="Paste job description text..."
            className="min-h-[240px] font-mono text-xs"
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
        </div>
      </div>

      <div>
        <Button onClick={handleScore} disabled={!canSubmit}>
          {streaming ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scoring...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-4 w-4" /> Score match
            </>
          )}
        </Button>
      </div>

      {(streaming || rawText || score !== null) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Result</span>
              {score !== null && (
                <span className="text-3xl font-bold tabular-nums">{score}/100</span>
              )}
            </CardTitle>
            {score !== null && <Progress value={score} className="mt-2" />}
            {latencyMs !== null && (
              <CardDescription>{(latencyMs / 1000).toFixed(2)}s</CardDescription>
            )}
          </CardHeader>
          <CardContent>
            {score === null && streaming && !rawText ? (
              <div className="flex flex-col gap-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-2/3" />
              </div>
            ) : (
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {rationale ?? rawText}
                {streaming && <span className="animate-pulse">▍</span>}
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
