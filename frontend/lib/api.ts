export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: string;
  model_backend: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health failed: ${res.status}`);
  return res.json();
}

export interface ScoreResult {
  score: number;
  rationale: string;
  latency_ms: number;
}

export interface CompareResponse {
  base: ScoreResult;
  finetuned: ScoreResult;
  score_delta: number;
}

export interface HistoryItem {
  id: number;
  resume_text: string;
  jd_text: string;
  finetuned_score: number;
  finetuned_rationale: string;
  base_score: number | null;
  is_comparison: boolean;
  created_at: string;
}

export interface HistoryPage {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface BenchmarkRow {
  approach: string;
  n: number;
  pearson: number;
  spearman: number;
  mae: number;
  rationale_quality_1_5: number | null;
  mean_latency_sec: number;
}

export async function compareScores(resume: string, jobDescription: string): Promise<CompareResponse> {
  const res = await fetch(`${API_URL}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume, job_description: jobDescription }),
  });
  if (!res.ok) throw new Error(`compare failed: ${res.status}`);
  return res.json();
}

export async function fetchHistory(page = 1, pageSize = 20): Promise<HistoryPage> {
  const res = await fetch(`${API_URL}/history?page=${page}&page_size=${pageSize}`);
  if (!res.ok) throw new Error(`history failed: ${res.status}`);
  return res.json();
}

export async function fetchBenchmark(): Promise<BenchmarkRow[]> {
  const res = await fetch(`${API_URL}/eval/benchmark`);
  if (!res.ok) throw new Error(`benchmark failed: ${res.status}`);
  return res.json();
}

export interface DatasetSplit {
  name: string;
  count: number;
}

export interface DatasetStats {
  available: boolean;
  source: string;
  labeler: string;
  total: number;
  splits: DatasetSplit[];
  score_histogram: { bucket: string; count: number }[];
  score_mean: number | null;
  instruction: string | null;
  samples: { input: string; output: string }[];
}

export async function fetchDatasetStats(): Promise<DatasetStats> {
  const res = await fetch(`${API_URL}/datasets/stats`);
  if (!res.ok) throw new Error(`dataset stats failed: ${res.status}`);
  return res.json();
}

export type ScoreStreamEvent =
  | { type: "chunk"; text: string }
  | { type: "done"; score: number; rationale: string; latency_ms: number };

export async function streamScore(
  resume: string,
  jobDescription: string,
  onEvent: (event: ScoreStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_URL}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume, job_description: jobDescription }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`score failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const jsonStr = trimmed.slice("data:".length).trim();
      if (!jsonStr) continue;
      onEvent(JSON.parse(jsonStr) as ScoreStreamEvent);
    }
  }
}
