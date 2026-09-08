/** Typed access to the fact knowledge layer API. */

export type Verdict = "corroborates" | "contradicts" | "reconciled";

export interface Fact {
  id: number;
  value_raw: string;
  unit_raw: string | null;
  period_raw: string | null;
  basis: string | null;
  modality: string;
  variant: string | null;
  value_norm: number | null;
  unit: string | null;
  scale: string | null;
  currency: string | null;
  period_start: string | null;
  period_end: string | null;
  period_kind: string | null;
  quote: string;
  page_no: number;
  char_start: number;
  char_end: number;
  bboxes: number[][];
  core_key: string | null;
  full_key: string | null;
  status: string;
  doc_id: number;
  entity: string | null;
  metric: string | null;
  doc_title: string | null;
  publisher: string | null;
  published_date: string | null;
  source_tier: number;
  filename: string;
}

export interface Relation {
  id: number;
  fact_a: number;
  fact_b: number;
  verdict: Verdict;
  rule: string;
  delta: number | null;
  explanation: string;
  qualifier_inferred: number;
  a: Fact | null;
  b: Fact | null;
}

export interface DocumentRow {
  id: number;
  filename: string;
  title: string | null;
  publisher: string | null;
  doc_type: string | null;
  published_date: string | null;
  source_tier: number;
  n_pages: number;
  n_facts: number;
  ingested_at: string;
}

export interface Cluster {
  core_key: string;
  entity: string | null;
  metric: string | null;
  period_start: string | null;
  period_end: string | null;
  n_facts: number;
  n_docs: number;
}

export interface Stats {
  documents: number;
  pages: number;
  facts: number;
  entities: number;
  metrics: number;
  relations: number;
  verdicts: Record<string, number>;
  grounding_rejection_rate: number;
  failures: { stage: string; reason: string; n: number }[];
}

export interface Job {
  id: string;
  filename: string;
  status: "queued" | "running" | "done" | "failed";
  detail: string;
  report: Record<string, number | string>;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  documents: () => get<DocumentRow[]>("/api/documents"),
  stats: () => get<Stats>("/api/stats"),
  clusters: () => get<Cluster[]>("/api/clusters"),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),

  facts: (params: Record<string, string> = {}) => {
    const query = new URLSearchParams(params).toString();
    return get<{ total: number; facts: Fact[] }>(`/api/facts?${query}`);
  },

  fact: (id: number) =>
    get<Fact & { relations: (Relation & { other: Fact | null })[] }>(`/api/facts/${id}`),

  conflicts: (verdict?: string) =>
    get<Relation[]>(`/api/conflicts${verdict ? `?verdict=${verdict}` : ""}`),

  async upload(file: File): Promise<Job> {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/documents", { method: "POST", body });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return (await response.json()) as Job;
  },
};
