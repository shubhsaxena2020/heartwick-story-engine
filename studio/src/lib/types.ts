// Snapshot schema 1, written by dashboard/studio_state.py on the main laptop. Every field may be missing or null:
// the UI shows an empty state instead of failing.

export type Stage = "script" | "narration" | "images" | "render" | "review" | "approved" | "published";
export const STAGES: Stage[] = ["script", "narration", "images", "render", "review", "approved", "published"];

export interface MediaRef {
  key: string;
  bytes: number;
}

export interface Approval {
  status: "approved" | "needs_changes" | "pending" | "published";
  note?: string;
  url?: string | null;
  at?: string;
  id?: string;
  applied_at?: string;
}

export interface Video {
  name: string;
  story: string;
  title?: string | null;
  series_tag?: string | null;
  kind?: "short" | "long";
  voice?: string | null;
  stage: Stage;
  reached: Partial<Record<Stage, string>>;
  progress: { lines_done: number; lines_total: number; images_done: number; images_total: number };
  chapters?: number;
  duration_s?: number | null;
  shots?: number | null;
  shots_per_min?: number | null;
  caption_chunks?: number | null;
  srt_cues?: number | null;
  lufs?: number | null;
  true_peak?: number | null;
  lra?: number | null;
  sync?: { chunks?: number | null; speech_present?: number | null; onset_chunks?: number | null; onset_median_s?: number | null; onset_p90_abs_s?: number | null } | null;
  asr?: {
    wer?: number | null;
    ref_words?: number | null;
    asr_words?: number | null;
    matched_chunk_starts?: number | null;
    caption_vs_speech_median_s?: number | null;
    caption_vs_speech_p90_abs_s?: number | null;
  } | null;
  render_s?: number | null;
  verify_s?: number | null;
  worker_total_s?: number | null;
  worker_stages?: Record<string, number>;
  mp4_bytes?: number | null;
  media: Partial<Record<"thumb" | "sheet" | "vsref" | "masters" | "preview", MediaRef | null>>;
  preview_pending?: boolean;
  samples?: (MediaRef & { name: string })[];
  srt?: string | null;
  seo_md?: string | null;
  seo?: { title?: string | null; titles?: string[]; filename?: string | null; tags?: number; hashtags?: string[] } | null;
  approval?: Approval | null;
  errors: string[];
  warnings: string[];
  updated_at: string;
  engine?: string | null;
  v3_stages?: Record<string, { status?: string; run_min?: number | null; wall_min?: number | null; gpu?: boolean; kernel?: string }> | null;
}

export interface Story {
  slug: string;
  title?: string | null;
  file: string;
  chapters?: number;
  shots?: number;
  lines?: number;
  words?: number;
  characters?: string[];
  mtime?: string;
  error?: string | null;
  outputs?: string[];
}

export interface KaggleJob {
  kernel: string;
  gpu: boolean;
  pushed_at: string;
  state: "queued" | "running" | "complete" | "error" | "cancelled" | "unknown" | "superseded";
  state_at?: string | null;
  wait_min?: number | null;
  run_min?: number | null;
  wall_min?: number | null;
  images?: number;
  utts?: number;
  render?: boolean;
  video?: string | null;
  story?: string | null;
  worker_stages?: Record<string, number> | null;
  worker_errors?: string[];
  worker_total_s?: number | null;
  note?: string | null;
  live_status?: string | null;
}

export interface Kaggle {
  quota?: { used_h: number; remaining_h: number; total_h: number; refresh_at: string; checked_at: string; source?: string } | null;
  week_start?: string;
  quota_total_h?: number;
  gpu_min_week_local?: number;
  gpu_min_total_local?: number;
  cpu_min_total_local?: number;
  jobs?: KaggleJob[];
  by_day?: { date: string; gpu_min: number; cpu_min: number }[];
  by_video?: { video: string; gpu_min: number; cpu_min: number }[];
  poll?: { checked_at?: string | null; error?: string | null };
}

export interface QueueJob {
  id: string;
  label: string;
  args?: string[];
  status: "pending" | "running" | "done" | "failed" | "interrupted";
  added_at?: string;
  started_at?: string;
  finished_at?: string;
  rc?: number;
  source?: string;
  note?: string;
}

export interface Queue {
  runner?: { running: boolean; pids: number[]; started_at?: string | null };
  current?: { text: string; since: string } | null;
  log_tail?: { at: string | null; text: string; level: "info" | "warn" | "err" }[];
  jobs?: QueueJob[];
}

export interface StudioEvent {
  at: string;
  level: "info" | "warn" | "err";
  source: string;
  text: string;
  video?: string | null;
}

export interface TruthCheck {
  key: string;
  label: string;
  expected: unknown;
  shown: unknown;
  ok: boolean | null;
  tol?: number | null;
  page?: string;
  source?: string;
  note?: string | null;
}

export interface Truth {
  at: string;
  n_checks: number;
  n_fail: number;
  n_skip: number;
  checks: TruthCheck[];
}

export interface Research {
  present?: boolean;
  report_md?: string | null;
  report_truncated?: boolean;
  report_mtime?: string | null;
  samples?: (MediaRef & { name: string })[];
  other_docs?: { name: string; bytes: number; mtime: string }[];
}

export interface AppliedAction {
  id: string;
  kind: string;
  video?: string | null;
  story?: string | null;
  at?: string | null;
  applied_at: string;
  via: string;
  result: "ok" | "rejected" | "error";
  msg?: string;
}

export interface Snapshot {
  schema: 1;
  generated_at: string;
  stages?: Stage[];
  counts?: Partial<Record<Stage, number>>;
  stories?: Story[];
  videos?: Video[];
  kaggle?: Kaggle;
  queue?: Queue;
  research?: Research;
  channel?: { exists: boolean; note?: string; youtube?: { subscribers?: number; views?: number } | null; instagram?: { followers?: number; views?: number } | null };
  actions?: { applied?: AppliedAction[] };
  events?: StudioEvent[];
  truth?: Truth | null;
  exporter?: {
    pid?: number;
    started_at?: string;
    heartbeat_s?: number;
    last_ingest?: { at: string; ok: boolean; bytes?: number; error?: string } | null;
    media?: { uploaded: number; deleted: number; failed: number; bytes_stored: number; at?: string } | null;
    uploaded_keys?: string[];
  } | null;
}

/** Live readings that ride on each poll of the laptop's tunnel. */
export interface LiveReading {
  at: string;
  cpu_pct?: number | null;
  ram_pct?: number | null;
  viewers?: number;
}

export interface LiveInfo {
  url: string | null;
  at: string | null;
}

export interface SnapshotResponse {
  snapshot: Snapshot | null;
  stored_at: string | null;
  live: LiveInfo;
}

export type ActionKind = "approve" | "needs_changes" | "reset" | "published" | "queue_story" | "rerender";

export interface StudioAction {
  id: string;
  kind: ActionKind;
  video?: string;
  story?: string;
  mode?: "short" | "full";
  note?: string;
  url?: string;
  at: string;
}
