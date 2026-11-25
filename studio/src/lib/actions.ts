import type { ActionKind, StudioAction } from "./types";

// Validation shared by /api/actions (server) and the buttons (client). The laptop validates again
// (dashboard/actions.py) against its real files before applying anything.

export const ACTION_KINDS: ActionKind[] = ["approve", "needs_changes", "reset", "published", "queue_story", "rerender"];
const NAME = /^[a-z0-9][a-z0-9_]{0,63}$/;

export function checkAction(raw: unknown): { ok: true; action: Omit<StudioAction, "id" | "at"> } | { ok: false; error: string } {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { ok: false, error: "expected an object" };
  const r = raw as Record<string, unknown>;
  const kind = r.kind as ActionKind;
  if (!ACTION_KINDS.includes(kind)) return { ok: false, error: "unknown kind" };
  const note = typeof r.note === "string" ? r.note.trim().slice(0, 2000) : undefined;
  if (kind === "queue_story") {
    if (typeof r.story !== "string" || !NAME.test(r.story)) return { ok: false, error: "bad story" };
    if (r.mode !== "short" && r.mode !== "full") return { ok: false, error: "mode must be short or full" };
    return { ok: true, action: { kind, story: r.story, mode: r.mode, ...(note ? { note } : {}) } };
  }
  if (typeof r.video !== "string" || !NAME.test(r.video)) return { ok: false, error: "bad video" };
  if (kind === "needs_changes" && !note) return { ok: false, error: "say what needs to change" };
  let url: string | undefined;
  if (kind === "published" && typeof r.url === "string" && r.url.trim()) {
    url = r.url.trim().slice(0, 500);
    if (!/^https:\/\/[^\s]+$/.test(url)) return { ok: false, error: "link must start with https://" };
  }
  return { ok: true, action: { kind, video: r.video, ...(note ? { note } : {}), ...(url ? { url } : {}) } };
}
