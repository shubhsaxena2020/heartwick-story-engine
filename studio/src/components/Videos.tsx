"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { arr, bytes, mmss, num, signed } from "@/lib/format";
import type { Snapshot, StudioAction, Video } from "@/lib/types";
import { STAGE_NAME, Stepper, stageTone } from "./Overview";
import { Bar, Btn, Chips, Empty, Img, Lamp, Panel, Pill, Section, Time, useMediaSrc, useStudio } from "./ui";

const FILTERS = ["all", "review", "approved", "changes", "making"] as const;
type Filter = (typeof FILTERS)[number];
const FILTER_NAME: Record<Filter, string> = { all: "All", review: "To review", approved: "Approved", changes: "Needs changes", making: "Being made" };

function matches(v: Video, f: Filter): boolean {
  if (f === "all") return true;
  if (f === "review") return v.stage === "review" && v.approval?.status !== "needs_changes";
  if (f === "approved") return v.stage === "approved" || v.stage === "published";
  if (f === "changes") return v.approval?.status === "needs_changes";
  return ["script", "narration", "images", "render"].includes(v.stage);
}

export function Videos({ s, pending }: { s: Snapshot; pending: Map<string, StudioAction> }) {
  const videos = arr(s.videos);
  const [f, setF] = useState<Filter>("all");
  const counts = useMemo(() => Object.fromEntries(FILTERS.map((k) => [k, videos.filter((v) => matches(v, k)).length])) as Record<Filter, number>, [videos]);
  const shown = videos.filter((v) => matches(v, f));
  return (
    <>
      <Section title="Videos" hint="newest first · approve or ask for changes; it reaches the laptop even if it is off" right={<Chips value={f} options={FILTERS} onChange={setF} label="Filter videos" counts={counts} names={FILTER_NAME} />}>
        {shown.length ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {shown.map((v, i) => (
              <VideoCard key={v.name} v={v} i={i} pending={[...pending.values()].filter((a) => a.video === v.name)} applied={arr(s.actions?.applied).filter((a) => a.video === v.name)} />
            ))}
          </div>
        ) : (
          <Empty>No video matches this filter.</Empty>
        )}
      </Section>
      <Section title="Make more" hint="adds a job to the laptop's queue (tools/run_queue.py, one Kaggle job at a time)">
        <QueuePanel s={s} pending={pending} />
      </Section>
    </>
  );
}

function Metric({ label, value, tone, sub }: { label: string; value: ReactNode; tone?: "ok" | "warn" | "err"; sub?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-[3px] border border-rule bg-bg/40 px-2.5 py-1.5">
      <div className="label !text-[0.59375rem]">{label}</div>
      <div className={`tnum text-[0.875rem] font-semibold ${tone === "ok" ? "text-ok" : tone === "warn" ? "text-warn" : tone === "err" ? "text-err" : "text-ink"}`}>{value}</div>
      {sub != null && <div className="truncate text-[0.625rem] text-faint">{sub}</div>}
    </div>
  );
}

function VideoCard({ v, i, pending, applied }: { v: Video; i: number; pending: StudioAction[]; applied: NonNullable<NonNullable<Snapshot["actions"]>["applied"]> }) {
  const [thumb] = useMediaSrc(v.media.thumb?.key);
  const [preview, previewFail] = useMediaSrc(v.media.preview?.key);
  const [doc, setDoc] = useState<{ title: string; text: string; file: string } | null>(null);
  const [pic, setPic] = useState<{ key: string; title: string } | null>(null);
  const lufsTone = v.lufs == null ? undefined : v.lufs >= -16 && v.lufs <= -14 ? "ok" : "err";
  const peakTone = v.true_peak == null ? undefined : v.true_peak <= -1 ? "ok" : "warn";
  const sync = v.sync;
  const asr = v.asr;
  const p90 = asr?.caption_vs_speech_p90_abs_s;
  const syncTone = p90 == null ? undefined : p90 <= 0.25 ? "ok" : "warn";
  const spm = v.shots_per_min;
  return (
    <article id={`v-${v.name}`} className={`panel bracket rise-in scroll-mt-28 min-w-0 ${v.stage === "review" && v.approval?.status !== "needs_changes" ? "ember" : ""}`} style={{ ["--i" as string]: i }} data-testid="video-card" data-video={v.name}>
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-4 py-2">
        <h3 className="display text-[1.25rem] font-bold uppercase">{v.name}</h3>
        <Pill tone={stageTone(v.stage)}>{STAGE_NAME[v.stage]}</Pill>
        {v.approval?.status === "needs_changes" && <Pill tone="err">needs changes</Pill>}
        <span className="text-[0.6875rem] text-faint">
          {v.kind === "short" ? "Short" : "Long"} · {v.story} · {v.voice ?? "voice ?"}
        </span>
        <span className="ml-auto">
          <Stepper stage={v.stage} />
        </span>
      </header>
      {v.engine === "v3" && v.v3_stages && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 border-b border-rule px-4 py-1.5 text-[0.6875rem]" data-testid="v3-stages">
          <span className="label">engine v3</span>
          {["shots", "audio", "keys", "clips", "assemble", "post"].map((k) => {
            const s = v.v3_stages?.[k];
            const st = s?.status ?? "waiting";
            const tone = st === "done" ? "text-ok" : st === "failed" ? "text-err" : st === "waiting" ? "text-faint" : "text-sodium";
            return (
              <span key={k} className={`tnum ${tone}`}>
                {k} {st}{s?.run_min != null ? ` · ${s.run_min} min${s.gpu ? " GPU" : ""}` : ""}
              </span>
            );
          })}
        </div>
      )}
      {!v.mp4_bytes && !v.media.thumb ? (
        <div className="grid gap-3 p-4 sm:grid-cols-2" data-testid="card-compact">
          <div>
            <div className="tnum mb-1 flex justify-between text-[0.6875rem]"><span className="label">Narration lines</span><span>{v.progress.lines_done} / {v.progress.lines_total}</span></div>
            <Bar done={v.progress.lines_done} total={v.progress.lines_total} tone={v.progress.lines_done === v.progress.lines_total ? "ok" : "sodium"} label="narration lines" />
          </div>
          <div>
            <div className="tnum mb-1 flex justify-between text-[0.6875rem]"><span className="label">Images</span><span>{v.progress.images_done} / {v.progress.images_total}</span></div>
            <Bar done={v.progress.images_done} total={v.progress.images_total} tone={v.progress.images_done === v.progress.images_total ? "ok" : "sodium"} label="images" />
          </div>
          <div className="text-[0.6875rem] text-faint sm:col-span-2">Not rendered yet · {v.chapters ?? "?"} chapters · planned <Time iso={v.reached.script} /></div>
          {(v.media.masters || arr(v.samples).length > 0) && (
            <div className="flex flex-col gap-1 sm:col-span-2">
              {v.media.masters && <div><Btn tone="ghost" onClick={() => setPic({ key: v.media.masters!.key, title: "Master candidates" })}>Master candidates</Btn></div>}
              {arr(v.samples).map((x) => <AudioRow key={x.key} k={x.key} name={x.name} />)}
            </div>
          )}
          {v.errors.map((e) => <div key={e} className="text-[0.75rem] text-err sm:col-span-2">{e}</div>)}
        </div>
      ) : (
      <div className="grid gap-4 p-4 sm:grid-cols-[minmax(0,11.5rem)_minmax(0,1fr)]">
        <div className="mx-auto flex w-full max-w-[14rem] flex-col gap-2 sm:max-w-none">
          <div className="relative aspect-[9/16] w-full overflow-hidden rounded-[3px] border border-rule bg-raised">
            {preview ? (
              <video
                key={preview}
                controls
                playsInline
                preload="none"
                poster={thumb ?? undefined}
                src={preview}
                onError={() => preview.startsWith("/api/") && previewFail()}
                className="h-full w-full bg-black object-contain"
                data-testid="preview-player"
              />
            ) : (
              <div className="grid h-full place-items-center p-3 text-center text-[0.6875rem] text-faint">
                {v.mp4_bytes ? "480p preview is being made on the laptop" : "not rendered yet"}
              </div>
            )}
          </div>
          <div className="text-[0.625rem] text-faint">
            {v.media.preview ? `480p preview ${bytes(v.media.preview.bytes)}` : ""}
            {v.mp4_bytes ? ` · master ${bytes(v.mp4_bytes)}` : ""}
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <button type="button" className="overflow-hidden rounded-[3px] border border-rule text-left hover:border-sodium" onClick={() => v.media.thumb && setPic({ key: v.media.thumb.key, title: `${v.name} thumbnail` })}>
            <Img mkey={v.media.thumb?.key} alt={`${v.name} thumbnail`} className="aspect-video w-full" />
          </button>
          {v.seo?.title && <div className="text-[0.8125rem] font-semibold leading-snug text-ink">{v.seo.title}</div>}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Metric label="Duration" value={mmss(v.duration_s)} sub={`${num(v.duration_s, 1)} s · ${v.chapters ?? "?"} chapters`} />
            <Metric label="Shots / min" value={num(spm, 1)} sub={`${v.shots ?? "—"} shots`} tone={spm == null ? undefined : spm >= 8 ? "ok" : "warn"} />
            <Metric label="Loudness" value={v.lufs == null ? "—" : `${num(v.lufs, 1)} LUFS`} tone={lufsTone} sub="target -16 … -14" />
            <Metric label="True peak" value={v.true_peak == null ? "—" : `${signed(v.true_peak)} dBFS`} tone={peakTone} sub={`LRA ${num(v.lra, 1)}`} />
            <Metric
              label="Caption sync"
              value={p90 == null ? "—" : `p90 ${num(p90 * 1000)} ms`}
              tone={syncTone}
              sub={sync ? `${sync.speech_present ?? "?"}/${sync.chunks ?? "?"} chunks on speech · median ${num((asr?.caption_vs_speech_median_s ?? 0) * 1000)} ms` : "no QA yet"}
            />
            <Metric label="Whisper check" value={asr?.wer == null ? "—" : `WER ${num(asr.wer * 100, 1)}%`} tone={asr?.wer == null ? undefined : asr.wer <= 0.05 ? "ok" : "warn"} sub={asr ? `${asr.asr_words}/${asr.ref_words} words · ${v.srt_cues ?? "?"} SRT cues` : ""} />
          </div>
          <div className="text-[0.6875rem] text-faint">
            Render {v.render_s ? `${Math.round(v.render_s)} s` : "—"} · Whisper {v.verify_s ? `${Math.round(v.verify_s)} s` : "—"} on Kaggle CPU · updated <Time iso={v.updated_at} />
          </div>
          <div className="flex flex-wrap gap-1.5">
            {v.srt && <Btn tone="ghost" onClick={() => setDoc({ title: `${v.name}.srt`, text: v.srt!, file: `${v.name}.srt` })}>SRT</Btn>}
            {v.seo_md && <Btn tone="ghost" onClick={() => setDoc({ title: "SEO pack", text: v.seo_md!, file: `${v.name}_SEO_PACK.md` })}>SEO pack</Btn>}
            {v.media.sheet && <Btn tone="ghost" onClick={() => setPic({ key: v.media.sheet!.key, title: "Contact sheet" })}>Contact sheet</Btn>}
            {v.media.vsref && <Btn tone="ghost" onClick={() => setPic({ key: v.media.vsref!.key, title: "Versus reference" })}>Vs reference</Btn>}
            {v.media.masters && <Btn tone="ghost" onClick={() => setPic({ key: v.media.masters!.key, title: "Master candidates" })}>Masters</Btn>}
          </div>
          {arr(v.samples).length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="label">Voice samples</div>
              {arr(v.samples).map((x) => <AudioRow key={x.key} k={x.key} name={x.name} />)}
            </div>
          )}
          {(v.errors.length > 0 || v.warnings.length > 0) && (
            <ul className="flex flex-col gap-1 text-[0.75rem]">
              {v.errors.map((e) => (
                <li key={e} className="flex gap-2 text-err"><span className="pt-1.5"><Lamp tone="err" /></span>{e}</li>
              ))}
              {v.warnings.map((e) => (
                <li key={e} className="flex gap-2 text-warn"><span className="pt-1.5"><Lamp tone="warn" /></span>{e}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
      )}
      <Review v={v} pending={pending} applied={applied} />
      {doc && <TextDialog {...doc} onClose={() => setDoc(null)} />}
      {pic && <PicDialog mkey={pic.key} title={pic.title} onClose={() => setPic(null)} />}
    </article>
  );
}

function AudioRow({ k, name }: { k: string; name: string }) {
  const [src, fail] = useMediaSrc(k);
  return (
    <div className="flex flex-wrap items-center gap-2 text-[0.6875rem]">
      <span className="w-28 truncate text-muted" title={name}>{name}</span>
      {src && <audio controls preload="none" src={src} onError={() => src.startsWith("/api/") && fail()} className="h-8 min-w-0 flex-1" />}
    </div>
  );
}

function Review({ v, pending, applied }: { v: Video; pending: StudioAction[]; applied: NonNullable<NonNullable<Snapshot["actions"]>["applied"]> }) {
  const { act } = useStudio();
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ tone: "ok" | "warn" | "err"; text: string } | null>(null);
  const [url, setUrl] = useState("");
  const canReview = v.stage === "review" || v.stage === "approved" || v.stage === "published";
  const send = async (kind: StudioAction["kind"], extra: Partial<StudioAction> = {}) => {
    if (kind === "needs_changes" && !note.trim()) {
      setMsg({ tone: "warn", text: "Write what needs to change first." });
      return;
    }
    setBusy(true);
    setMsg(null);
    const r = await act({ kind, video: v.name, ...(note.trim() ? { note: note.trim() } : {}), ...extra });
    setBusy(false);
    if (!r.ok) setMsg({ tone: "err", text: `Not sent: ${r.error ?? "unknown error"}. Your note is still here; try again.` });
    else {
      setMsg({ tone: "ok", text: r.instant ? "Saved on the laptop." : "Saved on Vercel. The laptop applies it as soon as it is on." });
      setNote("");
      setUrl("");
    }
  };
  const last = applied[0];
  return (
    <footer className="border-t border-rule px-4 py-3" data-testid="review">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[0.75rem]">
        <span className="label !text-ink">Owner&apos;s review</span>
        {v.approval ? (
          <span className={v.approval.status === "needs_changes" ? "text-err" : v.approval.status === "pending" ? "text-muted" : "text-ok"} data-testid="approval-status">
            {v.approval.status.replace("_", " ")} · <Time iso={v.approval.at} />
            {v.approval.note ? ` · “${v.approval.note}”` : ""}
            {v.approval.url ? (
              <a className="ml-1 text-sodium hover:underline" href={v.approval.url} target="_blank" rel="noopener noreferrer">link</a>
            ) : null}
          </span>
        ) : (
          <span className="text-faint" data-testid="approval-status">not reviewed yet</span>
        )}
        {pending.length > 0 && <span className="text-warn" data-testid="pending-note">{pending.length} change{pending.length > 1 ? "s" : ""} on the way to the laptop</span>}
        {last && <span className="text-faint">last applied <Time iso={last.applied_at} /> via {last.via}{last.result !== "ok" ? ` (${last.result}: ${last.msg})` : ""}</span>}
      </div>
      {canReview ? (
        <div className="flex flex-col gap-2">
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            maxLength={2000}
            placeholder="Note (needed for “needs changes”, optional for approve)"
            aria-label={`Note for ${v.name}`}
            className="w-full resize-y rounded-[3px] border border-rule-strong bg-bg px-2.5 py-2 text-[0.8125rem] text-ink outline-none placeholder:text-faint focus:border-sodium"
          />
          <div className="flex flex-wrap items-center gap-2">
            <Btn tone="primary" disabled={busy} onClick={() => send("approve")} testid="btn-approve">Approve</Btn>
            <Btn tone="warn" disabled={busy} onClick={() => send("needs_changes")} testid="btn-changes">Needs changes</Btn>
            {v.approval && v.approval.status !== "pending" && <Btn tone="ghost" disabled={busy} onClick={() => send("reset")}>Undo</Btn>}
            <Btn tone="ghost" disabled={busy} onClick={() => confirm(`Re-render ${v.name} with the current engine? It goes into the laptop's queue (Kaggle CPU kernel if every asset is cached).`) && send("rerender")} testid="btn-rerender">Re-render</Btn>
            {(v.stage === "approved" || v.stage === "published") && (
              <span className="flex flex-wrap items-center gap-1.5">
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://youtube.com/shorts/…"
                  aria-label="Published link"
                  className="w-56 rounded-[3px] border border-rule-strong bg-bg px-2 py-1.5 text-[0.75rem] outline-none focus:border-sodium"
                />
                <Btn disabled={busy} onClick={() => send("published", url.trim() ? { url: url.trim() } : {})}>Mark published</Btn>
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2 text-[0.75rem] text-faint">
          Review opens once the video is rendered and checked.
          {v.stage !== "script" && <Btn tone="ghost" disabled={busy} onClick={() => confirm(`Queue ${v.name} again?`) && send("rerender")}>Run again</Btn>}
        </div>
      )}
      {msg && <p role="status" className={`slide-in mt-2 text-[0.75rem] ${msg.tone === "ok" ? "text-ok" : msg.tone === "warn" ? "text-warn" : "text-err"}`} data-testid="review-msg">{msg.text}</p>}
    </footer>
  );
}

function QueuePanel({ s, pending }: { s: Snapshot; pending: Map<string, StudioAction> }) {
  const { act } = useStudio();
  const stories = arr(s.stories).filter((x) => !x.error);
  const [story, setStory] = useState(stories[0]?.slug ?? "");
  const [mode, setMode] = useState<"short" | "full">("short");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const jobs = arr(s.queue?.jobs).slice().reverse();
  const queuedHere = [...pending.values()].filter((a) => a.kind === "queue_story" || a.kind === "rerender");
  const submit = async () => {
    if (!story || !confirm(`Queue a ${mode === "short" ? "Short" : "full-length video"} of “${story}”? It uses Kaggle GPU time if images or narration are missing.`)) return;
    setBusy(true);
    const r = await act({ kind: "queue_story", story, mode });
    setBusy(false);
    setMsg(r.ok ? (r.instant ? "Queued on the laptop." : "Saved on Vercel; the laptop queues it when it is on.") : `Not sent: ${r.error}`);
  };
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <Panel title="Queue a story" bodyClass="p-4 flex flex-col gap-3">
        <label className="label" htmlFor="q-story">Story</label>
        <select id="q-story" value={story} onChange={(e) => setStory(e.target.value)} className="rounded-[3px] border border-rule-strong bg-bg px-2 py-2 text-[0.8125rem] outline-none focus:border-sodium">
          {stories.map((x) => (
            <option key={x.slug} value={x.slug}>
              {x.slug} · {x.chapters} ch · {x.lines} lines
            </option>
          ))}
        </select>
        <Chips value={mode} options={["short", "full"] as const} onChange={setMode} label="Length" names={{ short: "Short (Shorts pace)", full: "Full length" }} />
        <Btn tone="primary" onClick={submit} disabled={busy || !story} testid="btn-queue">Queue it</Btn>
        {msg && <p role="status" className="text-[0.75rem] text-muted">{msg}</p>}
        <p className="text-[0.6875rem] leading-relaxed text-faint">
          Runs <code>heartwick.py make --auto-pick</code> on the laptop, one job at a time. New characters get a master made in the same Kaggle job.
        </p>
      </Panel>
      <Panel title="Laptop queue" meta={s.queue?.runner?.running ? `runner alive (pid ${arr(s.queue?.runner?.pids).join(", ")})` : "runner idle"} bodyClass="p-0">
        {jobs.length === 0 && queuedHere.length === 0 ? (
          <div className="p-4 text-[0.75rem] text-muted">
            No dashboard jobs yet.{s.queue?.current ? <> The original queue runner is working through its own list: <code className="text-sodium">{s.queue.current.text}</code></> : null}
          </div>
        ) : (
          <ul className="divide-y divide-rule/60">
            {queuedHere.filter((a) => !jobs.some((j) => j.id === a.id)).map((a) => (
              <li key={a.id} className="slide-in flex flex-wrap items-center gap-2 px-4 py-2 text-[0.75rem]">
                <Pill tone="warn">on its way</Pill>
                <span>{a.kind === "queue_story" ? `Queue ${a.mode} video of ${a.story}` : `Re-render ${a.video}`}</span>
              </li>
            ))}
            {jobs.map((j) => (
              <li key={j.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[0.75rem]">
                <Pill tone={j.status === "done" ? "ok" : j.status === "failed" || j.status === "interrupted" ? "err" : j.status === "running" ? "sodium" : "warn"}>{j.status}</Pill>
                <span className="min-w-0 flex-1 truncate">{j.label}</span>
                <span className="text-faint">
                  <Time iso={j.finished_at ?? j.started_at ?? j.added_at} />
                  {j.rc != null && j.rc !== 0 ? ` · rc ${j.rc}` : ""}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}

function Dialog({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    d?.showModal();
    return () => d?.close();
  }, []);
  return (
    <dialog ref={ref} onClose={onClose} onClick={(e) => e.target === ref.current && onClose()} className={`m-auto max-h-[90dvh] rounded-[4px] border border-rule-strong bg-panel p-0 text-ink backdrop:bg-black/70`} style={{ width: `min(${wide ? "72rem" : "48rem"}, calc(100vw - 2rem))` }}>
      <div className="flex items-center gap-3 border-b border-rule px-4 py-2">
        <span className="label !text-ink">{title}</span>
        <button type="button" onClick={onClose} className="ml-auto text-[0.75rem] text-muted hover:text-ink">close</button>
      </div>
      {children}
    </dialog>
  );
}

function TextDialog({ title, text, file, onClose }: { title: string; text: string; file: string; onClose: () => void }) {
  const download = () => {
    const u = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = u;
    a.download = file;
    a.click();
    setTimeout(() => URL.revokeObjectURL(u), 1000);
  };
  return (
    <Dialog title={title} onClose={onClose}>
      <div className="flex gap-2 px-4 pt-3">
        <Btn onClick={() => navigator.clipboard?.writeText(text)}>Copy</Btn>
        <Btn onClick={download}>Download</Btn>
      </div>
      <pre className="scroll-thin max-h-[70dvh] overflow-auto whitespace-pre-wrap break-words px-4 py-3 text-[0.75rem] leading-relaxed">{text}</pre>
    </Dialog>
  );
}

function PicDialog({ mkey, title, onClose }: { mkey: string; title: string; onClose: () => void }) {
  return (
    <Dialog title={title} onClose={onClose} wide>
      <div className="scroll-thin max-h-[82dvh] overflow-auto p-3">
        <Img mkey={mkey} alt={title} className="h-auto w-full object-contain" />
      </div>
    </Dialog>
  );
}
