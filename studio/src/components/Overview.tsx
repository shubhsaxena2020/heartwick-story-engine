"use client";

import { arr, dur, minutes, toMs } from "@/lib/format";
import { STAGES, type KaggleJob, type Snapshot, type Stage, type StudioAction, type Video } from "@/lib/types";
import { Bar, Empty, Lamp, Num, Panel, Pill, Ring, Section, Split, Stat, Time, useNow, useStudio, type Tone } from "./ui";

export const STAGE_NAME: Record<Stage, string> = {
  script: "Script", narration: "Narration", images: "Images", render: "Render", review: "Review", approved: "Approved", published: "Published",
};
export const STAGE_HINT: Record<Stage, string> = {
  script: "story planned, narration not complete",
  narration: "every line voiced, images still missing",
  images: "all narration and images made",
  render: "mp4 rendered, QA report missing",
  review: "rendered with QA: waiting for the channel owner",
  approved: "approved by the channel owner",
  published: "uploaded by the channel owner",
};

export function stageTone(s: Stage): Tone {
  return s === "review" ? "warn" : s === "approved" || s === "published" ? "ok" : s === "render" ? "sodium" : "faint";
}

export function Overview({ s, pending }: { s: Snapshot; pending: Map<string, StudioAction> }) {
  const videos = arr(s.videos);
  const counts = s.counts ?? {};
  const jobs = arr(s.kaggle?.jobs);
  const active = jobs.filter((j) => j.state === "running" || j.state === "queued");
  const q = s.kaggle?.quota;
  const errs = arr(s.events).filter((e) => e.level === "err");
  const review = videos.filter((v) => v.stage === "review");
  return (
    <>
      <Section title="Right now" hint="from the laptop's real files">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="panel bracket p-4 rise-in" style={{ ["--i" as string]: 0 }}>
            <Stat label="Waiting for review" value={counts.review ?? 0} tone={(counts.review ?? 0) ? "warn" : "ink"} sub={`${counts.approved ?? 0} approved · ${counts.published ?? 0} published`} testid="kpi-review" />
          </div>
          <div className="panel bracket p-4 rise-in" style={{ ["--i" as string]: 1 }}>
            <Stat label="Videos in the pipeline" value={videos.length} sub={`${arr(s.stories).length} stories · ${(counts.script ?? 0) + (counts.narration ?? 0) + (counts.images ?? 0)} still being made`} testid="kpi-videos" />
          </div>
          <div className="panel bracket flex items-center gap-4 p-4 rise-in" style={{ ["--i" as string]: 2 }}>
            <Ring value={q?.used_h ?? 0} max={q?.total_h ?? s.kaggle?.quota_total_h ?? 30} size={68} label="Kaggle GPU hours used this week" tone={(q?.used_h ?? 0) > 24 ? "warn" : "sodium"}>
              <span className="tnum text-[0.6875rem] font-semibold">{q ? `${Math.round((q.used_h / q.total_h) * 100)}%` : "—"}</span>
            </Ring>
            <div className="min-w-0">
              <div className="label">GPU this week</div>
              <div className="display mt-1 text-[2rem] font-bold">
                <Num value={q?.used_h ?? null} digits={2} testid="kpi-gpu" />
                <span className="ml-1 font-mono text-[0.75rem] font-normal text-muted">/ {q?.total_h ?? 30} h</span>
              </div>
              <div className="text-[0.6875rem] text-faint">{q ? <>resets <Time iso={q.refresh_at} /></> : "quota not read yet"}</div>
            </div>
          </div>
          <div className="panel bracket p-4 rise-in" style={{ ["--i" as string]: 3 }}>
            <Stat label="Errors" value={errs.length} tone={errs.length ? "err" : "ok"} sub={errs[0] ? errs[0].text.slice(0, 70) : "nothing failing"} testid="kpi-errors" />
          </div>
        </div>
        <NowDoing s={s} active={active} />
      </Section>

      <Section title="Story pipeline" hint="script → narration → images → render → review → approved → published">
        <PipelineBoard videos={videos} counts={counts} />
        <Panel title="Per video" meta="lines voiced and images made, from cache/<story>/assets" bodyClass="p-0">
          <StoryTable videos={videos} pending={pending} />
        </Panel>
      </Section>

      <Split>
        <Section title="Needs attention">
          <Attention review={review} videos={videos} s={s} />
        </Section>
        <Section title="Activity">
          <Panel title="Latest" meta={`${arr(s.events).length} events`} bodyClass="p-0">
            <EventList events={arr(s.events).slice(0, 14)} />
          </Panel>
        </Section>
      </Split>
    </>
  );
}

function NowDoing({ s, active }: { s: Snapshot; active: KaggleJob[] }) {
  const now = useNow(1000);
  const cur = s.queue?.current;
  const runner = s.queue?.runner;
  return (
    <div className="panel bracket grid gap-3 p-4 md:grid-cols-[auto_minmax(0,1fr)]" data-testid="now-doing">
      <div className="label self-center">Now doing</div>
      <div className="flex min-w-0 flex-col gap-2">
        {cur ? (
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <Lamp tone="sodium" pulse />
            <span className="text-ink">Queue runner:</span>
            <code className="truncate rounded-[2px] bg-raised px-1.5 py-0.5 text-[0.75rem] text-sodium">{cur.text}</code>
            <span className="tnum text-faint">for {dur(now - (toMs(cur.since) ?? now))}</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-muted">
            <Lamp tone="faint" />
            {runner?.running ? "Queue runner is between jobs." : "Queue runner is idle (nothing pending)."}
          </div>
        )}
        {active.map((j) => (
          <div key={j.kernel + j.pushed_at} className="flex min-w-0 flex-wrap items-center gap-2 text-[0.75rem]">
            <Pill tone={j.state === "running" ? "sodium" : "warn"}>{j.state}</Pill>
            <span className="truncate">{j.kernel}</span>
            <span className="text-faint">{j.gpu ? "GPU" : "CPU"} · pushed <Time iso={j.pushed_at} /></span>
            <span className="tnum text-faint">
              {j.state === "queued" ? `waiting ${minutes(j.wait_min)}` : `running ${minutes(j.run_min)}${j.wait_min ? ` after ${minutes(j.wait_min)} queued` : ""}`}
            </span>
            <div className="w-28">
              <Bar done={1} total={1} tone={j.state === "running" ? "sodium" : "warn"} running label="job in progress" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PipelineBoard({ videos, counts }: { videos: Video[]; counts: Partial<Record<Stage, number>> }) {
  const { go } = useStudio();
  return (
    <div className="panel bracket overflow-hidden p-4" data-testid="pipeline-board">
      <div className="relative">
        <svg aria-hidden className="pointer-events-none absolute left-0 right-0 top-[1.15rem] hidden h-2 w-full md:block" preserveAspectRatio="none" viewBox="0 0 100 2">
          <line x1="4" y1="1" x2="96" y2="1" stroke="var(--color-rule-strong)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          <line x1="4" y1="1" x2="96" y2="1" stroke="var(--color-sodium-deep)" strokeWidth="2" strokeDasharray="4 4" className="flow-dash" vectorEffect="non-scaling-stroke" />
        </svg>
        <ol className="relative grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-7">
          {STAGES.map((st, i) => {
            const n = counts[st] ?? 0;
            const here = videos.filter((v) => v.stage === st);
            const tone = stageTone(st);
            return (
              <li key={st} className="rise-in flex min-w-0 flex-col items-start gap-2 md:items-center" style={{ ["--i" as string]: i }} data-stage={st}>
                <div
                  className={`grid size-[2.3rem] place-items-center rounded-full border-2 bg-panel text-[0.8125rem] font-bold ${
                    n ? (tone === "ok" ? "border-ok text-ok" : tone === "warn" ? "border-warn text-warn ember" : "border-sodium text-sodium") : "border-rule-strong text-faint"
                  }`}
                  title={STAGE_HINT[st]}
                >
                  <Num value={n} />
                </div>
                <div className="label !text-ink">{STAGE_NAME[st]}</div>
                <ul className="flex w-full min-w-0 flex-col gap-1">
                  {here.map((v) => (
                    <li key={v.name}>
                      <button
                        type="button"
                        onClick={() => go("videos", `v-${v.name}`)}
                        className="w-full truncate rounded-[2px] border border-rule bg-raised px-1.5 py-0.5 text-left text-[0.6875rem] text-muted hover:border-sodium hover:text-ink md:text-center"
                        title={v.title ?? v.name}
                      >
                        {v.name}
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

export function Stepper({ stage }: { stage: Stage }) {
  const at = STAGES.indexOf(stage);
  return (
    <div className="flex items-center gap-[3px]" role="img" aria-label={`stage ${STAGE_NAME[stage]} (${at + 1} of ${STAGES.length})`}>
      {STAGES.map((st, i) => (
        <span
          key={st}
          title={STAGE_NAME[st]}
          className={`h-[0.4375rem] w-[0.9rem] rounded-[1px] transition-colors duration-700 ${i < at ? "bg-sodium-deep" : i === at ? (stageTone(st) === "ok" ? "bg-ok" : stageTone(st) === "warn" ? "bg-warn" : "bg-sodium") : "bg-rule-strong/70"}`}
          style={{ transitionDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

function StoryTable({ videos, pending }: { videos: Video[]; pending: Map<string, StudioAction> }) {
  const { go } = useStudio();
  if (!videos.length) return <div className="p-4"><Empty>No video has been planned yet.</Empty></div>;
  const waiting = new Set([...pending.values()].map((a) => a.video).filter(Boolean));
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full min-w-[46rem] text-left text-[0.75rem]">
        <thead className="text-faint">
          <tr className="border-b border-rule">
            <th className="px-4 py-2 font-semibold">Video</th>
            <th className="px-2 py-2 font-semibold">Stage</th>
            <th className="w-[22%] px-2 py-2 font-semibold">Narration lines</th>
            <th className="w-[22%] px-2 py-2 font-semibold">Images</th>
            <th className="px-4 py-2 text-right font-semibold">Updated</th>
          </tr>
        </thead>
        <tbody>
          {videos.map((v, i) => {
            const p = v.progress;
            return (
              <tr key={v.name} className="rise-in border-b border-rule/60 hover:bg-raised/50" style={{ ["--i" as string]: i }} data-video={v.name}>
                <td className="px-4 py-2">
                  <button type="button" className="text-left font-semibold text-ink hover:text-sodium" onClick={() => go("videos", `v-${v.name}`)}>
                    {v.name}
                  </button>
                  <div className="text-[0.6875rem] text-faint">
                    {v.story} · {v.kind === "short" ? "Short" : "Long"} · {v.chapters ?? "?"} ch
                    {waiting.has(v.name) && <span className="ml-1 text-warn">· your change is on its way</span>}
                  </div>
                </td>
                <td className="px-2 py-2">
                  <div className="flex flex-col gap-1">
                    <Stepper stage={v.stage} />
                    <span className={`text-[0.6875rem] font-semibold uppercase tracking-[0.08em] ${v.stage === "review" ? "text-warn" : v.stage === "approved" || v.stage === "published" ? "text-ok" : "text-muted"}`}>
                      {STAGE_NAME[v.stage]}
                      {v.approval?.status === "needs_changes" && <span className="ml-1 text-err">· needs changes</span>}
                    </span>
                  </div>
                </td>
                <td className="px-2 py-2">
                  <div className="tnum mb-1 text-[0.6875rem]">
                    <span className={p.lines_done === p.lines_total ? "text-ok" : "text-ink"}>{p.lines_done}</span>
                    <span className="text-faint"> / {p.lines_total}</span>
                  </div>
                  <Bar done={p.lines_done} total={p.lines_total} tone={p.lines_done === p.lines_total ? "ok" : "sodium"} label={`${v.name} narration lines`} />
                </td>
                <td className="px-2 py-2">
                  <div className="tnum mb-1 text-[0.6875rem]">
                    <span className={p.images_done === p.images_total ? "text-ok" : "text-ink"}>{p.images_done}</span>
                    <span className="text-faint"> / {p.images_total}</span>
                    {p.images_done < p.images_total && STAGES.indexOf(v.stage) >= 3 && <span className="ml-1 text-faint">(rest from masters)</span>}
                  </div>
                  <Bar done={p.images_done} total={p.images_total} tone={p.images_done === p.images_total ? "ok" : "sodium"} label={`${v.name} images`} />
                </td>
                <td className="px-4 py-2 text-right text-faint">
                  <Time iso={v.updated_at} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Attention({ review, videos, s }: { review: Video[]; videos: Video[]; s: Snapshot }) {
  const { go } = useStudio();
  const changes = videos.filter((v) => v.approval?.status === "needs_changes");
  const errs = arr(s.events).filter((e) => e.level === "err").slice(0, 4);
  if (!review.length && !changes.length && !errs.length) return <Empty>Nothing needs attention.</Empty>;
  return (
    <div className="flex flex-col gap-3">
      {review.length > 0 && (
        <Panel title="Waiting for the owner's review" meta={`${review.length}`} bodyClass="p-3 flex flex-col gap-2">
          {review.map((v) => (
            <button key={v.name} type="button" onClick={() => go("videos", `v-${v.name}`)} className="flex items-center justify-between gap-3 rounded-[3px] border border-rule bg-raised px-3 py-2 text-left hover:border-warn">
              <span className="min-w-0">
                <span className="block truncate font-semibold text-ink">{v.name}</span>
                <span className="block truncate text-[0.6875rem] text-faint">{v.seo?.title ?? v.title}</span>
              </span>
              <span className="text-[0.6875rem] font-bold uppercase tracking-[0.1em] text-warn">Review →</span>
            </button>
          ))}
        </Panel>
      )}
      {changes.length > 0 && (
        <Panel title="Needs changes" meta={`${changes.length}`} bodyClass="p-3 flex flex-col gap-2">
          {changes.map((v) => (
            <div key={v.name} className="rounded-[3px] border border-err/40 bg-err/5 px-3 py-2">
              <div className="font-semibold text-ink">{v.name}</div>
              <div className="text-[0.75rem] text-muted">&ldquo;{v.approval?.note}&rdquo; · <Time iso={v.approval?.at} /></div>
            </div>
          ))}
        </Panel>
      )}
      {errs.length > 0 && (
        <Panel title="Recent errors" right={<button type="button" className="label hover:text-ink" onClick={() => go("errors")}>all →</button>} bodyClass="p-0">
          <EventList events={errs} />
        </Panel>
      )}
    </div>
  );
}

export function EventList({ events }: { events: NonNullable<Snapshot["events"]> }) {
  if (!events.length) return <div className="p-4 text-muted">Nothing yet.</div>;
  return (
    <ul className="divide-y divide-rule/60">
      {events.map((e, i) => (
        <li key={`${e.at}-${i}-${e.text.slice(0, 20)}`} className="slide-in flex gap-3 px-4 py-2 text-[0.75rem]" style={{ animationDelay: `${Math.min(i, 10) * 30}ms` }}>
          <span className="pt-1.5">
            <Lamp tone={e.level === "err" ? "err" : e.level === "warn" ? "warn" : "faint"} />
          </span>
          <span className="min-w-0 flex-1">
            <span className={`break-words ${e.level === "err" ? "text-err" : e.level === "warn" ? "text-warn" : "text-ink"}`}>{e.text}</span>
            <span className="block text-[0.6875rem] text-faint">
              <Time iso={e.at} /> · {e.source}
              {e.video ? ` · ${e.video}` : ""}
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}
