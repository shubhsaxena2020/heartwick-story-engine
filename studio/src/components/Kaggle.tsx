"use client";

import { arr, dayLabel, minutes, num, toMs } from "@/lib/format";
import type { KaggleJob, Snapshot } from "@/lib/types";
import { Bars, HBars, Legend, type Series } from "./chartkit";
import { Bar, Empty, Lamp, Num, Panel, Pill, Ring, Section, Split, Time, TickMeter, useNow, type Tone } from "./ui";

export const GPU_CPU: Series[] = [
  { key: "gpu_min", label: "GPU minutes", color: "var(--color-gpu)" },
  { key: "cpu_min", label: "CPU minutes (free)", color: "var(--color-cpu)" },
];

const STATE_TONE: Record<KaggleJob["state"], Tone> = {
  queued: "warn", running: "sodium", complete: "ok", error: "err", cancelled: "faint", unknown: "faint", superseded: "faint",
};

export function KagglePage({ s }: { s: Snapshot }) {
  const k = s.kaggle ?? {};
  const jobs = arr(k.jobs);
  const q = k.quota;
  const now = useNow(1000);
  const running = jobs.filter((j) => j.state === "running");
  const queued = jobs.filter((j) => j.state === "queued");
  const done = jobs.filter((j) => j.state === "complete");
  const resetMs = toMs(q?.refresh_at);
  const total = q?.total_h ?? k.quota_total_h ?? 30;
  return (
    <>
      <Section title="GPU quota" hint="Kaggle's own numbers (`kaggle quota`), read by the laptop every 30 min">
        <Split side="26rem">
          <Panel title="This week" meta={q ? <>checked <Time iso={q.checked_at} /></> : "not read yet"} bodyClass="p-4">
            <div className="flex flex-wrap items-center gap-6">
              <Ring value={q?.used_h ?? 0} max={total} size={112} label="GPU hours used" tone={(q?.used_h ?? 0) / total > 0.8 ? "warn" : "sodium"}>
                <div className="text-center">
                  <div className="display text-[1.75rem] font-bold"><Num value={q?.used_h ?? null} digits={2} /></div>
                  <div className="label !text-[0.5625rem]">of {total} h</div>
                </div>
              </Ring>
              <div className="flex min-w-[14rem] flex-1 flex-col gap-3">
                <TickMeter value={q?.used_h ?? null} max={total} limit={total * 0.9} label="GPU hours used of the weekly quota" />
                <div className="grid grid-cols-3 gap-3 text-[0.75rem]">
                  <div><div className="label">Used</div><div className="tnum font-semibold">{q ? `${num(q.used_h, 2)} h` : "—"}</div></div>
                  <div><div className="label">Left</div><div className="tnum font-semibold text-ok">{q ? `${num(q.remaining_h, 2)} h` : "—"}</div></div>
                  <div><div className="label">Resets</div><div className="tnum font-semibold">{resetMs ? <Time iso={q!.refresh_at} /> : "—"}</div><div className="text-[0.625rem] text-faint">{resetMs ? `in ${Math.max(0, Math.round((resetMs - now) / 3600_000))} h` : ""}</div></div>
                </div>
                <p className="text-[0.6875rem] leading-relaxed text-faint">
                  Local log (runs timed from RUNNING): {minutes(k.gpu_min_week_local)} this quota week. Kaggle bills whole sessions, including installs and model loads,
                  so its number is higher. The real limit is the T4 queue, not the quota.
                </p>
              </div>
            </div>
          </Panel>
          <Panel title="Jobs" bodyClass="p-4 grid grid-cols-3 gap-3">
            <div><div className="label">Running</div><div className="display text-[2rem] font-bold text-sodium"><Num value={running.length} /></div></div>
            <div><div className="label">Queued</div><div className="display text-[2rem] font-bold text-warn"><Num value={queued.length} /></div></div>
            <div><div className="label">Complete</div><div className="display text-[2rem] font-bold text-ok"><Num value={done.length} /></div></div>
            <div className="col-span-3 text-[0.6875rem] text-faint">
              Status of unfinished jobs is read every 5 min{k.poll?.checked_at ? <> (last <Time iso={k.poll.checked_at} />)</> : ""}. Only 2 GPU sessions at a time (queued ones count).
              {k.poll?.error ? <span className="text-err"> · {k.poll.error}</span> : null}
            </div>
          </Panel>
        </Split>
      </Section>

      <Section title="Jobs" hint="one row per push, from kaggle/usage.jsonl + observed status changes">
        <Panel title="All Kaggle jobs" meta={`${jobs.length}`} bodyClass="p-0">
          {jobs.length ? <JobTable jobs={jobs} now={now} /> : <div className="p-4"><Empty>No Kaggle job has been pushed yet.</Empty></div>}
        </Panel>
      </Section>

      <Section title="Where the minutes went">
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Minutes per day" right={<Legend series={GPU_CPU} />} bodyClass="p-4">
            <Bars rows={arr(k.by_day).map((d) => ({ label: dayLabel(d.date), values: { gpu_min: d.gpu_min, cpu_min: d.cpu_min } }))} series={GPU_CPU} unit=" min" />
          </Panel>
          <Panel title="Minutes per video" right={<Legend series={GPU_CPU} />} bodyClass="p-4">
            <HBars rows={arr(k.by_video).map((v) => ({ label: v.video, values: { gpu_min: v.gpu_min, cpu_min: v.cpu_min } }))} series={GPU_CPU} unit="m" />
          </Panel>
        </div>
      </Section>

      <Section title="Queue runner log" hint="kaggle/queue_runner.log, newest last">
        <Panel title="Last lines" meta={s.queue?.runner?.running ? "runner alive" : "runner not running"} bodyClass="p-0">
          <ul className="scroll-thin max-h-[26rem] divide-y divide-rule/60 overflow-y-auto font-mono text-[0.6875rem]">
            {arr(s.queue?.log_tail).slice(-40).map((l, i) => (
              <li key={i} className={`flex gap-3 px-4 py-1.5 ${l.level === "err" ? "text-err" : l.level === "warn" ? "text-warn" : "text-muted"}`}>
                <span className="shrink-0 text-faint"><Time iso={l.at} /></span>
                <span className="break-all">{l.text}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </Section>
    </>
  );
}

function JobTable({ jobs, now }: { jobs: KaggleJob[]; now: number }) {
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full min-w-[54rem] text-left text-[0.75rem]">
        <thead className="text-faint">
          <tr className="border-b border-rule">
            <th className="px-4 py-2 font-semibold">Kernel</th>
            <th className="px-2 py-2 font-semibold">For</th>
            <th className="px-2 py-2 font-semibold">State</th>
            <th className="px-2 py-2 font-semibold">Pushed</th>
            <th className="px-2 py-2 text-right font-semibold">Waited</th>
            <th className="px-2 py-2 text-right font-semibold">Ran</th>
            <th className="px-2 py-2 font-semibold">Made</th>
            <th className="w-[18%] px-4 py-2 font-semibold">Worker stages</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j, i) => {
            const st = Object.entries(j.worker_stages ?? {}).filter(([, v]) => v > 0.5);
            const tot = st.reduce((a, [, v]) => a + v, 0);
            const live = j.state === "running" || j.state === "queued";
            return (
              <tr key={j.kernel + j.pushed_at} className="rise-in border-b border-rule/60 align-top" style={{ ["--i" as string]: Math.min(i, 12) }} data-kernel={j.kernel}>
                <td className="px-4 py-2">
                  <div className="font-semibold text-ink">{j.kernel}</div>
                  <div className="text-[0.6875rem] text-faint">{j.gpu ? "GPU T4" : "CPU (no quota)"}</div>
                </td>
                <td className="px-2 py-2 text-muted">{j.video ?? (j.story ? `${j.story} (story)` : "—")}</td>
                <td className="px-2 py-2">
                  <span className="flex items-center gap-1.5">
                    {live && <Lamp tone={STATE_TONE[j.state]} pulse />}
                    <Pill tone={STATE_TONE[j.state]} title={j.note ?? j.live_status ?? undefined}>{j.state}</Pill>
                  </span>
                  {j.worker_errors?.length ? <div className="mt-1 max-w-[16rem] text-[0.6875rem] text-err">{j.worker_errors.join("; ")}</div> : null}
                </td>
                <td className="px-2 py-2 text-muted"><Time iso={j.pushed_at} /></td>
                <td className="tnum px-2 py-2 text-right">{minutes(j.wait_min)}</td>
                <td className="tnum px-2 py-2 text-right">
                  {minutes(j.run_min)}
                  {j.state === "running" && <div className="mt-1 w-16"><Bar done={1} total={1} running label="running" /></div>}
                </td>
                <td className="px-2 py-2 text-[0.6875rem] text-muted">
                  {j.images ? `${j.images} img` : ""}
                  {j.utts ? ` ${j.utts} lines` : ""}
                  {j.render ? " render" : ""}
                </td>
                <td className="px-4 py-2">
                  {tot > 0 ? (
                    <div className="flex h-2.5 overflow-hidden rounded-[1px]" title={st.map(([k, v]) => `${k} ${Math.round(v)} s`).join(" · ")}>
                      {st.map(([key, v], n) => (
                        <span key={key} className="grow-x h-full" style={{ width: `${(v / tot) * 100}%`, background: ["var(--color-sodium)", "var(--color-cpu)", "var(--color-ok)", "var(--color-warn)", "var(--color-live)", "var(--color-faint)"][n % 6], ["--i" as string]: n }} />
                      ))}
                    </div>
                  ) : (
                    <span className="text-faint">{live ? `${Math.round((now - (toMs(j.pushed_at) ?? now)) / 60_000)} min since push` : "—"}</span>
                  )}
                  {tot > 0 && <div className="mt-1 truncate text-[0.625rem] text-faint">{st.map(([k, v]) => `${k} ${Math.round(v)}s`).join(" · ")}</div>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
