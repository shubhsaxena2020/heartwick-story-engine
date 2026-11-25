"use client";

import { arr, dayLabel, toMs } from "@/lib/format";
import { STAGES, type Snapshot, type Stage } from "@/lib/types";
import { Bars, Legend, Line, Steps, type Series } from "./chartkit";
import { GPU_CPU } from "./Kaggle";
import { STAGE_NAME } from "./Overview";
import { Panel, Section } from "./ui";

const STAGE_COLOR: Record<Stage, string> = {
  script: "#8c8176", narration: "#6fb7d9", images: "#a970ff", render: "#ff8a3d", review: "#f5c04a", approved: "#58d68d", published: "#ff3d7f",
};

export function Charts({ s }: { s: Snapshot }) {
  const videos = arr(s.videos);
  const byTime = [...videos].sort((a, b) => (toMs(a.reached.render ?? a.updated_at) ?? 0) - (toMs(b.reached.render ?? b.updated_at) ?? 0));
  const rendered = byTime.filter((v) => v.duration_s != null);
  const stageSeries = STAGES.map((st) => ({
    key: st,
    label: STAGE_NAME[st],
    color: STAGE_COLOR[st],
    times: videos.map((v) => toMs(v.reached[st])).filter((t): t is number => t != null),
  })).filter((x) => x.times.length);
  const renderSeries: Series[] = [
    { key: "render", label: "Render (s)", color: "var(--color-sodium)" },
    { key: "verify", label: "Whisper check (s)", color: "var(--color-cpu)" },
  ];
  return (
    <>
      <Section title="Pipeline over time" hint="when each video reached each stage (from file times on the laptop)">
        <Panel title="Videos per stage" right={<Legend series={stageSeries} />} bodyClass="p-4">
          <Steps series={stageSeries} />
        </Panel>
      </Section>
      <Section title="Rendering and GPU">
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Render time per video" meta="Kaggle CPU kernel, worker_log.json" right={<Legend series={renderSeries} />} bodyClass="p-4">
            <Bars
              rows={rendered.filter((v) => v.render_s).map((v) => ({ label: v.name, values: { render: v.render_s ?? 0, verify: v.verify_s ?? 0 } }))}
              series={renderSeries}
              unit=" s"
              empty="No render has finished yet."
            />
          </Panel>
          <Panel title="Kaggle minutes per day" right={<Legend series={GPU_CPU} />} bodyClass="p-4">
            <Bars rows={arr(s.kaggle?.by_day).map((d) => ({ label: dayLabel(d.date), values: { gpu_min: d.gpu_min, cpu_min: d.cpu_min } }))} series={GPU_CPU} unit=" min" />
          </Panel>
        </div>
      </Section>
      <Section title="QA trend" hint="per rendered video, oldest first · green band = target">
        <div className="grid gap-4 lg:grid-cols-3">
          <Panel title="Loudness" meta="integrated LUFS, target -16 … -14" bodyClass="p-4">
            <Line points={rendered.map((v) => ({ label: v.name, value: v.lufs ?? null }))} band={[-16, -14]} unit=" LUFS" min={-18} max={-12} fmt={(v) => v.toFixed(1)} />
          </Panel>
          <Panel title="Caption sync" meta="caption vs speech p90, ms (target ≤ 250)" bodyClass="p-4">
            <Line
              points={rendered.map((v) => ({ label: v.name, value: v.asr?.caption_vs_speech_p90_abs_s != null ? v.asr.caption_vs_speech_p90_abs_s * 1000 : null }))}
              band={[0, 250]}
              unit=" ms"
              color="var(--color-cpu)"
              min={0}
              fmt={(v) => String(Math.round(v))}
            />
          </Panel>
          <Panel title="Whisper word error" meta="% of words, target ≤ 5" bodyClass="p-4">
            <Line points={rendered.map((v) => ({ label: v.name, value: v.asr?.wer != null ? v.asr.wer * 100 : null }))} band={[0, 5]} unit=" %" color="var(--color-ok)" min={0} fmt={(v) => v.toFixed(1)} />
          </Panel>
          <Panel title="True peak" meta="dBFS, must stay ≤ -1.0" bodyClass="p-4">
            <Line points={rendered.map((v) => ({ label: v.name, value: v.true_peak ?? null }))} band={[-3, -1]} unit=" dBFS" color="var(--color-warn)" min={-4} max={0} fmt={(v) => v.toFixed(1)} />
          </Panel>
          <Panel title="Pace" meta="shots per minute" bodyClass="p-4">
            <Line points={rendered.map((v) => ({ label: v.name, value: v.shots_per_min ?? null }))} unit="/min" color="var(--color-live)" min={0} fmt={(v) => v.toFixed(1)} />
          </Panel>
          <Panel title="Length" meta="seconds" bodyClass="p-4">
            <Line points={rendered.map((v) => ({ label: v.name, value: v.duration_s ?? null }))} unit=" s" color="var(--color-sodium)" min={0} fmt={(v) => String(Math.round(v))} />
          </Panel>
        </div>
      </Section>
    </>
  );
}
