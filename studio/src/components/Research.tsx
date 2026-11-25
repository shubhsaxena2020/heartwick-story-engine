"use client";

import { arr, bytes } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { Markdown } from "./Markdown";
import { Empty, Panel, Section, Split, Time, useMediaSrc } from "./ui";

function Sample({ k, name, size }: { k: string; name: string; size: number }) {
  const [src, fail] = useMediaSrc(k);
  return (
    <li className="flex flex-col gap-1 px-4 py-2">
      <div className="flex items-baseline justify-between gap-2 text-[0.75rem]">
        <span className="truncate font-semibold text-ink" title={name}>{name}</span>
        <span className="shrink-0 text-[0.625rem] text-faint">{bytes(size)}</span>
      </div>
      {src && <audio controls preload="none" src={src} onError={() => src.startsWith("/api/") && fail()} className="h-8 w-full" />}
    </li>
  );
}

export function Research({ s }: { s: Snapshot }) {
  const r = s.research ?? {};
  const samples = arr(r.samples);
  const engineSamples = arr(s.videos).flatMap((v) => arr(v.samples).map((x) => ({ ...x, name: `${v.name} · ${x.name}` })));
  return (
    <Section title="Research" hint="read-only view of the research/ folder on the laptop">
      <Split side="24rem">
        <Panel title="Research report" meta={r.report_mtime ? <>research/RESEARCH_REPORT.md · updated <Time iso={r.report_mtime} /></> : "research/RESEARCH_REPORT.md"} bodyClass="p-5">
          {r.report_md ? (
            <>
              <Markdown text={r.report_md} />
              {r.report_truncated && <p className="mt-4 text-[0.6875rem] text-warn">The report is longer than 200 KB; the rest is on the laptop.</p>}
            </>
          ) : (
            <Empty>
              {r.present ? "The research folder exists, but RESEARCH_REPORT.md is not written yet." : "No research yet."} Another agent is writing it; it appears here by
              itself as soon as the file exists.
            </Empty>
          )}
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel title="Voice samples" meta={`${samples.length} in research/`} bodyClass="p-0">
            {samples.length ? <ul className="divide-y divide-rule/60">{samples.map((x) => <Sample key={x.key} k={x.key} name={x.name} size={x.bytes} />)}</ul> : <div className="p-4"><Empty>No samples in research/ yet.</Empty></div>}
          </Panel>
          {engineSamples.length > 0 && (
            <Panel title="Engine voice samples" meta="out/<video>/samples" bodyClass="p-0">
              <ul className="divide-y divide-rule/60">{engineSamples.map((x) => <Sample key={x.key} k={x.key} name={x.name} size={x.bytes} />)}</ul>
            </Panel>
          )}
          {arr(r.other_docs).length > 0 && (
            <Panel title="Other notes" bodyClass="p-0">
              <ul className="divide-y divide-rule/60 text-[0.75rem]">
                {arr(r.other_docs).map((d) => (
                  <li key={d.name} className="flex justify-between gap-2 px-4 py-1.5"><span className="truncate">{d.name}</span><Time iso={d.mtime} className="text-faint" /></li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      </Split>
    </Section>
  );
}
