"use client";

import { arr } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { Empty, Panel, Section } from "./ui";

function Card({ label, platform, value, color }: { label: string; platform: string; value: number | null | undefined; color: string }) {
  return (
    <div className="panel bracket rise-in p-4">
      <div className="flex items-center gap-2">
        <span className="inline-block size-2 rounded-[2px]" style={{ background: color }} />
        <span className="label">{platform}</span>
      </div>
      <div className="label mt-3">{label}</div>
      <div className="display mt-1 text-[2.25rem] font-bold text-faint">{value ?? "—"}</div>
    </div>
  );
}

export function Channel({ s }: { s: Snapshot }) {
  const c = s.channel;
  const published = arr(s.videos).filter((v) => v.stage === "published");
  return (
    <Section title="Channel" hint="YouTube and Instagram numbers, once the channel exists">
      {!c?.exists && (
        <div className="panel bracket border-warn/40 p-5" data-testid="channel-placeholder">
          <div className="display text-[1.5rem] font-bold uppercase text-warn">Channel not created yet</div>
          <p className="mt-2 max-w-[70ch] text-[0.8125rem] text-muted">{c?.note ?? "Nothing is uploaded by the engine; the channel owner uploads every video."}</p>
        </div>
      )}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card platform="YouTube" label="Subscribers" value={c?.youtube?.subscribers} color="var(--color-yt)" />
        <Card platform="YouTube" label="Views" value={c?.youtube?.views} color="var(--color-yt)" />
        <Card platform="Instagram" label="Followers" value={c?.instagram?.followers} color="var(--color-ig)" />
        <Card platform="Instagram" label="Views" value={c?.instagram?.views} color="var(--color-ig)" />
      </div>
      <Panel title="Published videos" meta={`${published.length}`} bodyClass="p-4">
        {published.length ? (
          <ul className="flex flex-col gap-1 text-[0.75rem]">
            {published.map((v) => (
              <li key={v.name}>
                {v.name}
                {v.approval?.url && (
                  <a href={v.approval.url} target="_blank" rel="noopener noreferrer" className="ml-2 text-sodium hover:underline">{v.approval.url}</a>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No video is marked published yet. After uploading, use “Mark published” on its card.</Empty>
        )}
      </Panel>
    </Section>
  );
}
