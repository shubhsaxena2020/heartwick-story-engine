"use client";

import { useState } from "react";
import { arr } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { EventList } from "./Overview";
import { Chips, Empty, Panel, Section } from "./ui";

const LEVELS = ["problems", "err", "warn", "all"] as const;
type Level = (typeof LEVELS)[number];
const NAMES: Record<Level, string> = { problems: "Errors + warnings", err: "Errors", warn: "Warnings", all: "Everything" };

export function Errors({ s }: { s: Snapshot }) {
  const ev = arr(s.events);
  const [lv, setLv] = useState<Level>("problems");
  const [src, setSrc] = useState("all");
  const sources = ["all", ...Array.from(new Set(ev.map((e) => e.source)))];
  const shown = ev.filter((e) => (lv === "all" ? true : lv === "problems" ? e.level !== "info" : e.level === lv)).filter((e) => src === "all" || e.source === src);
  const counts = { problems: ev.filter((e) => e.level !== "info").length, err: ev.filter((e) => e.level === "err").length, warn: ev.filter((e) => e.level === "warn").length, all: ev.length };
  const fails = arr(s.truth?.checks).filter((c) => c.ok === false);
  return (
    <>
      <Section title="Errors and warnings" hint="Kaggle worker errors, failed or cancelled jobs, queue-runner failures, QA out of range, exporter problems" right={<Chips value={lv} options={LEVELS} onChange={setLv} label="Level" counts={counts} names={NAMES} />}>
        <div className="flex flex-wrap items-center gap-2">
          <span className="label">Source</span>
          <Chips value={src} options={sources} onChange={setSrc} label="Source" />
        </div>
        {fails.length > 0 && (
          <Panel title="Self-check mismatches" meta="a number shown here disagrees with a second way of computing it" bodyClass="p-0">
            <ul className="divide-y divide-rule/60">
              {fails.map((c) => (
                <li key={c.key} className="slide-in px-4 py-2 text-[0.75rem]">
                  <span className="font-semibold text-err">{c.label}</span>
                  <span className="text-muted"> · recomputed {String(c.expected)} vs shown {String(c.shown)}{c.tol ? ` (±${c.tol})` : ""}</span>
                  <div className="text-[0.6875rem] text-faint">{c.source}{c.note ? ` · ${c.note}` : ""}</div>
                </li>
              ))}
            </ul>
          </Panel>
        )}
        <Panel title="Feed" meta={`${shown.length} of ${ev.length}`} bodyClass="p-0">
          {shown.length ? <EventList events={shown.slice(0, 200)} /> : <div className="p-4"><Empty>Nothing here.</Empty></div>}
        </Panel>
      </Section>
    </>
  );
}
