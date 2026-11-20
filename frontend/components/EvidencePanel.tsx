"use client";

import { useEffect, useRef } from "react";
import type { Evidence } from "@/lib/types";

export function EvidencePanel({
  evidence,
  cited,
  selected,
  onSelect,
}: {
  evidence: Evidence[];
  cited: string[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const refs = useRef<Record<string, HTMLLIElement | null>>({});

  useEffect(() => {
    if (selected) refs.current[selected]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected]);

  if (!evidence.length) {
    return <p className="text-sm text-muted">Les sources consultées par l&apos;agent apparaîtront ici.</p>;
  }
  // Les preuves citées dans la réponse passent en premier
  const ordered = [...evidence].sort((a, b) => Number(cited.includes(b.id)) - Number(cited.includes(a.id)));
  return (
    <ul className="space-y-2">
      {ordered.map((e) => (
        <li
          key={e.id}
          ref={(node) => {
            refs.current[e.id] = node;
          }}
          onClick={() => onSelect(e.id)}
          className={`cursor-pointer rounded-lg border p-3 transition ${
            selected === e.id ? "border-accent bg-accent-soft" : "border-line bg-panel hover:border-accent"
          }`}
        >
          <div className="mb-1 flex flex-wrap items-center gap-1.5 text-xs">
            <span className="rounded bg-ink px-1.5 py-0.5 font-mono font-semibold text-panel">{e.id}</span>
            <span
              className={`rounded px-1.5 py-0.5 ${
                e.kind === "xbrl" ? "bg-ok-soft text-ok" : "bg-panel-2 text-muted"
              }`}
            >
              {e.kind === "xbrl" ? "Donnée officielle XBRL" : `Rapport · page ${e.page_number}`}
            </span>
            {cited.includes(e.id) && <span className="rounded bg-accent-soft px-1.5 py-0.5 text-accent">cité</span>}
          </div>
          <p className="text-sm font-medium">{e.title}</p>
          <p className={`mt-1 whitespace-pre-line text-xs text-muted ${selected === e.id ? "" : "line-clamp-3"}`}>
            {e.text}
          </p>
          <p className="mt-1 truncate font-mono text-[11px] text-muted">{e.source_id}</p>
        </li>
      ))}
    </ul>
  );
}
