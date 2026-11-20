"use client";

import { useMemo, useState } from "react";
import type { Company } from "@/lib/types";

export const EXAMPLES = [
  "What was 3M's capital expenditure in FY2018?",
  "What is Kraft Heinz's FY2019 inventory turnover ratio?",
  "Does 3M have a healthy quick ratio as of Q2 FY2023?",
  "Which segment drove most of AMD's revenue growth in FY2022?",
];

export function Sidebar({
  companies,
  onPick,
  disabled,
}: {
  companies: Company[];
  onPick: (question: string) => void;
  disabled: boolean;
}) {
  const [filter, setFilter] = useState("");
  const shown = useMemo(
    () =>
      companies.filter((c) => `${c.ticker} ${c.name}`.toLowerCase().includes(filter.toLowerCase())),
    [companies, filter],
  );

  return (
    <aside className="space-y-5">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Exemples</h2>
        <ul className="space-y-1.5">
          {EXAMPLES.map((q) => (
            <li key={q}>
              <button
                disabled={disabled}
                onClick={() => onPick(q)}
                className="w-full rounded-lg border border-line bg-panel p-2 text-left text-sm hover:border-accent disabled:opacity-50"
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
          Entreprises couvertes ({companies.length})
        </h2>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filtrer…"
          className="mb-2 w-full rounded-md border border-line bg-panel px-2 py-1 text-sm outline-none focus:border-accent"
        />
        <ul className="flex max-h-72 flex-wrap gap-1 overflow-y-auto">
          {shown.map((c) => (
            <li key={c.ticker} title={c.name} className="rounded bg-panel-2 px-1.5 py-0.5 font-mono text-xs">
              {c.ticker}
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
