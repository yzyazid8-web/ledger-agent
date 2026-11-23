"use client";

import { useEffect, useState } from "react";
import { fetchBenchmark } from "@/lib/api";
import type { AblationReport, AblationRow } from "@/lib/types";

const DESCRIPTIONS: Record<string, string> = {
  naive: "RAG classique : recherche vectorielle, une génération",
  hybrid: "Recherche hybride filtrée (vecteurs + BM25), sans agent",
  hybrid_rerank: "Recherche hybride + reranker",
  workflow: "Workflow agentique à étapes fixes (le LLM décide dans chaque étape)",
  workflow_120b: "Même workflow, avec gpt-oss-120b à toutes les étapes comme l'agent",
  agent_sans_xbrl: "Agent ReAct sans les outils de données XBRL",
  agent_sans_verif: "Agent ReAct sans la vérification des chiffres",
  agent: "Agent ReAct complet : le LLM choisit ses outils et lit leurs résultats",
};

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "–" : `${Math.round(value * 100)} %`;
}

function AccuracyBar({ row }: { row: AblationRow }) {
  const [low, high] = row.accuracy_ci95 ?? [0, 0];
  return (
    <div className="relative h-2.5 w-full rounded-full bg-panel-2">
      {/* L'intervalle de confiance, puis l'exactitude mesurée */}
      <div
        className="absolute h-full rounded-full bg-accent-soft"
        style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%` }}
      />
      <div
        className="absolute top-1/2 h-4 w-1 -translate-y-1/2 rounded bg-accent"
        style={{ left: `calc(${(row.accuracy ?? 0) * 100}% - 2px)` }}
      />
    </div>
  );
}

export default function Benchmark() {
  const [report, setReport] = useState<AblationReport | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "missing" | "error">("loading");

  useEffect(() => {
    fetchBenchmark("dev")
      .then((r) => {
        setReport(r);
        setState(r ? "ready" : "missing");
      })
      .catch(() => setState("error"));
  }, []);

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 space-y-6 px-4 py-6">
      <div>
        <h1 className="text-2xl font-semibold">Étude d&apos;ablation · FinanceBench</h1>
        <p className="mt-1 text-sm text-muted">
          Chaque configuration est évaluée sur les mêmes questions. Le trait est l&apos;exactitude mesurée, la bande
          son intervalle de confiance à 95 %. Une question non jugée ou en erreur compte comme ratée.
        </p>
      </div>

      {state === "loading" && <p className="text-sm text-muted">Chargement…</p>}
      {state === "missing" && (
        <p className="text-sm text-muted">
          Aucun rapport pour l&apos;instant. Génère-le avec <code className="font-mono">uv run poe eval-report</code>.
        </p>
      )}
      {state === "error" && <p className="text-sm text-err">L&apos;API ne répond pas.</p>}

      {report && (
        <>
          <div className="overflow-x-auto rounded-xl border border-line bg-panel">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-muted">
                <tr className="border-b border-line">
                  <th className="p-3">Configuration</th>
                  <th className="p-3 w-1/4">Exactitude</th>
                  <th className="p-3">Fausses</th>
                  <th className="p-3">Abstention</th>
                  <th className="p-3">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {report.rows.map((row) => (
                  <tr key={row.config} className="border-b border-line last:border-0">
                    <td className="p-3">
                      <p className="font-mono font-medium">{row.config}</p>
                      <p className="text-xs text-muted">{DESCRIPTIONS[row.config] ?? ""}</p>
                      {!row.complete && (
                        <p className="text-xs text-warn">
                          incomplet : {row.questions}/{row.questions_expected}
                        </p>
                      )}
                    </td>
                    <td className="p-3">
                      <p className="mb-1 font-semibold">{pct(row.accuracy)}</p>
                      <AccuracyBar row={row} />
                    </td>
                    <td className="p-3">{pct(row.wrong_answer_rate)}</td>
                    <td className="p-3">{pct(row.abstention_rate)}</td>
                    <td className="p-3 font-mono">{row.tokens_mean?.toLocaleString("fr-FR") ?? "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <h2 className="mb-2 font-semibold">Comparaisons appariées (test de McNemar)</h2>
            <ul className="space-y-2">
              {report.comparisons.map((c) => (
                <li key={`${c.a}-${c.b}`} className="rounded-lg border border-line bg-panel p-3 text-sm">
                  <span className="font-mono">{c.a}</span> → <span className="font-mono">{c.b}</span>
                  {c.status !== "ok" ? (
                    <span className="ml-2 text-muted">{c.status}</span>
                  ) : (
                    <span className="ml-2">
                      {c.difference! >= 0 ? "+" : ""}
                      {Math.round(c.difference! * 100)} points · {c.won_by_b} gagnées / {c.won_by_a} perdues · p ={" "}
                      {c.mcnemar_p}{" "}
                      <span className={c.significant ? "font-medium text-ok" : "text-muted"}>
                        {c.significant ? "(significatif)" : "(non significatif)"}
                      </span>
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </main>
  );
}
