"use client";

import { useEffect, useRef, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import { AnswerCard } from "@/components/AnswerCard";
import { EvidencePanel } from "@/components/EvidencePanel";
import { Sidebar } from "@/components/Sidebar";
import { fetchCompanies, streamChat } from "@/lib/api";
import type { AnswerEvent, Company, ErrorEvent, StepEvent } from "@/lib/types";

type Status = "idle" | "running" | "done" | "error";

export default function Home() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [apiDown, setApiDown] = useState(false);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [answer, setAnswer] = useState<AnswerEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchCompanies()
      .then(setCompanies)
      .catch(() => setApiDown(true));
  }, []);

  async function ask(text: string) {
    const q = text.trim();
    if (q.length < 3 || status === "running") return;
    abort.current?.abort();
    abort.current = new AbortController();
    setAsked(q);
    setQuestion("");
    setSteps([]);
    setAnswer(null);
    setError(null);
    setSelected(null);
    setStatus("running");
    try {
      await streamChat(
        q,
        (event) => {
          if (event.type === "step") setSteps((s) => [...s, event]);
          else if (event.type === "answer") setAnswer(event);
          else setError((event as ErrorEvent).message);
        },
        abort.current.signal,
      );
      setStatus("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError("Impossible de joindre l'API. Est-elle lancée ? (uv run poe api)");
      setStatus("error");
    }
  }

  const running = status === "running";

  return (
    <main className="mx-auto grid w-full max-w-[1500px] flex-1 gap-5 px-4 py-5 lg:grid-cols-[250px_minmax(0,1fr)_380px]">
      <Sidebar companies={companies} onPick={ask} disabled={running} />

      <section className="flex min-w-0 flex-col gap-4">
        {apiDown && (
          <div className="rounded-lg bg-err-soft p-3 text-sm text-err">
            L&apos;API ne répond pas sur {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}. Lance-la avec{" "}
            <code className="font-mono">uv run poe api</code>.
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
          className="flex gap-2"
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Pose une question sur les rapports annuels (en anglais), ex. What was 3M's capex in FY2018?"
            className="flex-1 rounded-xl border border-line bg-panel px-4 py-3 text-sm outline-none focus:border-accent"
            disabled={running}
          />
          <button
            type="submit"
            disabled={running || question.trim().length < 3}
            className="rounded-xl bg-accent px-5 text-sm font-medium text-white disabled:opacity-50"
          >
            {running ? "Analyse…" : "Demander"}
          </button>
        </form>

        {asked && (
          <div className="rounded-xl bg-panel-2 px-4 py-3 text-sm">
            <span className="text-muted">Question · </span>
            {asked}
          </div>
        )}

        {error && <div className="rounded-lg bg-err-soft p-3 text-sm text-err">{error}</div>}
        {answer && <AnswerCard answer={answer} onCite={setSelected} />}

        {!asked && (
          <div className="rounded-xl border border-dashed border-line p-8 text-center text-sm text-muted">
            <p className="mb-2 text-base font-medium text-ink">Des réponses financières dont chaque chiffre a une source</p>
            <p>
              L&apos;agent cherche dans les rapports SEC et les données officielles XBRL, délègue ses calculs à une
              calculatrice, puis vérifie chaque nombre. Sans source, il s&apos;abstient.
            </p>
          </div>
        )}

        <div className="rounded-xl border border-line bg-panel p-4">
          <h2 className="mb-3 text-sm font-semibold">Raisonnement de l&apos;agent</h2>
          <AgentTrace steps={steps} running={running} />
        </div>
      </section>

      <aside className="min-w-0">
        <h2 className="mb-2 text-sm font-semibold">
          Sources {answer ? `(${answer.evidence.length})` : ""}
        </h2>
        <div className="lg:sticky lg:top-4 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto">
          <EvidencePanel
            evidence={answer?.evidence ?? []}
            cited={answer?.cited ?? []}
            selected={selected}
            onSelect={setSelected}
          />
        </div>
      </aside>
    </main>
  );
}
