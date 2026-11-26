"use client";

import { useEffect, useRef, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import { AnswerCard } from "@/components/AnswerCard";
import { EvidencePanel } from "@/components/EvidencePanel";
import { Sidebar } from "@/components/Sidebar";
import { fetchCompanies, streamChat } from "@/lib/api";
import type { AnswerEvent, Company, ErrorEvent, StepEvent, Turn } from "@/lib/types";

type Status = "idle" | "running" | "done" | "error";
// État de l'API au chargement : sur Azure, elle s'arrête sans visiteur et met jusqu'à une minute
// à redémarrer. On réessaie pendant ce temps au lieu d'afficher tout de suite une erreur
type ApiState = "checking" | "waking" | "ready" | "down";
// Échanges renvoyés à l'API pour comprendre une question de suivi (« et en 2019 ? »)
const HISTORY_TURNS = 3;
const WAKE_RETRY_MS = 5000;
const WAKE_TIMEOUT_MS = 120_000;
const SLOW_START_MS = 2500;

export default function Home() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [api, setApi] = useState<ApiState>("checking");
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  // Échanges terminés de la conversation en cours, et question comprise par l'API
  const [turns, setTurns] = useState<Turn[]>([]);
  const [understood, setUnderstood] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [answer, setAnswer] = useState<AnswerEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    // Sans réponse rapide, on prévient le visiteur que le serveur démarre
    const slow = setTimeout(() => !cancelled && setApi((a) => (a === "checking" ? "waking" : a)), SLOW_START_MS);
    async function load() {
      while (!cancelled) {
        try {
          const list = await fetchCompanies();
          if (cancelled) return;
          setCompanies(list);
          setApi("ready");
          return;
        } catch {
          if (Date.now() - started > WAKE_TIMEOUT_MS) {
            if (!cancelled) setApi("down");
            return;
          }
          if (!cancelled) setApi("waking");
          await new Promise((resolve) => setTimeout(resolve, WAKE_RETRY_MS));
        }
      }
    }
    load();
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);

  async function ask(text: string) {
    const q = text.trim();
    if (q.length < 3 || status === "running" || api !== "ready") return;
    abort.current?.abort();
    abort.current = new AbortController();
    const history = turns.slice(-HISTORY_TURNS);
    let standalone: string | null = null;
    let final: string | null = null;
    setAsked(q);
    setUnderstood(null);
    setQuestion("");
    setSteps([]);
    setAnswer(null);
    setError(null);
    setSelected(null);
    setStatus("running");
    try {
      await streamChat(
        q,
        history,
        (event) => {
          if (event.type === "step") setSteps((s) => [...s, event]);
          else if (event.type === "rewrite") {
            standalone = event.question;
            setUnderstood(event.question);
          } else if (event.type === "answer") {
            final = event.abstained ? "(pas de réponse vérifiable)" : event.final_answer;
            setAnswer(event);
          } else setError((event as ErrorEvent).message);
        },
        abort.current.signal,
      );
      // La question complète, plutôt que « et en 2019 ? », aide à comprendre la suivante
      setTurns((t) => [...t, { question: standalone ?? q, answer: final }].slice(-6));
      setStatus("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError("Impossible de joindre l'API. Est-elle lancée ? (uv run poe api)");
      setStatus("error");
    }
  }

  function newConversation() {
    abort.current?.abort();
    setTurns([]);
    setAsked("");
    setUnderstood(null);
    setSteps([]);
    setAnswer(null);
    setError(null);
    setSelected(null);
    setStatus("idle");
  }

  const running = status === "running";
  // Pas de question tant que le serveur n'a pas répondu : elle échouerait pendant le démarrage
  const blocked = running || api !== "ready";

  return (
    <main className="mx-auto grid w-full max-w-[1500px] flex-1 gap-5 px-4 py-5 lg:grid-cols-[250px_minmax(0,1fr)_380px]">
      <Sidebar companies={companies} onPick={ask} disabled={blocked} />

      <section className="flex min-w-0 flex-col gap-4">
        {api === "waking" && (
          <div className="flex items-center gap-3 rounded-lg bg-accent-soft p-3 text-sm text-accent">
            <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            <span>
              Démarrage du serveur, environ 30 secondes à une minute. Le site s&apos;arrête quand personne ne
              l&apos;utilise, pour rester gratuit. La page se met à jour toute seule.
            </span>
          </div>
        )}
        {api === "down" && (
          <div className="rounded-lg bg-err-soft p-3 text-sm text-err">
            Le serveur ne répond pas pour le moment. Réessaie dans quelques minutes.
            {process.env.NODE_ENV === "development" && (
              <>
                {" "}
                En local, lance l&apos;API avec <code className="font-mono">uv run poe api</code>.
              </>
            )}
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
            placeholder={
              turns.length
                ? "Question de suivi, ex. And in 2019?"
                : "Pose une question sur les rapports annuels (en anglais), ex. What was 3M's capex in FY2018?"
            }
            className="flex-1 rounded-xl border border-line bg-panel px-4 py-3 text-sm outline-none focus:border-accent"
            disabled={running}
          />
          <button
            type="submit"
            disabled={blocked || question.trim().length < 3}
            className="rounded-xl bg-accent px-5 text-sm font-medium text-white disabled:opacity-50"
          >
            {running ? "Analyse…" : "Demander"}
          </button>
        </form>
        {/* Le suivi en production (Langfuse) conserve les questions posées : on le dit */}
        <p className="-mt-2 text-xs text-muted">
          Les questions posées peuvent être enregistrées pour améliorer l&apos;agent. N&apos;y saisis pas
          d&apos;informations personnelles.
        </p>

        {turns.length > 0 && (
          <div className="flex items-start justify-between gap-3">
            {/* Échanges précédents de la conversation, en version courte */}
            <ol className="min-w-0 flex-1 space-y-1 text-xs text-muted">
              {turns.slice(0, running ? turns.length : -1).map((t, i) => (
                <li key={i} className="truncate">
                  <span className="text-ink">{t.question}</span> → {t.answer ?? "sans réponse"}
                </li>
              ))}
            </ol>
            <button
              onClick={newConversation}
              disabled={running}
              className="shrink-0 rounded-md border border-line px-2 py-1 text-xs hover:border-accent disabled:opacity-50"
            >
              Nouvelle conversation
            </button>
          </div>
        )}

        {asked && (
          <div className="rounded-xl bg-panel-2 px-4 py-3 text-sm">
            <span className="text-muted">Question · </span>
            {asked}
            {understood && (
              <p className="mt-1 text-xs text-muted">
                J&apos;ai compris : <span className="text-ink">{understood}</span>
              </p>
            )}
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
