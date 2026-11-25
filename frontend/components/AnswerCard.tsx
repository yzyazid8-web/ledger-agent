import { TrendChart } from "@/components/TrendChart";
import { segment } from "@/lib/provenance";
import { findTrend } from "@/lib/trend";
import type { AnswerEvent } from "@/lib/types";

function Text({ text, answer, onCite }: { text: string; answer: AnswerEvent; onCite: (id: string) => void }) {
  return (
    <>
      {segment(text, answer.calculations, answer.evidence).map((part, i) => {
        if (part.kind === "text") return <span key={i}>{part.text}</span>;
        if (part.kind === "citation") {
          return (
            <button
              key={i}
              onClick={() => onCite(part.id)}
              className="mx-0.5 rounded bg-accent-soft px-1 font-mono text-xs text-accent hover:underline"
            >
              {part.id}
            </button>
          );
        }
        if (part.kind === "calculation") {
          // Un calcul n'a pas de fiche source : sa formule s'affiche au survol
          const { name, expression, value } = part.calculation;
          return (
            <span key={i} className="group relative mx-0.5 cursor-help">
              <span className="rounded bg-ok-soft px-1 font-mono text-xs text-ok">{name}</span>
              <span className="pointer-events-none absolute bottom-full left-0 z-10 mb-1 hidden w-72 rounded-md bg-ink p-2 font-mono text-xs text-panel shadow-lg group-hover:block">
                {name} = {expression} = {value.toLocaleString("en-US", { maximumFractionDigits: 4 })}
              </span>
            </span>
          );
        }
        if (!part.provenance) return <span key={i}>{part.text}</span>;
        // Chaque nombre traçable est souligné ; son origine s'affiche au survol
        const origin =
          part.provenance.kind === "calculation"
            ? `Calcul : ${part.provenance.calculation.expression}`
            : `Source : ${part.provenance.evidence.id} · ${part.provenance.evidence.title}`;
        return (
          <span key={i} className="group relative cursor-help">
            <span className="border-b-2 border-dotted border-ok font-medium">{part.text}</span>
            <span className="pointer-events-none absolute bottom-full left-0 z-10 mb-1 hidden w-72 rounded-md bg-ink p-2 text-xs text-panel shadow-lg group-hover:block">
              {origin}
            </span>
          </span>
        );
      })}
    </>
  );
}

export function AnswerCard({ answer, onCite }: { answer: AnswerEvent; onCite: (id: string) => void }) {
  const reasoning = answer.text.split(/final answer\s*:/i)[0].trim();
  // Graphique seulement si les preuves forment une série d'au moins trois périodes
  const trend = answer.abstained ? null : findTrend(answer.evidence, answer.cited);
  return (
    <div className="rise space-y-3 rounded-xl border border-line bg-panel p-4">
      {answer.abstained ? (
        <div className="rounded-lg bg-warn-soft p-3 text-sm text-warn">
          <b>L&apos;agent s&apos;abstient.</b> Les preuves trouvées ne permettent pas de donner une réponse dont
          chaque chiffre soit vérifiable. C&apos;est volontaire : mieux vaut aucune réponse qu&apos;un chiffre
          inventé.
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-ok-soft px-2 py-0.5 text-xs font-medium text-ok">
              ✓ chiffres vérifiés
            </span>
            <span className="text-xs text-muted">survolez un nombre souligné pour voir son origine</span>
          </div>
          <p className="text-2xl font-semibold leading-snug">
            <Text text={answer.final_answer ?? ""} answer={answer} onCite={onCite} />
          </p>
          {reasoning && (
            <p className="text-sm leading-relaxed text-muted">
              <Text text={reasoning} answer={answer} onCite={onCite} />
            </p>
          )}
          {trend && (
            <div className="rounded-lg border border-line p-3">
              <TrendChart trend={trend} onSelect={onCite} />
            </div>
          )}
        </>
      )}
      <p className="font-mono text-xs text-muted">
        {answer.elapsed_s} s · {answer.llm_calls} appels LLM · {answer.tokens.toLocaleString("fr-FR")} tokens
      </p>
    </div>
  );
}
