import type { NodeName, StepEvent } from "@/lib/types";

const LABELS: Record<NodeName, { title: string; hint: string }> = {
  agent: { title: "Décision de l'agent", hint: "quel outil appeler, ou répondre" },
  tool: { title: "Résultat des outils", hint: "ce que l'agent observe avant de décider" },
  plan: { title: "Planification", hint: "entreprise, exercice, chiffres à chercher" },
  gather: { title: "Collecte des preuves", hint: "faits XBRL et passages des rapports" },
  grade: { title: "Contrôle de suffisance", hint: "faut-il chercher davantage ?" },
  synthesize: { title: "Rédaction et calculs", hint: "calculs exécutés par la calculatrice" },
  verify: { title: "Vérification des chiffres", hint: "chaque nombre doit avoir une source" },
};

type Dict = Record<string, unknown>;

function asList<T = Dict>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

const TOOL_NAMES: Record<string, string> = {
  search_filings: "Recherche dans les rapports",
  find_concepts: "Recherche de concept XBRL",
  get_fact: "Donnée officielle XBRL",
  calculate: "Calcul",
  submit_answer: "Réponse soumise",
  abstain: "Abstention",
};

function Detail({ step }: { step: StepEvent }) {
  switch (step.node) {
    case "agent": {
      if (step.error) return <p className="text-warn">Appel d&apos;outil mal formé, l&apos;agent réessaie.</p>;
      const actions = asList(step.actions);
      return actions.length ? (
        <ul className="space-y-1">
          {actions.map((a, i) => (
            <li key={i}>
              <b>{TOOL_NAMES[String(a.tool)] ?? String(a.tool)}</b>
              <span className="ml-1 break-all font-mono text-[11px] text-muted">{JSON.stringify(a.args)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-muted">Aucun outil appelé : l&apos;agent est relancé.</p>
      );
    }
    case "tool":
      return (
        <ul className="space-y-1">
          {asList(step.results).map((r, i) => (
            <li key={i}>
              {TOOL_NAMES[String(r.tool)] ?? String(r.tool)} :{" "}
              <span className={String(r.summary).startsWith("erreur") ? "text-warn" : ""}>
                {String(r.summary) || "reçu"}
              </span>
            </li>
          ))}
        </ul>
      );
    case "plan": {
      const plan = (step.plan ?? {}) as Dict;
      const metrics = asList(plan.metrics).map((m) => String(m.label));
      return (
        <div className="space-y-1">
          <p>
            <b>{String(plan.ticker ?? "?")}</b> · exercice {String(plan.fiscal_year ?? "?")}
            {step.fallback ? " · plan de secours (règles)" : ""}
          </p>
          {metrics.length > 0 && <p>Chiffres officiels : {metrics.join(", ")}</p>}
          <p className="text-muted">Requêtes : {asList<string>(plan.search_queries).join(" · ")}</p>
        </div>
      );
    }
    case "gather":
      return (
        <ul className="space-y-1">
          {asList(step.facts).map((f, i) => (
            <li key={`f${i}`}>
              {f.found ? "✓" : "✗"} {String(f.metric)} {String(f.year)}
              {!f.found && <span className="text-warn"> · absent des données XBRL</span>}
            </li>
          ))}
          {asList(step.searches).map((s, i) => (
            <li key={`s${i}`} className="text-muted">
              🔎 « {String(s.query)} »
              {s.relaxed ? <span className="text-warn"> · filtres élargis</span> : null}
            </li>
          ))}
          <li>{String(step.evidence)} preuves au total</li>
        </ul>
      );
    case "grade":
      return step.sufficient ? (
        <p>Preuves suffisantes.</p>
      ) : (
        <div>
          <p className="text-warn">Il manque : {String(step.missing || "?")}</p>
          {asList<string>(step.new_queries).length > 0 && (
            <p className="text-muted">Nouvelle recherche : {asList<string>(step.new_queries).join(" · ")}</p>
          )}
        </div>
      );
    case "synthesize": {
      if (step.abstained) return <p className="text-warn">Abstention : {String(step.reason)}</p>;
      const calcs = asList(step.calculations);
      return calcs.length ? (
        <ul className="space-y-1 font-mono text-xs">
          {calcs.map((c, i) => (
            <li key={i}>
              {String(c.name)} = {String(c.expression)} = <b>{Number(c.value).toLocaleString("fr-FR")}</b>
            </li>
          ))}
        </ul>
      ) : (
        <p>Réponse rédigée sans calcul.</p>
      );
    }
    case "verify":
      if (step.ok) return <p className="text-ok">Tous les nombres ont une source.</p>;
      if (step.skipped) return <p className="text-muted">Rien à vérifier (abstention).</p>;
      return (
        <div>
          <ul className="text-err">
            {asList<string>(step.problems).map((p, i) => (
              <li key={i}>✗ {p}</li>
            ))}
          </ul>
          <p className="text-muted">{step.retry ? "Nouvelle rédaction demandée." : "L'agent s'abstient."}</p>
        </div>
      );
  }
}

// Soumettre la réponse ou s'abstenir n'a pas de résultat à montrer : la décision suffit
const FINAL_TOOLS = new Set(["submit_answer", "abstain"]);

function visible(steps: StepEvent[]): StepEvent[] {
  return steps.flatMap((step) => {
    if (step.node !== "tool") return [step];
    const results = asList(step.results).filter((r) => !FINAL_TOOLS.has(String(r.tool)));
    return results.length ? [{ ...step, results }] : [];
  });
}

export function AgentTrace({ steps, running }: { steps: StepEvent[]; running: boolean }) {
  if (!steps.length && !running) {
    return <p className="text-sm text-muted">Le raisonnement de l&apos;agent s&apos;affichera ici, étape par étape.</p>;
  }
  return (
    <ol className="relative space-y-3 border-l border-line pl-4">
      {visible(steps).map((step, i) => (
        <li key={i} className="rise">
          <span className="absolute -left-[5px] mt-1.5 h-2.5 w-2.5 rounded-full bg-accent" />
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm font-medium">{LABELS[step.node]?.title ?? step.node}</span>
            <span className="font-mono text-xs text-muted">{(step.ms / 1000).toFixed(1)} s</span>
          </div>
          <p className="mb-1 text-xs text-muted">{LABELS[step.node]?.hint}</p>
          <div className="rounded-md bg-panel-2 p-2 text-xs leading-relaxed">
            <Detail step={step} />
          </div>
        </li>
      ))}
      {running && (
        <li className="flex items-center gap-2 text-sm text-muted">
          <span className="absolute -left-[5px] h-2.5 w-2.5 animate-ping rounded-full bg-accent" />
          L&apos;agent travaille…
        </li>
      )}
    </ol>
  );
}
