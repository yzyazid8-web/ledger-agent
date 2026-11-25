import type { Evidence } from "./types";

// Série chronologique tirée des preuves XBRL d'une réponse. Chaque point est une preuve :
// le graphique ne lit jamais un chiffre dans le texte du modèle.

export interface TrendPoint {
  evidenceId: string;
  label: string; // « 2018 » ou « 2023 T2 »
  order: number; // clé de tri chronologique
  value: number;
}

export interface Trend {
  ticker: string;
  concept: string;
  title: string; // libellé officiel du concept
  unit: string; // « USD », « shares »…
  annual: boolean;
  points: TrendPoint[];
}

// Au moins trois points : avec deux, une évolution en % se lit mieux qu'une courbe
export const MIN_TREND_POINTS = 3;

// Identifiant d'un fait : xbrl:<ticker>:<concept>:<exercice><période>, ex. xbrl:MMM:Revenues:2018FY
const SOURCE_ID = /^xbrl:([^:]+):([^:]+):(\d{4})(FY|Q[1-4])$/;
const TITLE_LABEL = /^XBRL \S+ (.+?) \[/;
const UNIT = /^-?[\d,.]+\s+([A-Za-z/]+)/;

function pointOf(e: Evidence): { key: string; ticker: string; concept: string; point: TrendPoint; annual: boolean } | null {
  if (e.kind !== "xbrl" || e.value === null) return null;
  const match = SOURCE_ID.exec(e.source_id);
  if (!match) return null;
  const [, ticker, concept, year, period] = match;
  const annual = period === "FY";
  const quarter = annual ? 0 : Number(period.slice(1));
  return {
    // Exercices et trimestres ne se mélangent jamais dans une même série
    key: `${ticker}|${concept}|${annual ? "FY" : "Q"}`,
    ticker,
    concept,
    annual,
    point: {
      evidenceId: e.id,
      label: annual ? year : `${year} T${quarter}`,
      order: Number(year) * 10 + quarter,
      value: e.value,
    },
  };
}

/** Série la plus longue (au moins 3 périodes distinctes), en privilégiant les preuves citées. */
export function findTrend(evidence: Evidence[], cited: string[]): Trend | null {
  const groups = new Map<string, { trend: Trend; cited: number; seen: Set<number> }>();
  for (const e of evidence) {
    const parsed = pointOf(e);
    if (!parsed) continue;
    let group = groups.get(parsed.key);
    if (!group) {
      group = {
        trend: {
          ticker: parsed.ticker,
          concept: parsed.concept,
          title: TITLE_LABEL.exec(e.title)?.[1] ?? parsed.concept,
          unit: UNIT.exec(e.text)?.[1] ?? "",
          annual: parsed.annual,
          points: [],
        },
        cited: 0,
        seen: new Set(),
      };
      groups.set(parsed.key, group);
    }
    // Une même période récupérée deux fois ne compte qu'une fois
    if (group.seen.has(parsed.point.order)) continue;
    group.seen.add(parsed.point.order);
    group.trend.points.push(parsed.point);
    if (cited.includes(e.id)) group.cited += 1;
  }
  const candidates = [...groups.values()].filter((g) => g.trend.points.length >= MIN_TREND_POINTS);
  if (!candidates.length) return null;
  candidates.sort((a, b) => b.trend.points.length - a.trend.points.length || b.cited - a.cited);
  const best = candidates[0].trend;
  best.points.sort((a, b) => a.order - b.order);
  return best;
}

// Échelle d'affichage adaptée à l'ordre de grandeur des valeurs
export function displayScale(values: number[], unit: string): { divisor: number; suffix: string } {
  const max = Math.max(...values.map(Math.abs));
  const money = unit === "USD";
  // En dessous de 10 milliards, les millions restent lisibles (« 1 577 M$ ») et collent au texte
  if (max >= 1e10) return { divisor: 1e9, suffix: money ? "Md$" : "Md" };
  if (max >= 1e6) return { divisor: 1e6, suffix: money ? "M$" : "M" };
  if (max >= 1e3 && !money) return { divisor: 1e3, suffix: "k" };
  return { divisor: 1, suffix: money ? "$" : unit };
}

export function formatValue(value: number, divisor: number, suffix: string): string {
  const scaled = value / divisor;
  const digits = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
  return `${scaled.toLocaleString("fr-FR", { maximumFractionDigits: digits })} ${suffix}`.trim();
}
