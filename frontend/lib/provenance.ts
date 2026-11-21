import type { Calculation, Evidence } from "./types";

// Découpe la réponse en morceaux : texte, citation d'une preuve (E3) ou d'un calcul (C1),
// ou nombre avec son origine

export type Provenance =
  | { kind: "calculation"; calculation: Calculation }
  | { kind: "evidence"; evidence: Evidence };

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; id: string }
  | { kind: "calculation"; calculation: Calculation }
  | { kind: "number"; text: string; provenance: Provenance | null };

// Groupe de citations entre crochets, ordinaires ou asiatiques (gpt-oss écrit souvent « 【E1】 ») :
// « [E1] », « [E1, E2, C1] » ; citation nue « E3 » ; ou nombre. Les citations passent avant
// les nombres : sinon « E3 » serait lu comme le nombre 3
const REF = String.raw`[EC]\d+`;
const TOKEN = new RegExp(
  String.raw`[\[【]\s*(${REF}(?:\s*[,;]?\s*${REF})*)\s*[\]】]|\b(${REF})\b|\$?\d[\d,]*(?:\.\d+)?(?:\s?%)?`,
  "g",
);
// Le gras Markdown (« **14.86 %** ») n'est pas interprété : ses marques sont retirées
const MARKDOWN = /\*\*|__/g;
const SCALES = [1, 1e3, 1e6, 1e9];

function parse(text: string): number {
  return Number(text.replace(/[$,%\s]/g, ""));
}

function close(a: number, b: number): boolean {
  return Math.abs(a - b) <= Math.max(Math.abs(b) * 0.005, 0.005);
}

export function findProvenance(
  value: number,
  calculations: Calculation[],
  evidence: Evidence[],
): Provenance | null {
  const calculation = calculations.find((c) => close(value, Math.abs(c.value)));
  if (calculation) return { kind: "calculation", calculation };
  // Un fait XBRL en dollars peut être cité en milliers, millions ou milliards
  const fact = evidence.find(
    (e) => e.value !== null && SCALES.some((s) => close(value, Math.abs(e.value as number) / s)),
  );
  return fact ? { kind: "evidence", evidence: fact } : null;
}

export function segment(
  text: string,
  calculations: Calculation[],
  evidence: Evidence[],
): Segment[] {
  const segments: Segment[] = [];
  const known = new Set(evidence.map((e) => e.id));
  const calcs = new Map(calculations.map((c) => [c.name, c]));
  const clean = text.replace(MARKDOWN, "");
  let last = 0;
  for (const match of clean.matchAll(TOKEN)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: clean.slice(last, start) });
    const group = match[1] ?? match[2];
    if (group) {
      // Une citation n'est mise en forme que si elle désigne une source ou un calcul réels
      for (const id of group.match(new RegExp(REF, "g")) ?? []) {
        const calculation = calcs.get(id);
        if (known.has(id)) segments.push({ kind: "citation", id });
        else if (calculation) segments.push({ kind: "calculation", calculation });
        else segments.push({ kind: "text", text: id });
      }
    } else {
      const value = parse(match[0]);
      // Les années (« FY2018 ») ne sont pas des montants à expliquer
      const isYear = Number.isInteger(value) && value >= 1900 && value <= 2100 && !match[0].includes("$");
      segments.push({
        kind: "number",
        text: match[0],
        provenance: isYear ? null : findProvenance(value, calculations, evidence),
      });
    }
    last = start + match[0].length;
  }
  if (last < clean.length) segments.push({ kind: "text", text: clean.slice(last) });
  return segments;
}
