import type { Calculation, Evidence } from "./types";

// Découpe la réponse en morceaux : texte, citation [E3], ou nombre avec son origine

export type Provenance =
  | { kind: "calculation"; calculation: Calculation }
  | { kind: "evidence"; evidence: Evidence };

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; id: string }
  | { kind: "number"; text: string; provenance: Provenance | null };

const TOKEN = /\[(E\d+)\]|\$?\d[\d,]*(?:\.\d+)?%?/g;
const SCALES = [1, 1e3, 1e6, 1e9];

function parse(text: string): number {
  return Number(text.replace(/[$,%]/g, ""));
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
  let last = 0;
  for (const match of text.matchAll(TOKEN)) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: text.slice(last, start) });
    if (match[1]) {
      segments.push({ kind: "citation", id: match[1] });
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
  if (last < text.length) segments.push({ kind: "text", text: text.slice(last) });
  return segments;
}
