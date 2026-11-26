// Événements diffusés par POST /chat (voir backend/src/ledger_agent/agent/graph.py)

// Agent ReAct : « agent » (décision) et « tool » (appel d'outil) ; workflow : les cinq étapes fixes
export type NodeName = "agent" | "tool" | "plan" | "gather" | "grade" | "synthesize" | "verify";

export interface StepEvent {
  type: "step";
  node: NodeName;
  ms: number;
  [detail: string]: unknown;
}

export interface Evidence {
  id: string;
  kind: "xbrl" | "passage";
  source_id: string;
  title: string;
  text: string;
  value: number | null;
  doc_name: string | null;
  page_number: number | null;
}

export interface Calculation {
  name: string;
  expression: string;
  value: number;
  source: string;
}

export interface AnswerEvent {
  type: "answer";
  text: string;
  final_answer: string | null;
  abstained: boolean;
  cited: string[];
  evidence: Evidence[];
  calculations: Calculation[];
  llm_calls: number;
  tokens: number;
  elapsed_s: number;
}

export interface ErrorEvent {
  type: "error";
  // « limit » : trop de questions de ce visiteur dans l'heure (protège le quota gratuit)
  code: "quota" | "limit" | "llm" | "internal";
  message: string;
  retry_after_s?: number;
}

// Question de suivi reformulée par l'API avant d'atteindre l'agent (« et en 2019 ? » → question complète)
export interface RewriteEvent {
  type: "rewrite";
  question: string;
}

export type AgentEvent = StepEvent | AnswerEvent | ErrorEvent | RewriteEvent;

// Échange terminé d'une conversation, renvoyé à l'API avec la question suivante
export interface Turn {
  question: string;
  answer: string | null;
}

export interface Company {
  ticker: string;
  name: string;
}

export interface AblationRow {
  config: string;
  questions: number;
  questions_expected: number;
  complete: boolean;
  accuracy: number | null;
  accuracy_ci95: [number, number] | null;
  wrong_answer_rate: number | null;
  abstention_rate: number | null;
  page_hit_rate: number | null;
  tokens_mean: number | null;
  latency_p50_s: number | null;
}

export interface Comparison {
  a: string;
  b: string;
  status: string;
  difference?: number;
  difference_ci95?: [number, number];
  won_by_b?: number;
  won_by_a?: number;
  mcnemar_p?: number;
  significant?: boolean;
}

export interface AblationReport {
  split: string;
  rows: AblationRow[];
  comparisons: Comparison[];
}
