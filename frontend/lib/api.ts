import type { AblationReport, AgentEvent, Company } from "./types";

// Le navigateur passe par le relais de l'interface (app/api/[...path]/route.ts), qui joint l'API
export const API_URL = "/api";

export async function fetchCompanies(): Promise<Company[]> {
  const response = await fetch(`${API_URL}/companies`);
  if (!response.ok) throw new Error(`API indisponible (${response.status})`);
  return response.json();
}

export async function fetchBenchmark(split = "dev"): Promise<AblationReport | null> {
  const response = await fetch(`${API_URL}/benchmark?split=${split}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`API indisponible (${response.status})`);
  return response.json();
}

// EventSource ne gère que GET : on lit donc le flux SSE de POST /chat à la main
export async function streamChat(
  question: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`L'API a répondu ${response.status}`);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // Un événement SSE se termine par une ligne vide
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) onEvent(JSON.parse(data) as AgentEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}
