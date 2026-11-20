// Relais vers l'API Python. L'adresse est lue à l'exécution (API_URL), pas à la construction :
// la même image fonctionne en local, dans docker compose et sur Azure.

const API_URL = process.env.API_URL ?? "http://localhost:8000";

// Liste fermée des routes relayées : impossible de détourner le relais vers une autre adresse
const ROUTES: Record<string, "GET" | "POST"> = {
  health: "GET",
  companies: "GET",
  benchmark: "GET",
  chat: "POST",
  ask: "POST",
};

async function relay(request: Request, path: string[], method: "GET" | "POST"): Promise<Response> {
  const endpoint = path.join("/");
  if (ROUTES[endpoint] !== method) {
    return Response.json({ detail: "Route inconnue" }, { status: 404 });
  }
  const target = new URL(endpoint, `${API_URL}/`);
  target.search = new URL(request.url).search;
  try {
    const upstream = await fetch(target, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "POST" ? await request.text() : undefined,
      signal: request.signal,
      cache: "no-store",
    });
    // Le corps est transmis tel quel : le flux SSE de /chat arrive au navigateur au fil de l'eau
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json({ detail: "API injoignable" }, { status: 502 });
  }
}

export async function GET(request: Request, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  return relay(request, path, "GET");
}

export async function POST(request: Request, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  return relay(request, path, "POST");
}
