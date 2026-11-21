import { ImageResponse } from "next/og";

// Image d'aperçu affichée par LinkedIn, Slack ou WhatsApp quand on partage le lien du site
export const alt = "Ledger Agent : RAG agentique qui vérifie chaque chiffre financier";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const STEPS = ["Recherche 10-K", "Données XBRL", "Calcul", "Vérification des chiffres"];
const STACK = "LangGraph · FastAPI · Next.js · Docker · Azure · CI/CD";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "64px 72px",
          background: "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)",
          color: "white",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 64,
              height: 64,
              borderRadius: 14,
              background: "#2563eb",
              fontSize: 38,
              fontWeight: 700,
            }}
          >
            L
          </div>
          <div style={{ fontSize: 36, fontWeight: 600 }}>Ledger Agent</div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ fontSize: 64, fontWeight: 700, lineHeight: 1.1 }}>RAG agentique</div>
          <div style={{ fontSize: 40, color: "#cbd5e1", lineHeight: 1.25 }}>
            qui vérifie chaque chiffre des rapports financiers SEC
          </div>
          <div style={{ display: "flex", gap: 14, marginTop: 12 }}>
            {STEPS.map((step) => (
              <div
                key={step}
                style={{
                  display: "flex",
                  padding: "10px 20px",
                  borderRadius: 999,
                  background: "rgba(255,255,255,0.12)",
                  fontSize: 26,
                }}
              >
                {step}
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 24, color: "#94a3b8" }}>
          <div>{STACK}</div>
          <div>Démo en ligne</div>
        </div>
      </div>
    ),
    size,
  );
}
