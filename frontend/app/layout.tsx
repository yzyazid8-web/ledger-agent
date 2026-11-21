import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const TITLE = "Ledger Agent : RAG agentique qui vérifie chaque chiffre financier";
const DESCRIPTION =
  "Agent ReAct sur les rapports SEC : il choisit ses outils (recherche dans les 10-K, données XBRL " +
  "officielles, calculatrice), lit leurs résultats et vérifie la source de chaque chiffre avant de répondre.";

export const metadata: Metadata = {
  // Adresse publique, pour que l'image d'aperçu ait une URL absolue (sinon : localhost).
  // La page est générée au build : SITE_URL se règle donc au moment de construire l'image
  metadataBase: new URL(
    process.env.SITE_URL ?? "https://ledger-agent-web.politebeach-befb1bc3.francecentral.azurecontainerapps.io",
  ),
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, type: "website", locale: "fr_FR" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fr" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <header className="border-b border-line bg-panel">
          <div className="mx-auto flex max-w-[1500px] items-center justify-between px-4 py-3">
            <Link href="/" className="flex items-center gap-2 font-semibold">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-sm text-white">
                L
              </span>
              ledger-agent
              <span className="hidden text-sm font-normal text-muted sm:inline">
                · RAG agentique sur les rapports SEC, chaque chiffre vérifié
              </span>
            </Link>
            <nav className="flex gap-1 text-sm">
              <Link href="/" className="rounded-md px-3 py-1.5 hover:bg-panel-2">
                Assistant
              </Link>
              <Link href="/benchmark" className="rounded-md px-3 py-1.5 hover:bg-panel-2">
                Benchmark
              </Link>
            </nav>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
