import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ledger-agent",
  description:
    "Analyse financière sur les rapports SEC : chaque chiffre est relié à sa source, ou l'agent s'abstient.",
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
                · chiffres traçables sur les rapports SEC
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
