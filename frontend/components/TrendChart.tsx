"use client";

import { useState } from "react";
import { displayScale, formatValue, type Trend } from "@/lib/trend";

// Courbe d'un indicateur sur plusieurs périodes. Une seule série : pas de légende, le titre la
// nomme. Chaque point est une preuve XBRL, cliquable vers sa fiche dans le panneau des sources

const WIDTH = 600;
const HEIGHT = 220;
const PAD = { top: 18, right: 84, bottom: 30, left: 64 };
const TICKS = 4;

function niceTicks(min: number, max: number): number[] {
  const span = max - min || Math.abs(max) || 1;
  const raw = span / (TICKS - 1);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  // Graduations englobantes : la plus haute est au-dessus du maximum, sinon le point le plus
  // haut sortirait du graphique
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= end + step * 1e-6; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

export function TrendChart({ trend, onSelect }: { trend: Trend; onSelect: (id: string) => void }) {
  const [active, setActive] = useState<number | null>(null);
  const { points } = trend;
  const { divisor, suffix } = displayScale(
    points.map((p) => p.value),
    trend.unit,
  );
  const ticks = niceTicks(Math.min(...points.map((p) => p.value)), Math.max(...points.map((p) => p.value)));
  const yMin = ticks[0];
  const yMax = ticks[ticks.length - 1];
  // Abscisse proportionnelle au temps : une année manquante laisse un vide visible
  const xKeys = points.map((p) => (trend.annual ? Math.floor(p.order / 10) : Math.floor(p.order / 10) * 4 + (p.order % 10)));
  const xMin = Math.min(...xKeys);
  const xSpan = Math.max(...xKeys) - xMin || 1;
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + ((xKeys[i] - xMin) / xSpan) * innerW;
  const y = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin || 1)) * innerH;
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const last = points.length - 1;

  function nearest(clientX: number, rect: DOMRect): number {
    const px = ((clientX - rect.left) / rect.width) * WIDTH;
    let best = 0;
    points.forEach((_, i) => {
      if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    });
    return best;
  }

  const shown = active === null ? null : points[active];

  return (
    <figure className="space-y-2">
      <figcaption className="text-sm font-medium">
        {trend.title} · {trend.ticker}
        <span className="ml-1 font-normal text-muted">({suffix})</span>
      </figcaption>
      <div className="relative">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full"
          role="img"
          aria-label={`${trend.title}, ${trend.ticker} : ${points
            .map((p) => `${p.label} ${formatValue(p.value, divisor, suffix)}`)
            .join(", ")}`}
        >
          {/* Grille et axe : traits fins et discrets */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
              <text x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)">
                {formatValue(t, divisor, "")}
              </text>
            </g>
          ))}
          {points.map((p, i) => (
            <text key={p.evidenceId} x={x(i)} y={HEIGHT - 8} textAnchor="middle" fontSize={11} fill="var(--muted)">
              {p.label}
            </text>
          ))}
          {active !== null && (
            <line x1={x(active)} x2={x(active)} y1={PAD.top} y2={PAD.top + innerH} stroke="var(--muted)" strokeWidth={1} />
          )}
          <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {/* Zone de survol plus large que la courbe : le curseur se cale sur la période la plus proche */}
          <rect
            x={PAD.left - 20}
            y={PAD.top}
            width={innerW + 40}
            height={innerH}
            fill="transparent"
            onPointerMove={(e) => setActive(nearest(e.clientX, e.currentTarget.ownerSVGElement!.getBoundingClientRect()))}
            onPointerLeave={() => setActive(null)}
            onClick={() => active !== null && onSelect(points[active].evidenceId)}
            style={{ cursor: "pointer" }}
          />
          {points.map((p, i) => (
            <circle
              key={p.evidenceId}
              cx={x(i)}
              cy={y(p.value)}
              r={active === i ? 6 : 4.5}
              fill="var(--accent)"
              stroke="var(--panel)"
              strokeWidth={2}
              tabIndex={0}
              role="button"
              aria-label={`${p.label} : ${formatValue(p.value, divisor, suffix)}, source ${p.evidenceId}`}
              onFocus={() => setActive(i)}
              onBlur={() => setActive(null)}
              onClick={() => onSelect(p.evidenceId)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(p.evidenceId)}
              style={{ cursor: "pointer", outline: "none" }}
            />
          ))}
          {/* Une seule étiquette directe : la dernière valeur */}
          <text x={x(last) + 10} y={y(points[last].value)} dy="0.32em" fontSize={12} fontWeight={600} fill="var(--text)">
            {formatValue(points[last].value, divisor, suffix)}
          </text>
        </svg>
        {shown && active !== null && (
          <div
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded-md border border-line bg-panel px-2 py-1 text-xs shadow-sm"
            style={{ left: `${(x(active) / WIDTH) * 100}%`, top: `${(y(shown.value) / HEIGHT) * 100}%`, marginTop: -10 }}
          >
            <span className="font-medium">{shown.label}</span> · {formatValue(shown.value, divisor, suffix)}
            <span className="ml-1 font-mono text-muted">{shown.evidenceId}</span>
          </div>
        )}
      </div>
      {/* Vue tableau : les valeurs exactes, lisibles sans le graphique */}
      <details className="text-xs text-muted">
        <summary className="cursor-pointer select-none">Voir les valeurs</summary>
        <table className="mt-1 w-full max-w-sm text-left">
          <thead>
            <tr>
              <th className="py-0.5 font-medium">Période</th>
              <th className="py-0.5 font-medium">Valeur</th>
              <th className="py-0.5 font-medium">Source</th>
            </tr>
          </thead>
          <tbody className="text-ink">
            {points.map((p) => (
              <tr key={p.evidenceId}>
                <td className="py-0.5">{p.label}</td>
                <td className="py-0.5 font-mono">{p.value.toLocaleString("fr-FR")}</td>
                <td className="py-0.5">
                  <button onClick={() => onSelect(p.evidenceId)} className="font-mono text-accent hover:underline">
                    {p.evidenceId}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
