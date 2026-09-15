import React from "react";

export function Button({
  children,
  variant = "default",
  size = "md",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger" | "ghost";
  size?: "sm" | "md";
}) {
  const styles: Record<string, string> = {
    default:
      "border-neutral-700/80 bg-neutral-800/60 text-neutral-100 hover:border-neutral-600 hover:bg-neutral-700/60",
    primary:
      "border-emerald-600/60 bg-emerald-600/20 text-emerald-100 hover:border-emerald-500 hover:bg-emerald-600/30",
    danger:
      "border-red-800/70 bg-red-900/30 text-red-100 hover:border-red-700 hover:bg-red-900/50",
    ghost:
      "border-transparent text-neutral-400 hover:bg-neutral-800/70 hover:text-neutral-100",
  };
  const sizes: Record<string, string> = {
    sm: "px-2 py-1 text-xs",
    md: "px-3 py-1.5 text-sm",
  };
  return (
    <button
      {...rest}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border font-medium shadow-sm transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none ${sizes[size]} ${styles[variant]} ${rest.className ?? ""}`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-[11px] font-medium uppercase tracking-wider text-neutral-400">
        {label}
      </span>
      {children}
      {hint && <span className="block text-xs leading-relaxed text-neutral-500">{hint}</span>}
    </label>
  );
}

const CONTROL =
  "w-full rounded-md border border-neutral-700 bg-neutral-950/60 px-2.5 py-1.5 text-sm text-neutral-100 shadow-inner outline-none transition-colors placeholder:text-neutral-600 hover:border-neutral-600 focus:border-neutral-400";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`mono ${CONTROL} ${props.className ?? ""}`} />;
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${CONTROL} ${props.className ?? ""}`} />;
}

/** Status vocabulary, grouped by what the status *means* so the tree, the runs
 *  table and the loop strip all read the same way. */
const STATUS_COLORS: Record<string, string> = {
  // in flight
  running: "border-blue-700/60 bg-blue-500/15 text-blue-200",
  implementing: "border-amber-700/60 bg-amber-500/15 text-amber-200",
  queued: "border-neutral-700 bg-neutral-800/80 text-neutral-400",
  proposed: "border-neutral-700 bg-neutral-800/80 text-neutral-300",
  baseline: "border-sky-700/60 bg-sky-500/15 text-sky-200",
  holdout: "border-violet-700/60 bg-violet-500/15 text-violet-200",
  // settled, good
  kept: "border-emerald-600/60 bg-emerald-500/15 text-emerald-200",
  done: "border-emerald-600/60 bg-emerald-500/15 text-emerald-200",
  ready: "border-neutral-600 bg-neutral-700/50 text-neutral-200",
  ran: "border-neutral-600 bg-neutral-700/50 text-neutral-200",
  // settled, not good
  crash: "border-red-800/70 bg-red-500/15 text-red-200",
  failed: "border-red-800/70 bg-red-500/15 text-red-200",
  error: "border-red-800/70 bg-red-500/15 text-red-200",
  timeout: "border-orange-700/60 bg-orange-500/15 text-orange-200",
  cost_fail: "border-orange-700/60 bg-orange-500/15 text-orange-200",
  stalled: "border-amber-700/60 bg-amber-500/15 text-amber-200",
  // retired
  discarded: "border-neutral-800 bg-neutral-900 text-neutral-500",
  killed: "border-neutral-800 bg-neutral-900 text-neutral-500",
  stopped: "border-neutral-800 bg-neutral-900 text-neutral-500",
  frozen: "border-neutral-700 bg-neutral-800/80 text-neutral-400",
  idle: "border-neutral-800 bg-neutral-900 text-neutral-500",
};

const LIVE = new Set(["running", "implementing", "baseline", "holdout", "queued"]);

export function Badge({ status, children }: { status: string; children?: React.ReactNode }) {
  const cls = STATUS_COLORS[status] ?? "border-neutral-700 bg-neutral-800/80 text-neutral-300";
  return (
    <span
      className={`mono inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] leading-4 ${cls}`}
    >
      {LIVE.has(status) && (
        <span className="live-dot h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      )}
      {children ?? status}
    </span>
  );
}

export function Panel({
  title,
  subtitle,
  right,
  padded = true,
  children,
}: {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  padded?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900/50 shadow-lg shadow-black/20">
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 border-b border-neutral-800 bg-neutral-900/60 px-4 py-2.5">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-neutral-100">{title}</h2>
            {subtitle && <p className="truncate text-xs text-neutral-500">{subtitle}</p>}
          </div>
          <div className="flex shrink-0 items-center gap-2">{right}</div>
        </header>
      )}
      <div className={padded ? "p-4" : ""}>{children}</div>
    </section>
  );
}

/** A labelled number. The ledger and the loop strip are built from these. */
export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2">
      <p className="text-[11px] uppercase tracking-wider text-neutral-500">{label}</p>
      <p className="mono nums mt-0.5 truncate text-sm text-neutral-100">{value}</p>
      {hint && <p className="truncate text-[11px] text-neutral-500">{hint}</p>}
    </div>
  );
}

export function KeyValue({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-neutral-800/60 py-1 last:border-0">
      <dt className="shrink-0 text-xs text-neutral-500">{k}</dt>
      <dd className="mono nums min-w-0 truncate text-xs text-neutral-200">{v}</dd>
    </div>
  );
}

export function Empty({ title, children }: { title?: string; children?: React.ReactNode }) {
  return (
    <div className="px-4 py-10 text-center">
      {title && <p className="text-sm font-medium text-neutral-300">{title}</p>}
      {children && (
        <p className="mx-auto mt-1 max-w-md text-sm leading-relaxed text-neutral-500">
          {children}
        </p>
      )}
    </div>
  );
}

export function ErrorNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="flex items-start gap-2 rounded-lg border border-red-900/70 bg-red-950/40 px-3 py-2 text-sm text-red-200">
      <span aria-hidden className="mt-px select-none text-red-400">
        !
      </span>
      <span className="min-w-0 break-words">{children}</span>
    </p>
  );
}

export function fmt(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}

export function signed(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined) return "—";
  return `${n > 0 ? "+" : ""}${Number(n).toFixed(digits)}`;
}
