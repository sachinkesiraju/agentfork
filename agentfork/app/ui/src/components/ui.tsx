import React from "react";

export function Button({
  children,
  variant = "default",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger" | "ghost";
}) {
  const styles: Record<string, string> = {
    default:
      "border-neutral-700 bg-neutral-900 hover:bg-neutral-800 text-neutral-200",
    primary: "border-emerald-700 bg-emerald-800/70 hover:bg-emerald-700 text-white",
    danger: "border-red-800 bg-red-900/60 hover:bg-red-800 text-red-100",
    ghost: "border-transparent hover:bg-neutral-800 text-neutral-400",
  };
  return (
    <button
      {...rest}
      className={`rounded border px-3 py-1.5 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]} ${rest.className ?? ""}`}
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
    <label className="block space-y-1">
      <span className="text-xs uppercase tracking-wide text-neutral-500">{label}</span>
      {children}
      {hint && <span className="block text-xs text-neutral-500">{hint}</span>}
    </label>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`mono w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-neutral-500 ${props.className ?? ""}`}
    />
  );
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={`w-full rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-neutral-500 ${props.className ?? ""}`}
    />
  );
}

const STATUS_COLORS: Record<string, string> = {
  baseline: "bg-sky-900/60 text-sky-200 border-sky-800",
  proposed: "bg-neutral-800 text-neutral-300 border-neutral-700",
  implementing: "bg-amber-900/50 text-amber-200 border-amber-800",
  ready: "bg-neutral-800 text-neutral-200 border-neutral-600",
  running: "bg-blue-900/50 text-blue-200 border-blue-800",
  ran: "bg-neutral-800 text-neutral-200 border-neutral-600",
  kept: "bg-emerald-900/60 text-emerald-200 border-emerald-700",
  discarded: "bg-neutral-900 text-neutral-500 border-neutral-800",
  cost_fail: "bg-orange-900/50 text-orange-200 border-orange-800",
  crash: "bg-red-900/60 text-red-200 border-red-800",
  done: "bg-emerald-900/60 text-emerald-200 border-emerald-700",
  failed: "bg-red-900/60 text-red-200 border-red-800",
  killed: "bg-neutral-900 text-neutral-400 border-neutral-800",
  timeout: "bg-orange-900/50 text-orange-200 border-orange-800",
  queued: "bg-neutral-800 text-neutral-400 border-neutral-700",
};

export function Badge({ status }: { status: string }) {
  const cls = STATUS_COLORS[status] ?? "bg-neutral-800 text-neutral-300 border-neutral-700";
  return (
    <span className={`mono rounded border px-1.5 py-0.5 text-[11px] ${cls}`}>{status}</span>
  );
}

export function Panel({
  title,
  right,
  children,
}: {
  title?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40">
      {(title || right) && (
        <header className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
          <h2 className="text-sm font-medium text-neutral-300">{title}</h2>
          <div className="flex items-center gap-2">{right}</div>
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="py-6 text-center text-sm text-neutral-500">{children}</p>;
}

export function fmt(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}
