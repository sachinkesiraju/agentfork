import React from "react";
import { Link } from "react-router-dom";

/** The mark: a trunk that forks. Small enough to read at 20px. */
export function Logo({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={`h-5 w-5 ${className}`} aria-hidden>
      <path
        d="M12 21V13M12 13C12 10 8 10 6 8M12 13c0-3 4-3 6-5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <circle cx="6" cy="6" r="2.2" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="18" cy="6" r="2.2" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="12" cy="21" r="1.6" fill="currentColor" />
    </svg>
  );
}

/** Sticky top bar shared by every route, so the product always has a frame. */
export function TopBar({ children }: { children?: React.ReactNode }) {
  return (
    <div className="sticky top-0 z-20 border-b border-neutral-800/80 bg-neutral-950/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-6">
        <Link
          to="/"
          className="flex items-center gap-2 text-neutral-200 transition-colors hover:text-white"
        >
          <Logo className="text-emerald-400" />
          <span className="text-sm font-semibold tracking-tight">agentfork</span>
        </Link>
        <div className="flex min-w-0 items-center gap-2">{children}</div>
      </div>
    </div>
  );
}

export function Page({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-7xl space-y-5 px-6 py-6">{children}</div>;
}

export function LocalFooter({ home }: { home: string }) {
  if (!home) return null;
  return (
    <footer className="flex flex-wrap items-center gap-x-2 gap-y-1 pt-2 text-xs text-neutral-600">
      <span className="inline-flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500/80" />
        loopback only — nothing leaves this machine
      </span>
      <span className="text-neutral-700">·</span>
      <span className="mono truncate">{home}</span>
    </footer>
  );
}
