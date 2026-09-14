import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Empty } from "./ui";

/** Incremental log tail: polls with a byte offset so a long eval log is not
 *  re-fetched on every tick. */
export default function LogTerminal({ runId }: { runId: string | null }) {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("");
  const boxRef = useRef<HTMLPreElement>(null);
  const offset = useRef(0);

  useEffect(() => {
    offset.current = 0;
    setText("");
    if (!runId) return;
    let stop = false;
    async function tick() {
      if (stop || !runId) return;
      try {
        const out = await api.log(runId, offset.current);
        offset.current = out.offset;
        setStatus(out.alive ? "running" : `${out.status} (exit ${out.exit_code ?? "—"})`);
        if (out.text) setText((t) => t + out.text);
        if (!out.alive) return;
      } catch {
        /* run may not have a log yet */
      }
      setTimeout(tick, 700);
    }
    tick();
    return () => {
      stop = true;
    };
  }, [runId]);

  useEffect(() => {
    boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight });
  }, [text]);

  if (!runId)
    return (
      <Empty title="No run selected">
        Pick a run in the Runs tab, or click one from a node, to tail its output here.
      </Empty>
    );
  const live = status === "running";
  return (
    <div>
      <div className="flex items-center justify-between gap-3 border-b border-neutral-800 bg-neutral-950/60 px-4 py-2">
        <span className="mono truncate text-xs text-neutral-400">{runId}</span>
        <span className="mono flex shrink-0 items-center gap-1.5 text-xs text-neutral-400">
          {live && <span className="live-dot h-1.5 w-1.5 rounded-full bg-blue-400" />}
          {status}
        </span>
      </div>
      <pre
        ref={boxRef}
        className="mono h-[460px] overflow-auto whitespace-pre-wrap break-words bg-[#08080a] p-4 text-xs leading-relaxed text-neutral-300"
      >
        {text || <span className="text-neutral-600">waiting for output…</span>}
      </pre>
    </div>
  );
}
