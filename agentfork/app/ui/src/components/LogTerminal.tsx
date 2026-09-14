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

  if (!runId) return <Empty>Select a run to see its log.</Empty>;
  return (
    <div className="space-y-2">
      <div className="mono flex items-center justify-between text-xs text-neutral-500">
        <span>{runId}</span>
        <span>{status}</span>
      </div>
      <pre
        ref={boxRef}
        className="mono h-[420px] overflow-auto whitespace-pre-wrap rounded border border-neutral-800 bg-black/60 p-3 text-xs text-neutral-300"
      >
        {text || "(no output yet)"}
      </pre>
    </div>
  );
}
