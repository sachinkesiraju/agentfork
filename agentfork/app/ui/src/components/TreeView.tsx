import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Node as FlowNode,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { Node } from "../api";

/** Fill + accent per status. The accent is the left rail on the card, so the
 *  status is readable at a glance even when the tree is zoomed out. */
const STYLE: Record<string, { bg: string; accent: string; dim?: boolean }> = {
  baseline: { bg: "#0b2c3f", accent: "#38bdf8" },
  proposed: { bg: "#1b1b1f", accent: "#52525b" },
  implementing: { bg: "#3a2a0c", accent: "#f59e0b" },
  ready: { bg: "#27272b", accent: "#a1a1aa" },
  running: { bg: "#142046", accent: "#60a5fa" },
  ran: { bg: "#27272b", accent: "#a1a1aa" },
  kept: { bg: "#0a3b2e", accent: "#34d399" },
  discarded: { bg: "#141416", accent: "#3f3f46", dim: true },
  cost_fail: { bg: "#3d1a0b", accent: "#fb923c" },
  crash: { bg: "#3d1213", accent: "#f87171" },
};

const COL_W = 236;
const ROW_H = 108;

/** Layered layout: generation on the y axis, siblings spread on x — the
 *  "stacked bushes" shape the map-reduce program wants to make obvious. */
export default function TreeView({
  nodes,
  frontier,
  selected,
  onSelect,
}: {
  nodes: Node[];
  frontier: string[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const { flowNodes, edges } = useMemo(() => {
    const byGen = new Map<number, Node[]>();
    nodes.forEach((n) => {
      const list = byGen.get(n.gen) ?? [];
      list.push(n);
      byGen.set(n.gen, list);
    });
    const flowNodes: FlowNode[] = [];
    byGen.forEach((list, gen) => {
      const width = (list.length - 1) * COL_W;
      list.forEach((n, i) => {
        const isFrontier = frontier.includes(n.id);
        const isSelected = selected === n.id;
        const style = STYLE[n.status] ?? STYLE.proposed;
        flowNodes.push({
          id: n.id,
          position: { x: i * COL_W - width / 2, y: gen * ROW_H },
          data: {
            label: (
              <div
                className="flex w-[194px] items-stretch gap-2 text-left"
                style={{ opacity: style.dim ? 0.55 : 1 }}
              >
                <span
                  className="w-[3px] shrink-0 rounded-full"
                  style={{ background: style.accent }}
                />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-medium leading-tight text-neutral-50">
                    {n.title}
                  </div>
                  <div className="mt-1 flex items-baseline justify-between gap-2">
                    <span
                      className="mono text-[10px] uppercase tracking-wide"
                      style={{ color: style.accent }}
                    >
                      {n.status}
                    </span>
                    <span className="mono nums text-[11px] text-neutral-200">
                      {n.score !== null ? Number(n.score).toFixed(4) : "—"}
                    </span>
                  </div>
                  {n.delta !== null && (
                    <div className="mono nums text-[10px] leading-tight text-neutral-400">
                      {n.delta > 0 ? "+" : ""}
                      {Number(n.delta).toFixed(4)} vs parent
                    </div>
                  )}
                </div>
              </div>
            ),
          },
          style: {
            background: style.bg,
            color: "#e5e5e5",
            border: isSelected
              ? "1.5px solid #fafafa"
              : isFrontier
                ? "1.5px solid #34d399"
                : "1px solid #2f2f36",
            boxShadow: isSelected
              ? "0 0 0 3px rgba(250,250,250,0.12)"
              : isFrontier
                ? "0 0 0 3px rgba(52,211,153,0.12)"
                : "0 1px 2px rgba(0,0,0,0.4)",
            borderRadius: 10,
            padding: 10,
            width: 214,
            fontSize: 12,
          },
        });
      });
    });
    const edges: Edge[] = nodes
      .filter((n) => n.parent_id)
      .map((n) => ({
        id: `${n.parent_id}-${n.id}`,
        source: n.parent_id!,
        target: n.id,
        type: "smoothstep",
        animated: n.status === "running" || n.status === "implementing",
        style: { stroke: frontier.includes(n.id) ? "#34d399" : "#3f3f46", strokeWidth: 1.5 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: frontier.includes(n.id) ? "#34d399" : "#3f3f46",
          width: 14,
          height: 14,
        },
      }));
    return { flowNodes, edges };
  }, [nodes, frontier, selected]);

  return (
    <div className="h-[540px] w-full bg-neutral-950/40">
      <ReactFlow
        nodes={flowNodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.2}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => onSelect(n.id)}
      >
        <Background color="#2a2a30" gap={20} size={1} />
        <Controls
          showInteractive={false}
          className="!border !border-neutral-800 !bg-neutral-900 [&>button]:!border-neutral-800 [&>button]:!bg-neutral-900 [&>button]:!fill-neutral-400 [&>button:hover]:!bg-neutral-800"
        />
      </ReactFlow>
    </div>
  );
}
