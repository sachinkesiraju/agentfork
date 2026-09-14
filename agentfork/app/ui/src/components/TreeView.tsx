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

const COLOR: Record<string, string> = {
  baseline: "#0c4a6e",
  proposed: "#262626",
  implementing: "#78350f",
  ready: "#404040",
  running: "#1e3a8a",
  ran: "#404040",
  kept: "#065f46",
  discarded: "#1c1c1c",
  cost_fail: "#7c2d12",
  crash: "#7f1d1d",
};

const COL_W = 230;
const ROW_H = 96;

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
        flowNodes.push({
          id: n.id,
          position: { x: i * COL_W - width / 2, y: gen * ROW_H },
          data: {
            label: (
              <div className="w-[190px] text-left">
                <div className="truncate text-xs font-medium">{n.title}</div>
                <div className="mono text-[10px] opacity-70">
                  {n.status}
                  {n.score !== null ? ` · ${Number(n.score).toFixed(4)}` : ""}
                  {n.delta !== null ? ` (${n.delta > 0 ? "+" : ""}${Number(n.delta).toFixed(4)})` : ""}
                </div>
              </div>
            ),
          },
          style: {
            background: COLOR[n.status] ?? "#262626",
            color: "#e5e5e5",
            border:
              selected === n.id
                ? "2px solid #e5e5e5"
                : isFrontier
                  ? "2px solid #34d399"
                  : "1px solid #404040",
            borderRadius: 8,
            padding: 8,
            width: 206,
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
        style: { stroke: "#525252" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#525252" },
      }));
    return { flowNodes, edges };
  }, [nodes, frontier, selected]);

  return (
    <div className="h-[520px] w-full rounded border border-neutral-800">
      <ReactFlow
        nodes={flowNodes}
        edges={edges}
        fitView
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => onSelect(n.id)}
      >
        <Background color="#262626" gap={18} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
