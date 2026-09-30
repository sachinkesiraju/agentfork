"""Deterministic bookkeeping for the agentic map-reduce loop.

A library port of ``amr.py`` from ``sachinkesiraju/agent-mapreduce``: the agent
reasons, this module does arithmetic. State lives entirely in an append-only
``results.tsv`` (byte-compatible with the original, so a run started with the
standalone script resumes here and vice versa). Stdlib only.

Row schema (tab separated)::

    gen  commit  parent  score  costs  regions  status  description

``status`` is ``ran`` (numeric score), ``crash`` (score ``-``) or ``note``
(a written-down law such as ``LAW: wider models OOM on this GPU``).

``reduce`` is the selection rule: a candidate survives only by beating *its
own parent* by more than the noise margin and passing every cost guard;
survivors are ranked globally and the top ``beam`` become the next frontier.
Disjoint-region survivors yield a ``FUSE_CANDIDATE``; no survivor is a
``STALL`` (frontier unchanged).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

HEADER = "gen\tcommit\tparent\tscore\tcosts\tregions\tstatus\tdescription"
OLD_HEADER = "gen\tcommit\tparent\tscore\tstatus\tdescription"
STATUSES = ("ran", "crash", "note")


class AmrError(ValueError):
    """Malformed input or results file."""


def clean(s: str) -> str:
    return s.replace("\t", " ").replace("\n", " ").strip()


def parse_cost_list(items: list[str]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for item in items:
        for part in item.split(","):
            if not part:
                continue
            if "=" not in part:
                raise AmrError(f"cost must be name=value, got {part!r}")
            k, v = part.split("=", 1)
            try:
                costs[clean(k)] = float(v)
            except ValueError:
                raise AmrError(f"cost value must be numeric, got {part!r}") from None
    return costs


def format_costs(costs: dict[str, float]) -> str:
    return ",".join(f"{k}={v:g}" for k, v in costs.items())


def parse_regions(items: list[str]) -> set[str]:
    regions: set[str] = set()
    for item in items:
        regions |= {clean(p) for p in item.split(",") if clean(p)}
    return regions


def parse_guards(items: list[str]) -> dict[str, float]:
    guards: dict[str, float] = {}
    for item in items:
        if ":+" not in item:
            raise AmrError(f"cost guard must be name:+tolerance, got {item!r}")
        k, v = item.split(":+", 1)
        try:
            guards[clean(k)] = float(v)
        except ValueError:
            raise AmrError(f"cost tolerance must be numeric, got {item!r}") from None
    return guards


@dataclass
class Row:
    gen: int
    commit: str
    parent: str
    score: float | None
    costs: dict[str, float] = field(default_factory=dict)
    regions: set[str] = field(default_factory=set)
    status: str = "ran"
    desc: str = ""

    def to_dict(self) -> dict:
        return {"gen": self.gen, "commit": self.commit, "parent": self.parent,
                "score": self.score, "costs": dict(self.costs),
                "regions": sorted(self.regions), "status": self.status,
                "description": self.desc}

    def line(self) -> str:
        score = "-" if self.score is None else f"{self.score}"
        return (f"{self.gen}\t{self.commit}\t{self.parent}\t{score}\t"
                f"{format_costs(self.costs)}\t{','.join(sorted(self.regions))}\t"
                f"{self.status}\t{clean(self.desc)}")


class Results:
    """The append-only ``results.tsv`` ledger."""

    def __init__(self, path: str | Path = "results.tsv"):
        self.path = Path(path)

    # -- io ------------------------------------------------------------------

    def ensure_header(self) -> None:
        if not self.path.exists():
            self.path.write_text(HEADER + "\n")
            return
        lines = self.path.read_text().splitlines()
        if not lines:
            self.path.write_text(HEADER + "\n")
            return
        if lines[0] == HEADER:
            return
        if lines[0] != OLD_HEADER:
            raise AmrError(f"{self.path}: bad header, expected: {HEADER}")
        upgraded = [HEADER]
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 6:
                raise AmrError("cannot upgrade old results.tsv: expected 6 columns")
            gen, commit, parent, score, status, desc = parts
            upgraded.append(f"{gen}\t{commit}\t{parent}\t{score}\t\t\t{status}\t{desc}")
        self.path.write_text("\n".join(upgraded) + "\n")

    def rows(self) -> list[Row]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        if not lines:
            return []
        old = lines[0] == OLD_HEADER
        if lines[0] not in (HEADER, OLD_HEADER):
            raise AmrError(f"{self.path}: bad header, expected: {HEADER}")
        rows = []
        for n, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            parts = line.split("\t")
            if old:
                if len(parts) != 6:
                    raise AmrError(f"{self.path}:{n}: expected 6 columns, got {len(parts)}")
                gen, commit, parent, score, status, desc = parts
                costs, regions = "", ""
            else:
                if len(parts) != 8:
                    raise AmrError(f"{self.path}:{n}: expected 8 columns, got {len(parts)}")
                gen, commit, parent, score, costs, regions, status, desc = parts
            rows.append(Row(
                gen=int(gen), commit=commit, parent=parent,
                score=None if score == "-" else float(score),
                costs=parse_cost_list([costs]) if costs else {},
                regions=parse_regions([regions]) if regions else set(),
                status=status, desc=desc))
        return rows

    def log(self, gen: int, commit: str, parent: str, score: float | None,
            status: str, description: str, *,
            costs: dict[str, float] | None = None,
            regions: set[str] | None = None) -> Row:
        if status not in STATUSES:
            raise AmrError(f"status must be one of {STATUSES}")
        if status == "ran" and score is None:
            raise AmrError("status 'ran' requires a numeric score")
        row = Row(gen, clean(commit), clean(parent), score, dict(costs or {}),
                  set(regions or ()), status, clean(description))
        self.ensure_header()
        with self.path.open("a") as f:
            f.write(row.line() + "\n")
        return row

    # -- selection -----------------------------------------------------------

    @staticmethod
    def node_best(rows: list[Row], minimize: bool) -> dict[str, Row]:
        best: dict[str, Row] = {}
        for r in rows:
            if r.status != "ran" or r.score is None:
                continue
            cur = best.get(r.commit)
            if cur is None or (r.score < cur.score if minimize else r.score > cur.score):
                best[r.commit] = r
        return best

    def reduce(self, gen: int, margin: float, *, beam: int = 2,
               minimize: bool = True,
               guards: dict[str, float] | None = None) -> "Reduction":
        guards = dict(guards or {})
        rows = self.rows()
        best = self.node_best(rows, minimize)
        cands: dict[str, Row] = {}
        for r in rows:
            if r.gen == gen and r.status == "ran" and r.commit not in cands:
                cands[r.commit] = r
        if not cands:
            raise AmrError(f"no 'ran' rows for gen {gen}")
        red = Reduction(gen=gen, margin=margin, beam=beam, minimize=minimize,
                        candidates=len(cands), guards=guards)
        survivors: list[Survivor] = []
        parents: dict[str, float] = {}
        for c, r in cands.items():
            p = r.parent
            if p == "-":
                continue
            if p not in best:
                red.warnings.append(f"{c}: parent {p} has no logged score; skipping")
                continue
            parents[p] = best[p].score  # type: ignore[assignment]
            child, parent = best[c], best[p]
            delta = child.score - parent.score  # type: ignore[operator]
            if (-delta if minimize else delta) <= margin:
                red.discarded.append(Survivor(c, child.score, p, delta,
                                              set(child.regions), r.desc))
                continue
            ok, reason = _cost_ok(child, parent, guards)
            if not ok:
                red.cost_failures.append((c, reason, r.desc))
                continue
            survivors.append(Survivor(c, child.score, p, delta,
                                      set(child.regions), r.desc))
        survivors.sort(key=lambda s: s.score, reverse=not minimize)
        red.beat_parent = len(survivors)
        red.kept = survivors[:beam]
        red.dropped = survivors[beam:]
        if red.kept:
            red.frontier = [s.commit for s in red.kept]
            for i, a in enumerate(red.kept):
                for b in red.kept[i + 1:]:
                    if a.regions and b.regions and not (a.regions & b.regions):
                        red.fuse.append((a.commit, b.commit,
                                         sorted(a.regions | b.regions)))
        else:
            red.stall = True
            red.frontier = sorted(parents, key=parents.get, reverse=not minimize)
        return red

    def tree(self, *, minimize: bool = True) -> "Tree":
        rows = self.rows()
        scores = {c: r.score for c, r in self.node_best(rows, minimize).items()}
        nodes: dict[str, Row] = {}
        children: dict[str, list[str]] = {}
        for r in rows:
            if r.status == "note" or r.commit in nodes:
                continue
            nodes[r.commit] = r
            children.setdefault(r.parent, []).append(r.commit)
        roots = [c for c, r in nodes.items()
                 if r.parent == "-" or r.parent not in nodes]
        notes = [r for r in rows if r.status == "note"]
        return Tree(nodes, children, roots, scores, notes)


def _cost_ok(child: Row, parent: Row, guards: dict[str, float]) -> tuple[bool, str]:
    for name, tol in guards.items():
        if name not in child.costs:
            return False, f"missing child cost {name}"
        if name not in parent.costs:
            return False, f"missing parent cost {name}"
        limit = parent.costs[name] * (1 + tol)
        if child.costs[name] > limit:
            return False, f"{name} {child.costs[name]:g} > {limit:g}"
    return True, ""


@dataclass
class Survivor:
    commit: str
    score: float
    parent: str
    delta: float
    regions: set[str]
    desc: str

    def to_dict(self) -> dict:
        return {"commit": self.commit, "score": self.score, "parent": self.parent,
                "delta": self.delta, "regions": sorted(self.regions),
                "description": self.desc}


@dataclass
class Reduction:
    gen: int
    margin: float
    beam: int
    minimize: bool
    candidates: int
    guards: dict[str, float]
    beat_parent: int = 0
    kept: list[Survivor] = field(default_factory=list)
    dropped: list[Survivor] = field(default_factory=list)
    discarded: list[Survivor] = field(default_factory=list)
    cost_failures: list[tuple[str, str, str]] = field(default_factory=list)
    fuse: list[tuple[str, str, list[str]]] = field(default_factory=list)
    frontier: list[str] = field(default_factory=list)
    stall: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"gen": self.gen, "margin": self.margin, "beam": self.beam,
                "minimize": self.minimize, "candidates": self.candidates,
                "beat_parent": self.beat_parent,
                "kept": [s.to_dict() for s in self.kept],
                "dropped": [s.to_dict() for s in self.dropped],
                "discarded": [s.to_dict() for s in self.discarded],
                "cost_failures": [{"commit": c, "reason": r, "description": d}
                                  for c, r, d in self.cost_failures],
                "fuse": [{"a": a, "b": b, "regions": regs}
                         for a, b, regs in self.fuse],
                "frontier": list(self.frontier), "stall": self.stall,
                "warnings": list(self.warnings)}

    def render(self) -> str:
        """The exact text the standalone ``amr.py reduce`` prints."""
        out = []
        for c, reason, desc in self.cost_failures:
            out.append(f"COST_FAIL\t{c}\t{reason}\t{desc}")
        extra = (f", {len(self.cost_failures)} failed cost guards"
                 if self.guards else "")
        out.append(f"gen {self.gen}: {self.candidates} candidates, "
                   f"{self.beat_parent} beat their parent (margin "
                   f"{self.margin:g}{extra}), keeping {len(self.kept)}")
        for s in self.kept:
            region_text = (f"\tregions={','.join(sorted(s.regions))}"
                           if s.regions else "")
            out.append(f"KEEP\t{s.commit}\t{s.score:.6f}\t(parent {s.parent}, "
                       f"delta {s.delta:+.6f})\t{s.desc}{region_text}")
        if self.stall:
            out.append("STALL: no candidate beat its parent by more than the "
                       "margin. Frontier unchanged.")
        out.append("FRONTIER: " + " ".join(self.frontier))
        for a, b, regs in self.fuse:
            out.append(f"FUSE_CANDIDATE\t{a}\t{b}\tregions={','.join(regs)}")
        return "\n".join(out)


@dataclass
class Tree:
    nodes: dict[str, Row]
    children: dict[str, list[str]]
    roots: list[str]
    scores: dict[str, float]
    notes: list[Row]

    def render(self) -> str:
        if not self.nodes and not self.notes:
            return ""
        out: list[str] = []

        def render(c: str, prefix: str, last: bool, root: bool) -> None:
            s = f"{self.scores[c]:.6f}" if c in self.scores else "crash"
            connector = "" if root else ("└── " if last else "├── ")
            out.append(f"{prefix}{connector}{c}  {s}  {self.nodes[c].desc}")
            kids = self.children.get(c, [])
            ext = "" if root else ("    " if last else "│   ")
            for i, k in enumerate(kids):
                render(k, prefix + ext, i == len(kids) - 1, False)

        for root in self.roots:
            render(root, "", True, True)
        if self.notes:
            out.append("")
            out.append("notes/laws:")
            for r in self.notes:
                out.append(f"  gen {r.gen}: {r.desc}")
        return "\n".join(out)


# -- amr.py-compatible command line -------------------------------------------

def main(argv: list[str] | None = None, tsv: str | Path = "results.tsv") -> int:
    ap = argparse.ArgumentParser(
        prog="agentfork amr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=str(tsv), help="results file (default results.tsv)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("log", help="append one row to results.tsv")
    p.add_argument("--cost", action="append", default=[], help="name=value, repeatable")
    p.add_argument("--region", action="append", default=[], help="region name, repeatable")
    p.add_argument("gen", type=int)
    p.add_argument("commit")
    p.add_argument("parent")
    p.add_argument("score", help="metric value, or '-' for crash/note rows")
    p.add_argument("status", help="ran | crash | note")
    p.add_argument("description", nargs="+")

    p = sub.add_parser("reduce", help="pick the surviving frontier for a generation")
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--beam", type=int, default=2)
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--cost", action="append", default=[], help="name:+tolerance, repeatable")

    p = sub.add_parser("tree", help="print the search tree")
    p.add_argument("--maximize", action="store_true")

    args = ap.parse_args(argv)
    results = Results(args.tsv)
    try:
        if args.cmd == "log":
            if args.score != "-":
                try:
                    score: float | None = float(args.score)
                except ValueError:
                    raise AmrError(
                        f"score must be a float or '-', got {args.score!r}") from None
            else:
                score = None
            row = results.log(args.gen, args.commit, args.parent, score,
                              args.status, " ".join(args.description),
                              costs=parse_cost_list(args.cost),
                              regions=parse_regions(args.region))
            print(f"logged: gen {row.gen} {row.commit} {row.status} {row.desc}")
        elif args.cmd == "reduce":
            red = results.reduce(args.gen, args.margin, beam=args.beam,
                                 minimize=not args.maximize,
                                 guards=parse_guards(args.cost))
            for w in red.warnings:
                print(f"warn: {w}", file=sys.stderr)
            print(red.render())
        elif args.cmd == "tree":
            text = results.tree(minimize=not args.maximize).render()
            if not text:
                raise AmrError("no results.tsv yet")
            print(text)
    except AmrError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
