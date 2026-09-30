#!/usr/bin/env python3
"""amr.py -- deterministic bookkeeping for the agentic map-reduce loop (see program.md).

The agent reasons; this file does arithmetic. State lives entirely in
results.tsv (append-only) + git. Stdlib only, no config, no dependencies.

  python3 amr.py log [--cost name=value] [--region name] <gen> <commit> <parent> <score|-> <ran|crash|note> <description...>
  python3 amr.py reduce --gen N --margin M [--beam B] [--maximize] [--cost name:+tol]
  python3 amr.py tree [--maximize]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TSV = Path("results.tsv")
HEADER = "gen\tcommit\tparent\tscore\tcosts\tregions\tstatus\tdescription"
OLD_HEADER = "gen\tcommit\tparent\tscore\tstatus\tdescription"
STATUSES = ("ran", "crash", "note")


def clean(s: str) -> str:
    return s.replace("\t", " ").replace("\n", " ").strip()


def parse_cost_list(items: list[str]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for item in items:
        for part in item.split(","):
            if not part:
                continue
            if "=" not in part:
                sys.exit(f"cost must be name=value, got {part!r}")
            k, v = part.split("=", 1)
            try:
                costs[clean(k)] = float(v)
            except ValueError:
                sys.exit(f"cost value must be numeric, got {part!r}")
    return costs


def format_costs(costs: dict[str, float]) -> str:
    return ",".join(f"{k}={v:g}" for k, v in costs.items())


def parse_regions(items: list[str]) -> set[str]:
    regions: set[str] = set()
    for item in items:
        regions |= {clean(p) for p in item.split(",") if clean(p)}
    return regions


def ensure_tsv_header() -> None:
    if not TSV.exists():
        TSV.write_text(HEADER + "\n")
        return
    lines = TSV.read_text().splitlines()
    if not lines:
        TSV.write_text(HEADER + "\n")
        return
    if lines[0] == HEADER:
        return
    if lines[0] != OLD_HEADER:
        sys.exit(f"results.tsv: bad header, expected: {HEADER}")
    upgraded = [HEADER]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            sys.exit("cannot upgrade old results.tsv: expected 6 columns")
        gen, commit, parent, score, status, desc = parts
        upgraded.append(f"{gen}\t{commit}\t{parent}\t{score}\t\t\t{status}\t{desc}")
    TSV.write_text("\n".join(upgraded) + "\n")


def read_rows() -> list[dict]:
    if not TSV.exists():
        return []
    lines = TSV.read_text().splitlines()
    if not lines:
        return []
    old = lines[0] == OLD_HEADER
    if lines[0] not in (HEADER, OLD_HEADER):
        sys.exit(f"results.tsv: bad header, expected: {HEADER}")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if old:
            if len(parts) != 6:
                sys.exit(f"results.tsv:{n}: expected 6 tab-separated columns, got {len(parts)}")
            gen, commit, parent, score, status, desc = parts
            costs, regions = "", ""
        else:
            if len(parts) != 8:
                sys.exit(f"results.tsv:{n}: expected 8 tab-separated columns, got {len(parts)}")
            gen, commit, parent, score, costs, regions, status, desc = parts
        rows.append({
            "gen": int(gen), "commit": commit, "parent": parent,
            "score": None if score == "-" else float(score),
            "costs": parse_cost_list([costs]) if costs else {},
            "regions": parse_regions([regions]) if regions else set(),
            "status": status, "desc": desc,
        })
    return rows


def node_best(rows: list[dict], minimize: bool) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        if r["status"] != "ran" or r["score"] is None:
            continue
        cur = best.get(r["commit"])
        if cur is None or (r["score"] < cur["score"] if minimize else r["score"] > cur["score"]):
            best[r["commit"]] = r
    return best


def node_scores(rows: list[dict], minimize: bool) -> dict[str, float]:
    return {c: r["score"] for c, r in node_best(rows, minimize).items()}


def parse_guards(items: list[str]) -> dict[str, float]:
    guards: dict[str, float] = {}
    for item in items:
        if ":+" not in item:
            sys.exit(f"cost guard must be name:+tolerance, got {item!r}")
        k, v = item.split(":+", 1)
        try:
            guards[clean(k)] = float(v)
        except ValueError:
            sys.exit(f"cost tolerance must be numeric, got {item!r}")
    return guards


def cost_ok(child: dict, parent: dict, guards: dict[str, float]) -> tuple[bool, str]:
    for name, tol in guards.items():
        if name not in child["costs"]:
            return False, f"missing child cost {name}"
        if name not in parent["costs"]:
            return False, f"missing parent cost {name}"
        limit = parent["costs"][name] * (1 + tol)
        if child["costs"][name] > limit:
            return False, f"{name} {child['costs'][name]:g} > {limit:g}"
    return True, ""


def cmd_log(args) -> None:
    if args.status not in STATUSES:
        sys.exit(f"status must be one of {STATUSES}")
    if args.score != "-":
        try:
            float(args.score)
        except ValueError:
            sys.exit(f"score must be a float or '-', got {args.score!r}")
    elif args.status == "ran":
        sys.exit("status 'ran' requires a numeric score")
    costs = parse_cost_list(args.cost or [])
    regions = parse_regions(args.region or [])
    desc = clean(" ".join(args.description))
    ensure_tsv_header()
    with TSV.open("a") as f:
        f.write(f"{args.gen}\t{args.commit}\t{args.parent}\t{args.score}\t{format_costs(costs)}\t{','.join(sorted(regions))}\t{args.status}\t{desc}\n")
    print(f"logged: gen {args.gen} {args.commit} {args.status} {desc}")


def cmd_reduce(args) -> None:
    minimize = not args.maximize
    rows = read_rows()
    best = node_best(rows, minimize)
    scores = {c: r["score"] for c, r in best.items()}
    guards = parse_guards(args.cost or [])
    cands: dict[str, dict] = {}
    for r in rows:
        if r["gen"] == args.gen and r["status"] == "ran" and r["commit"] not in cands:
            cands[r["commit"]] = r
    if not cands:
        sys.exit(f"no 'ran' rows for gen {args.gen}")
    survivors, parents = [], {}
    cost_failures = 0
    for c, r in cands.items():
        p = r["parent"]
        if p == "-":
            continue
        if p not in best:
            print(f"warn: {c}: parent {p} has no logged score; skipping", file=sys.stderr)
            continue
        parents[p] = scores[p]
        child, parent = best[c], best[p]
        delta = child["score"] - parent["score"]
        if (-delta if minimize else delta) <= args.margin:
            continue
        ok, reason = cost_ok(child, parent, guards)
        if not ok:
            cost_failures += 1
            print(f"COST_FAIL\t{c}\t{reason}\t{r['desc']}")
            continue
        survivors.append((c, child["score"], p, delta, child["regions"], r["desc"]))
    survivors.sort(key=lambda t: t[1], reverse=not minimize)
    kept = survivors[:args.beam]
    extra = f", {cost_failures} failed cost guards" if guards else ""
    print(f"gen {args.gen}: {len(cands)} candidates, {len(survivors)} beat their parent "
          f"(margin {args.margin:g}{extra}), keeping {len(kept)}")
    for c, s, p, d, regions, desc in kept:
        region_text = f"\tregions={','.join(sorted(regions))}" if regions else ""
        print(f"KEEP\t{c}\t{s:.6f}\t(parent {p}, delta {d:+.6f})\t{desc}{region_text}")
    if kept:
        print("FRONTIER: " + " ".join(c for c, *_ in kept))
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                if a[4] and b[4] and not (a[4] & b[4]):
                    regs = ",".join(sorted(a[4] | b[4]))
                    print(f"FUSE_CANDIDATE\t{a[0]}\t{b[0]}\tregions={regs}")
    else:
        print("STALL: no candidate beat its parent by more than the margin. Frontier unchanged.")
        print("FRONTIER: " + " ".join(sorted(parents, key=parents.get, reverse=not minimize)))


def cmd_tree(args) -> None:
    rows = read_rows()
    if not rows:
        sys.exit("no results.tsv yet")
    scores = node_scores(rows, minimize=not args.maximize)
    nodes: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for r in rows:
        if r["status"] == "note" or r["commit"] in nodes:
            continue
        nodes[r["commit"]] = r
        children.setdefault(r["parent"], []).append(r["commit"])
    roots = [c for c, r in nodes.items() if r["parent"] == "-" or r["parent"] not in nodes]

    def render(c: str, prefix: str, last: bool, root: bool) -> None:
        s = f"{scores[c]:.6f}" if c in scores else "crash"
        connector = "" if root else ("└── " if last else "├── ")
        print(f"{prefix}{connector}{c}  {s}  {nodes[c]['desc']}")
        kids = children.get(c, [])
        ext = "" if root else ("    " if last else "│   ")
        for i, k in enumerate(kids):
            render(k, prefix + ext, i == len(kids) - 1, False)

    for root in roots:
        render(root, "", True, True)
    notes = [r for r in rows if r["status"] == "note"]
    if notes:
        print("\nnotes/laws:")
        for r in notes:
            print(f"  gen {r['gen']}: {r['desc']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("reduce", help="pick the surviving frontier for a generation")
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--beam", type=int, default=2)
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--cost", action="append", default=[], help="name:+tolerance, repeatable")
    p.set_defaults(func=cmd_reduce)

    p = sub.add_parser("tree", help="print the search tree")
    p.add_argument("--maximize", action="store_true")
    p.set_defaults(func=cmd_tree)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
