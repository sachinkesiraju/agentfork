"""``agentfork`` — the executable.

The dashboard owns the state; the CLI is a thin client over the same HTTP API
(the OpenResearch split: ``agentfork up`` runs the product, the other
subcommands inspect and drive it — so `agentfork projects`, `tree`, `runs`,
`logs` work whether you use the UI or not).

    agentfork up [--port N]            start dashboard + API
    agentfork status                   is a dashboard up?
    agentfork harnesses                what agents are detected
    agentfork projects                 list projects
    agentfork new <path> [--baseline BR] [--harness H] [--eval CMD] [--metric G]
    agentfork project <id>             project detail
    agentfork baseline <project>       create root + queue baseline evals
    agentfork tree <project>           experiment tree (rendered)
    agentfork fanout <project> <node> [-k N]  propose + fan out (alias: descend)
    agentfork reduce <project> --gen N [--margin M]
    agentfork runs <project>
    agentfork logs <run> [-f]
    agentfork kill <run>
    agentfork laws <project>
    agentfork loop <project> start|stop|status
    agentfork metrics <project>        KV + orchestrator counters
    agentfork install-skills [dir]
    agentfork skill                    the agentfork usage skill (stdout)
    agentfork amr ...                  the vendored bookkeeping tool
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentfork.app import amr

DEFAULT_PORT = 8474


def _api(port: int, method: str, path: str, body: dict | None = None,
         timeout: float = 30.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"}
    token = os.environ.get("AGENTFORK_TOKEN")
    if token:  # required when the dashboard is bound off loopback
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            msg = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            msg = detail
        raise SystemExit(f"error: {msg}") from None
    except urllib.error.URLError:
        raise SystemExit(
            f"no dashboard on port {port} — start one with `agentfork up`") \
            from None


def _print(obj) -> None:
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def _kv_table(d: dict, keys: list[str]) -> None:
    for k in keys:
        if k in d:
            print(f"{k:>16}: {d[k]}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `agentfork amr` is a passthrough: everything after it is amr's own
    # args. The split must happen at the subcommand position — the first
    # non-flag token — or a repo path/value equal to "amr" hijacks the parse.
    amr_args: list[str] | None = None
    i = 0
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in ("--port", "--home") else 1
    if i < len(argv) and argv[i] == "amr":
        argv, amr_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="agentfork", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--home", type=Path, default=None,
                    help="data dir (default ~/.agentfork)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("up", help="start the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)

    sub.add_parser("status")
    sub.add_parser("harnesses")
    sub.add_parser("projects")

    p = sub.add_parser("new", help="register a repo as a project")
    p.add_argument("path")
    p.add_argument("--name")
    p.add_argument("--baseline", default="main")
    p.add_argument("--harness", default="claude-code",
                   choices=["claude-code", "codex", "opencode",
                            "cursor-agent", "api", "fake"])
    p.add_argument("--model", default="",
                   help="model id passed to the harness (e.g. "
                   "--harness api --model qwen3:14b)")
    p.add_argument("--api-base", dest="api_base", default="",
                   help="harness=api only: OpenAI-compatible endpoint "
                   "(e.g. http://127.0.0.1:11434/v1 for Ollama)")
    p.add_argument("--eval", dest="eval_cmd", required=True)
    p.add_argument("--metric", dest="metric_grep", required=True,
                   help="regex over eval log; first group = the number")
    p.add_argument("--maximize", action="store_true")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("-b", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-gens", type=int, default=3)
    p.add_argument("--holdout", dest="holdout_cmd", default="")
    p.add_argument("--eval-slots", type=int, default=2,
                   help="max concurrent eval processes")
    p.add_argument("--cost", dest="costs", action="append", default=[],
                   metavar="NAME=TOL", help="cost guard, repeatable "
                   "(e.g. --cost seconds=0.5)")
    p.add_argument("--baseline-runs", type=int, default=2)

    p = sub.add_parser("project")
    p.add_argument("id")
    p = sub.add_parser("baseline")
    p.add_argument("project")
    p = sub.add_parser("tree")
    p.add_argument("project")
    p = sub.add_parser("fanout", aliases=["descend"])
    p.add_argument("project")
    p.add_argument("node")
    p.add_argument("-k", type=int, default=None)
    p = sub.add_parser("reduce")
    p.add_argument("project")
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--margin", type=float, default=None)
    p = sub.add_parser("runs")
    p.add_argument("project")
    p = sub.add_parser("logs")
    p.add_argument("run")
    p.add_argument("-f", "--follow", action="store_true")
    p = sub.add_parser("kill")
    p.add_argument("run")
    p = sub.add_parser("laws")
    p.add_argument("project")
    p = sub.add_parser("metrics")
    p.add_argument("project")
    p = sub.add_parser("loop")
    p.add_argument("project")
    p.add_argument("action", choices=["start", "stop", "status"])
    p = sub.add_parser("install-skills")
    p.add_argument("dir", nargs="?", default=None,
                   help="skills root (default: .claude/skills or agent dir)")
    sub.add_parser("skill", help="print the agentfork usage skill")
    sub.add_parser("amr", help="vendored agent-mapreduce bookkeeping")

    args = ap.parse_args(argv + (["amr"] if amr_args is not None else []))
    port = getattr(args, "port", DEFAULT_PORT)

    if args.cmd == "up":
        from agentfork.app.server import serve
        serve(home=args.home, host=args.host, port=args.port,
              open_browser=not args.no_browser)
        return 0
    if args.cmd == "status":
        print(json.dumps(_api(port, "GET", "/api/health"), indent=2))
        return 0
    if args.cmd == "harnesses":
        print(json.dumps(_api(port, "GET", "/api/harnesses"), indent=2))
        return 0
    if args.cmd == "projects":
        for p in _api(port, "GET", "/api/projects"):
            print(f"{p['id']}  {p['name']:<20} {p['repo_path']}  "
                  f"[{p['baseline_branch']}]")
        return 0
    if args.cmd == "new":
        try:
            cost_guards = {k.strip(): float(v)
                           for k, v in (c.split("=", 1) for c in args.costs)}
        except ValueError:
            raise SystemExit("error: --cost wants NAME=TOLERANCE pairs") \
                from None
        params = {"eval_cmd": args.eval_cmd, "metric_grep": args.metric_grep,
                  "minimize": not args.maximize, "k": args.k, "b": args.b,
                  "timeout_s": args.timeout, "max_gens": args.max_gens,
                  "holdout_cmd": args.holdout_cmd,
                  "eval_slots": args.eval_slots,
                  "baseline_runs": args.baseline_runs,
                  "cost_guards": cost_guards,
                  "model": args.model, "api_base": args.api_base}
        proj = _api(port, "POST", "/api/projects",
                    {"name": args.name or Path(args.path).resolve().name,
                     "repo_path": str(Path(args.path).resolve()),
                     "baseline_branch": args.baseline,
                     "harness": args.harness, "params": params})
        print(f"created {proj['id']} — next: agentfork baseline {proj['id']}")
        return 0
    if args.cmd == "project":
        _print(_api(port, "GET", f"/api/projects/{args.id}"))
        return 0
    if args.cmd == "baseline":
        _print(_api(port, "POST", f"/api/projects/{args.project}/baseline", {}))
        return 0
    if args.cmd == "tree":
        r = _api(port, "GET", f"/api/projects/{args.project}/results")
        loop = _api(port, "GET", f"/api/projects/{args.project}")["loop"]
        print(f"loop: {loop['state']} gen={loop['gen']} "
              f"margin={loop['margin']} frontier={loop['frontier']}")
        print(r["tree"] or "(no results yet)")
        return 0
    if args.cmd in ("fanout", "descend"):
        body = {"parent_id": args.node}
        if args.k:
            body["k"] = args.k
        out = _api(port, "POST", f"/api/projects/{args.project}/fanout", body)
        for n in out["nodes"]:
            print(f"{n['id']}  gen{n['gen']}  {n['title']}")
        return 0
    if args.cmd == "reduce":
        body = {"gen": args.gen}
        if args.margin is not None:
            body["margin"] = args.margin
        out = _api(port, "POST", f"/api/projects/{args.project}/reduce", body)
        print(_render_reduction(out))
        return 0
    if args.cmd == "runs":
        for r in _api(port, "GET", f"/api/projects/{args.project}/runs"):
            print(f"{r['id']}  {r['kind']:<7} {r['status']:<8} "
                  f"node={r['node_id']} score={r['score']} "
                  f"exit={r['exit_code']}")
        return 0
    if args.cmd == "logs":
        offset = 0
        while True:
            out = _api(port, "GET",
                       f"/api/runs/{args.run}/log?offset={offset}")
            if out["text"]:
                print(out["text"], end="")
            offset = out["offset"]
            if not args.follow or not out["alive"]:
                break
            time.sleep(0.5)
        return 0
    if args.cmd == "kill":
        _print(_api(port, "POST", f"/api/runs/{args.run}/kill", {}))
        return 0
    if args.cmd == "laws":
        for law in _api(port, "GET", f"/api/projects/{args.project}/laws"):
            print(f"gen {law['gen']}: {law['text']}")
        return 0
    if args.cmd == "metrics":
        _print(_api(port, "GET", f"/api/projects/{args.project}/metrics"))
        return 0
    if args.cmd == "loop":
        if args.action == "status":
            _print(_api(port, "GET", f"/api/projects/{args.project}/loop"))
        else:
            _print(_api(port, "POST",
                        f"/api/projects/{args.project}/loop/{args.action}", {}))
        return 0
    if args.cmd == "install-skills":
        return _install_skills(args.dir)
    if args.cmd == "skill":
        print((Path(__file__).parent / "skills" / "agentfork" / "SKILL.md")
              .read_text())
        return 0
    if args.cmd == "amr":
        return amr.main(amr_args or [])
    return 1


def _install_skills(dest: str | None) -> int:
    src = Path(__file__).parent / "skills"
    if dest is None:
        dest = ".claude/skills" if Path(".claude").is_dir() else ".agents/skills"
    root = Path(dest)
    for skill in sorted(src.iterdir()):
        if not skill.is_dir():
            continue
        target = root / skill.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(skill, target)
        print(f"installed {skill.name} -> {target}")
    return 0


def _render_reduction(red: dict) -> str:
    out = [f"gen {red['gen']}: {red['candidates']} candidates, "
           f"{red['beat_parent']} beat their parent "
           f"(margin {red['margin']:g}), keeping {len(red['kept'])}"]
    for s in red["kept"]:
        out.append(f"KEEP\t{s['commit']}\t{s['score']:.6f}\t"
                   f"(parent {s['parent']}, delta {s['delta']:+.6f})")
    for f in red["cost_failures"]:
        out.append(f"COST_FAIL\t{f['commit']}\t{f['reason']}")
    if red["stall"]:
        out.append("STALL: frontier unchanged.")
    out.append("FRONTIER: " + " ".join(red["frontier"]))
    for f in red["fuse"]:
        out.append(f"FUSE_CANDIDATE\t{f['a']}\t{f['b']}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
