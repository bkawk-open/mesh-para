"""
Lightweight lab-manager for mesh-para autonomous runs.

This sits above `research.py` and answers the questions the lower loop
cannot answer by itself:
- has a search neighborhood plateaued?
- which strategy family should run next?
- how do we seed a fresh run from the current best baseline?
- how do we launch the next run without manual babysitting?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research import build_layout, load_state, read_text, write_text


DEFAULT_AUTORESEARCH_ROOT = "artifacts/autoresearch"
DEFAULT_MANAGER_ROOT = "artifacts/manager"
DEFAULT_MANAGER_NAME = "default"
DEFAULT_REMOTE_PROJECT_DIR = "/data/projects/mesh-para/cadresearch"
DEFAULT_DATASET_CACHE = "/data/projects/mesh-para/cadresearch/artifacts/abc_cache_512_boundary"
ACTIVE_RUN_PATTERN = re.compile(r"research\.py loop --run-name ([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class StrategyPreset:
    name: str
    description: str
    prompt_extra: str
    min_improvement: float = 0.001
    iterations: int = 16


@dataclass
class StrategyStats:
    name: str
    launches: int = 0
    improvements: int = 0
    near_misses: int = 0
    confirm_reverts: int = 0
    errors: int = 0
    avg_best_delta: float = 0.0
    last_run_name: str = ""
    last_timestamp: str = ""
    score: float = 0.0


STRATEGIES: tuple[StrategyPreset, ...] = (
    StrategyPreset(
        name="boundary_calibration",
        description="Refine how boundary supervision influences the shared features without adding heavy compute.",
        prompt_extra=(
            "Build on the current best boundary-supervision baseline. Make exactly one focused change in train.py. "
            "Target only boundary-head usage, boundary-conditioned feature fusion, or boundary loss calibration. "
            "Keep the local neighborhood block unchanged. Avoid deeper local stacks, larger k, second neighborhood "
            "passes, or broad optimizer rewrites. Protect throughput first."
        ),
    ),
    StrategyPreset(
        name="global_context",
        description="Search only around low-cost global context mixing and point/global fusion.",
        prompt_extra=(
            "Build on the current best boundary-supervision baseline. Make exactly one focused change in train.py. "
            "Target only global-context aggregation or how global features are fused back into per-point features. "
            "Keep the geometry-aware local edge block unchanged. Avoid deeper local stacks, larger k, or extra "
            "neighborhood passes. Protect throughput first."
        ),
    ),
    StrategyPreset(
        name="class_fusion",
        description="Try small class-aware feature routing without large per-class heads.",
        prompt_extra=(
            "Build on the current best boundary-supervision baseline. Make exactly one focused change in train.py. "
            "Target only lightweight class-aware or task-aware feature fusion for classification and parameter "
            "prediction. Keep the local block and neighborhood count unchanged. Avoid heavy separate heads, deeper "
            "stacks, or expensive new branches. Protect throughput first."
        ),
    ),
    StrategyPreset(
        name="throughput_trim",
        description="Recover throughput with tiny simplifications that may improve effective optimization in 5 minutes.",
        prompt_extra=(
            "Build on the current best boundary-supervision baseline. Make exactly one focused change in train.py. "
            "Target only tiny simplifications or fusion changes that could increase effective optimization progress "
            "within the fixed 5-minute budget. Do not make the local neighborhood path heavier. Avoid larger k, "
            "second neighborhood passes, or deeper model stacks."
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", value.lower()).strip("_")


def manager_paths(project_dir: Path, manager_root: str, manager_name: str) -> dict[str, Path]:
    root = project_dir / manager_root / manager_name
    return {
        "root": root,
        "history": root / "history.jsonl",
        "state": root / "state.json",
        "logs": root / "logs",
        "workspaces": root / "workspaces",
    }


def ensure_manager_dirs(paths: dict[str, Path]) -> None:
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["workspaces"].mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def load_run_history(project_dir: Path, run_name: str, artifact_root: str) -> list[dict[str, Any]]:
    layout = build_layout(project_dir, run_name, artifact_root)
    if not layout.history_path.exists():
        return []
    return load_jsonl(layout.history_path)


def load_manager_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(read_text(path))


def active_run_names() -> set[str]:
    result = subprocess.run(
        ["ps", "-ax", "-o", "command"],
        text=True,
        capture_output=True,
        check=True,
    )
    names: set[str] = set()
    for line in result.stdout.splitlines():
        match = ACTIVE_RUN_PATTERN.search(line)
        if match:
            names.add(match.group(1))
    return names


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)


def run_layout_for(project_dir: Path, artifact_root: str, run_name: str):
    return build_layout(project_dir, run_name, artifact_root)


def run_last_progress_timestamp(project_dir: Path, artifact_root: str, run_name: str) -> float | None:
    layout = run_layout_for(project_dir, artifact_root, run_name)
    candidates = []
    for path in (layout.history_path, layout.state_path):
        if path.exists():
            candidates.append(path.stat().st_mtime)
    return max(candidates) if candidates else None


def stop_run_process(run_name: str) -> None:
    session_name = sanitize_name(run_name)
    subprocess.run(["screen", "-S", session_name, "-X", "quit"], text=True, capture_output=True)
    subprocess.run(
        ["pkill", "-f", f"research.py loop --run-name {run_name}"],
        text=True,
        capture_output=True,
    )


def detect_stale_runs(project_dir: Path, artifact_root: str, active_runs: list[str], stale_seconds: int) -> list[dict[str, Any]]:
    now = time.time()
    stale: list[dict[str, Any]] = []
    for run_name in active_runs:
        last_progress = run_last_progress_timestamp(project_dir, artifact_root, run_name)
        if last_progress is None:
            continue
        age = now - last_progress
        if age >= stale_seconds:
            stale.append({"run_name": run_name, "age_seconds": age})
    return stale


def analyze_run(history: list[dict[str, Any]], tail_window: int) -> dict[str, Any]:
    nonseed = [row for row in history if row.get("iteration", 0) > 0]
    tail = nonseed[-tail_window:]
    keepers = [row for row in nonseed if row.get("status") == "keep"]
    reverts = [row for row in tail if row.get("status") == "revert"]
    best_tail = max((float(row.get("val_score", float("-inf"))) for row in tail), default=float("-inf"))
    deltas = [float(row.get("val_score", 0.0)) - float(history[0].get("val_score", 0.0)) for row in tail] if history else []
    return {
        "completed": len(nonseed),
        "keepers": len(keepers),
        "tail_window": tail_window,
        "tail_count": len(tail),
        "tail_reverts": len(reverts),
        "tail_keepers": sum(1 for row in tail if row.get("status") == "keep"),
        "tail_best_score": None if best_tail == float("-inf") else best_tail,
        "tail_best_delta_from_seed": None if not deltas else max(deltas),
        "plateaued": len(tail) >= tail_window and sum(1 for row in tail if row.get("status") == "keep") == 0,
        "latest": tail[-1] if tail else (nonseed[-1] if nonseed else None),
    }


def strategy_map() -> dict[str, StrategyPreset]:
    return {strategy.name: strategy for strategy in STRATEGIES}


def choose_strategy(manager_history: list[dict[str, Any]], preferred: str | None = None) -> StrategyPreset:
    strategies = strategy_map()
    if preferred:
        if preferred not in strategies:
            raise SystemExit(f"Unknown strategy {preferred!r}. Available: {', '.join(strategies)}")
        return strategies[preferred]
    raise RuntimeError("choose_strategy requires scorecards; use choose_strategy_with_stats")


def build_run_name(source_run: str, strategy: StrategyPreset, stamp: str | None = None) -> str:
    stamp = stamp or datetime.now().strftime("%m%d_%H%M")
    return sanitize_name(f"{source_run}_{strategy.name}_{stamp}")


def resolve_state_path(path_value: str, *bases: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def best_paths_for_run(
    project_dir: Path,
    run_name: str,
    artifact_root: str,
    workspace_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    layout = build_layout(project_dir, run_name, artifact_root)
    state = load_state(layout)
    if state is None:
        raise SystemExit(f"No state found for run {run_name!r} in {layout.run_dir}")
    bases = [workspace_dir] if workspace_dir is not None else []
    bases.append(project_dir)
    best_train = resolve_state_path(state["best_train_path"], *bases)
    best_log = resolve_state_path(state["best_log"], *bases)
    return best_train, best_log, state


def completed_manager_runs(project_dir: Path, manager_history: list[dict[str, Any]], artifact_root: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    completed: list[dict[str, Any]] = []
    for row in manager_history:
        run_name = row.get("run_name")
        if not run_name or run_name in seen or row.get("status") != "launched":
            continue
        seen.add(run_name)
        workspace_dir = None
        if row.get("workspace_dir"):
            workspace_dir = (project_dir / str(row["workspace_dir"])).resolve()
        try:
            best_train, best_log, state = best_paths_for_run(project_dir, run_name, artifact_root, workspace_dir=workspace_dir)
        except SystemExit:
            continue
        completed.append(
            {
                "run_name": run_name,
                "best_score": float(state["best_score"]),
                "best_log": best_log,
                "best_train_path": best_train,
                "state": state,
                "strategy": row.get("strategy", ""),
                "timestamp": row.get("timestamp", ""),
            }
        )
    return completed


def strategy_scorecards(project_dir: Path, manager_history: list[dict[str, Any]], artifact_root: str) -> dict[str, StrategyStats]:
    cards = {strategy.name: StrategyStats(name=strategy.name) for strategy in STRATEGIES}
    for row in manager_history:
        if row.get("status") != "launched":
            continue
        strategy_name = row.get("strategy")
        run_name = row.get("run_name")
        if not strategy_name or strategy_name not in cards or not run_name:
            continue
        card = cards[strategy_name]
        card.launches += 1
        card.last_run_name = run_name
        card.last_timestamp = row.get("timestamp", "")
        source_best = float(row.get("source_best_score", 0.0))

        try:
            layout = build_layout(project_dir, run_name, artifact_root)
            state = load_state(layout)
            history = load_run_history(project_dir, run_name, artifact_root)
        except SystemExit:
            continue
        if state is None:
            continue

        best_delta = float(state["best_score"]) - source_best
        prev_count = max(card.launches - 1, 0)
        card.avg_best_delta = ((card.avg_best_delta * prev_count) + best_delta) / card.launches
        if best_delta > 1e-12:
            card.improvements += 1

        for item in history:
            status = item.get("status")
            if status == "near_miss":
                card.near_misses += 1
            elif status == "confirm_revert":
                card.confirm_reverts += 1
            elif status in {"train_error", "sync_error", "compile_error", "agent_error", "confirm_error"}:
                card.errors += 1

    for card in cards.values():
        if card.launches == 0:
            card.score = 1000.0
            continue
        card.score = (
            (200.0 * card.improvements)
            + (25.0 * card.near_misses)
            + (1000.0 * card.avg_best_delta)
            - (15.0 * card.confirm_reverts)
            - (10.0 * card.errors)
        )
    return cards


def choose_strategy_with_stats(
    manager_history: list[dict[str, Any]],
    cards: dict[str, StrategyStats],
    preferred: str | None = None,
) -> StrategyPreset:
    strategies = strategy_map()
    if preferred:
        if preferred not in strategies:
            raise SystemExit(f"Unknown strategy {preferred!r}. Available: {', '.join(strategies)}")
        return strategies[preferred]

    recent = [
        row.get("strategy")
        for row in reversed(manager_history)
        if row.get("strategy") and row.get("status") == "launched"
    ]
    last_strategy = recent[0] if recent else None

    ordered = sorted(
        STRATEGIES,
        key=lambda strategy: (
            cards[strategy.name].score,
            cards[strategy.name].avg_best_delta,
            -cards[strategy.name].launches,
        ),
        reverse=True,
    )
    if not ordered:
        return STRATEGIES[0]
    if last_strategy is None:
        return ordered[0]
    for strategy in ordered:
        if strategy.name != last_strategy:
            return strategy
    return ordered[0]


def resolve_source_candidate(
    project_dir: Path,
    artifact_root: str,
    manager_history: list[dict[str, Any]],
    explicit_source_run: str,
) -> dict[str, Any]:
    best_train, best_log, source_state = best_paths_for_run(project_dir, explicit_source_run, artifact_root)
    candidate = {
        "run_name": explicit_source_run,
        "best_score": float(source_state["best_score"]),
        "best_log": best_log,
        "best_train_path": best_train,
        "state": source_state,
        "strategy": "source",
        "timestamp": source_state.get("updated_at", ""),
    }
    for managed in completed_manager_runs(project_dir, manager_history, artifact_root):
        if managed["best_score"] > candidate["best_score"]:
            candidate = managed
    return candidate


def launch_screen(
    workspace_dir: Path,
    run_name: str,
    command: str,
    log_path: Path,
) -> None:
    session_name = sanitize_name(run_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "screen",
            "-dmS",
            session_name,
            "/bin/zsh",
            "-lc",
            f"cd {shlex.quote(str(workspace_dir))} && {command} >> {shlex.quote(str(log_path))} 2>&1",
        ],
        check=True,
        text=True,
    )


def build_research_command(
    run_name: str,
    strategy: StrategyPreset,
    baseline_log: Path,
    artifact_root: Path,
    remote_project_dir: str,
    dataset_cache: str,
    agent: str,
    agent_model: str | None,
) -> str:
    parts = [
        "python3",
        "research.py",
        "loop",
        "--run-name",
        run_name,
        "--iterations",
        str(strategy.iterations),
        "--agent",
        agent,
        "--agent-prompt-extra",
        strategy.prompt_extra,
        "--pre-train-command",
        f"scp train.py bkawk.local:{remote_project_dir}/train.py",
        "--train-command",
        (
            "ssh bkawk.local "
            f"'cd {remote_project_dir} && CADRESEARCH_CACHE_DIR={dataset_cache} PYTHONUNBUFFERED=1 python3 train.py'"
        ),
        "--baseline-log",
        str(baseline_log),
        "--artifact-root",
        str(artifact_root),
        "--min-improvement",
        f"{strategy.min_improvement:.3f}",
    ]
    if agent_model:
        parts.extend(["--agent-model", agent_model])
    return shlex.join(parts)


def create_workspace(
    source_project_dir: Path,
    workspaces_root: Path,
    run_name: str,
    best_train_path: Path,
) -> Path:
    workspace_dir = workspaces_root / sanitize_name(run_name)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    def ignore(dirname: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in {".git", "artifacts", ".venv", "__pycache__"}:
                ignored.add(name)
            elif name.endswith(".pyc"):
                ignored.add(name)
            elif name.startswith("._"):
                ignored.add(name)
        return ignored

    shutil.copytree(source_project_dir, workspace_dir, ignore=ignore)
    shutil.copy2(best_train_path, workspace_dir / "train.py")
    return workspace_dir


def do_recommend(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    paths = manager_paths(project_dir, args.manager_root, args.manager_name)
    manager_history = load_jsonl(paths["history"])
    cards = strategy_scorecards(project_dir, manager_history, args.artifact_root)
    resolved = resolve_source_candidate(project_dir, args.artifact_root, manager_history, args.source_run)
    history = load_run_history(project_dir, resolved["run_name"], args.artifact_root)
    analysis = analyze_run(history, args.tail_window)
    strategy = choose_strategy_with_stats(manager_history, cards, args.strategy)

    print(f"source_run:      {args.source_run}")
    print(f"resolved_run:    {resolved['run_name']}")
    print(f"resolved_score:  {resolved['best_score']:.6f}")
    print(f"completed:       {analysis['completed']}")
    print(f"plateaued:       {analysis['plateaued']}")
    print(f"tail_reverts:    {analysis['tail_reverts']}/{analysis['tail_count']}")
    print(f"tail_keepers:    {analysis['tail_keepers']}/{analysis['tail_count']}")
    if analysis["tail_best_score"] is not None:
        print(f"tail_best_score: {analysis['tail_best_score']:.6f}")
    if analysis["latest"] is not None:
        latest = analysis["latest"]
        print(f"latest_status:   {latest.get('status')}")
        if "val_score" in latest:
            print(f"latest_score:    {float(latest['val_score']):.6f}")
    print(f"next_strategy:   {strategy.name}")
    print(f"description:     {strategy.description}")
    for name in [s.name for s in STRATEGIES]:
        card = cards[name]
        print(
            f"strategy_{name}: launches={card.launches} improvements={card.improvements} "
            f"near_misses={card.near_misses} errors={card.errors} avg_delta={card.avg_best_delta:.6f} score={card.score:.2f}"
        )


def do_launch_next(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    paths = manager_paths(project_dir, args.manager_root, args.manager_name)
    ensure_manager_dirs(paths)

    active = active_run_names()
    if args.require_idle and active:
        raise SystemExit(f"Active research runs detected: {', '.join(sorted(active))}")

    manager_history = load_jsonl(paths["history"])
    cards = strategy_scorecards(project_dir, manager_history, args.artifact_root)
    resolved = resolve_source_candidate(project_dir, args.artifact_root, manager_history, args.source_run)
    source_history = load_run_history(project_dir, resolved["run_name"], args.artifact_root)
    analysis = analyze_run(source_history, args.tail_window)
    strategy = choose_strategy_with_stats(manager_history, cards, args.strategy)

    best_train = resolved["best_train_path"]
    best_log = resolved["best_log"]
    source_state = resolved["state"]
    run_name = args.run_name or build_run_name(resolved["run_name"], strategy)
    log_path = project_dir / args.artifact_root / run_name / "overnight_loop.log"
    workspace_dir = create_workspace(project_dir, paths["workspaces"], run_name, best_train)

    command = build_research_command(
        run_name=run_name,
        strategy=strategy,
        baseline_log=best_log,
        artifact_root=(project_dir / args.artifact_root).resolve(),
        remote_project_dir=args.remote_project_dir,
        dataset_cache=args.dataset_cache,
        agent=args.agent,
        agent_model=args.agent_model,
    )

    record = {
        "timestamp": utc_now(),
        "manager_name": args.manager_name,
        "source_run": args.source_run,
        "resolved_source_run": resolved["run_name"],
        "source_best_score": float(source_state["best_score"]),
        "source_best_log": str(best_log.relative_to(project_dir)),
        "source_best_train": str(best_train.relative_to(project_dir)),
        "source_plateaued": analysis["plateaued"],
        "strategy": strategy.name,
        "strategy_description": strategy.description,
        "strategy_score": cards[strategy.name].score,
        "run_name": run_name,
        "workspace_dir": str(workspace_dir.relative_to(project_dir)),
        "command": command,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        record["status"] = "planned"
        append_jsonl(paths["history"], record)
        save_json(paths["state"], record)
        print(f"planned_run:     {run_name}")
        print(f"strategy:        {strategy.name}")
        print(f"baseline_log:    {best_log}")
        print(f"plateaued:       {analysis['plateaued']}")
        print(f"command:         {command}")
        return

    launch_screen(workspace_dir, run_name, command, log_path)
    record["status"] = "launched"
    record["screen_session"] = sanitize_name(run_name)
    record["log_path"] = str(log_path.relative_to(project_dir))
    append_jsonl(paths["history"], record)
    save_json(paths["state"], record)

    print(f"launched_run:    {run_name}")
    print(f"strategy:        {strategy.name}")
    print(f"screen_session:  {sanitize_name(run_name)}")
    print(f"log_path:        {log_path}")


def do_supervise(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    paths = manager_paths(project_dir, args.manager_root, args.manager_name)
    ensure_manager_dirs(paths)
    log_path = paths["logs"] / f"{sanitize_name(args.supervisor_name)}.log"

    iteration = 0
    while True:
        iteration += 1
        timestamp = utc_now()
        active = sorted(active_run_names())
        stale_runs = detect_stale_runs(project_dir, args.artifact_root, active, args.stale_seconds) if active else []
        if stale_runs:
            for stale in stale_runs:
                run_name = stale["run_name"]
                age = stale["age_seconds"]
                log_line(log_path, f"[{timestamp}] stale_run run={run_name} age_seconds={age:.0f} action=stop")
                stop_run_process(run_name)
            active = sorted(active_run_names())

        if active:
            message = f"[{timestamp}] idle=false active_runs={','.join(active)}"
            log_line(log_path, message)
        else:
            message = f"[{timestamp}] idle=true launching_next"
            log_line(log_path, message)
            launch_args = argparse.Namespace(**vars(args))
            launch_args.dry_run = False
            launch_args.require_idle = False
            launch_args.run_name = None
            do_launch_next(launch_args)
            if args.once:
                return

        if args.once:
            return
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-manager for autonomous mesh-para runs.")
    parser.add_argument("--project-dir", default=".", help="Path to the mesh-para project.")
    parser.add_argument("--artifact-root", default=DEFAULT_AUTORESEARCH_ROOT, help="Autoresearch artifact root.")
    parser.add_argument("--manager-root", default=DEFAULT_MANAGER_ROOT, help="Manager artifact root.")
    parser.add_argument("--manager-name", default=DEFAULT_MANAGER_NAME, help="Manager history namespace.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-run", required=True, help="Existing run to analyze and promote from.")
    common.add_argument("--tail-window", type=int, default=6, help="How many recent iterations define a plateau.")
    common.add_argument("--strategy", default=None, help="Optional explicit strategy preset.")

    recommend = subparsers.add_parser("recommend", parents=[common], help="Suggest the next strategy for a source run.")
    recommend.set_defaults(func=do_recommend)

    launch = subparsers.add_parser("launch-next", parents=[common], help="Promote the best baseline and launch the next run.")
    launch.add_argument("--run-name", default=None, help="Optional explicit run name.")
    launch.add_argument("--agent", choices=["codex", "claude", "none"], default="codex", help="Editing agent to invoke.")
    launch.add_argument("--agent-model", default=None, help="Optional model override for the agent CLI.")
    launch.add_argument("--remote-project-dir", default=DEFAULT_REMOTE_PROJECT_DIR, help="Remote project path on bkawk.local.")
    launch.add_argument("--dataset-cache", default=DEFAULT_DATASET_CACHE, help="Remote cache dir used by train.py.")
    launch.add_argument("--dry-run", action="store_true", help="Plan the next run without launching it.")
    launch.add_argument(
        "--require-idle",
        action="store_true",
        help="Refuse to launch when any research.py loop process is already running.",
    )
    launch.set_defaults(func=do_launch_next)

    supervise = subparsers.add_parser("supervise", parents=[common], help="Keep launching the next run whenever no research run is active.")
    supervise.add_argument("--poll-seconds", type=int, default=600, help="How often to check for an idle lab state.")
    supervise.add_argument("--supervisor-name", default="autonomy", help="Name used for supervisor log files.")
    supervise.add_argument("--agent", choices=["codex", "claude", "none"], default="codex", help="Editing agent to invoke.")
    supervise.add_argument("--agent-model", default=None, help="Optional model override for the agent CLI.")
    supervise.add_argument("--remote-project-dir", default=DEFAULT_REMOTE_PROJECT_DIR, help="Remote project path on bkawk.local.")
    supervise.add_argument("--dataset-cache", default=DEFAULT_DATASET_CACHE, help="Remote cache dir used by train.py.")
    supervise.add_argument("--stale-seconds", type=int, default=1800, help="Kill and replace active runs that show no progress for this many seconds.")
    supervise.add_argument("--once", action="store_true", help="Run a single supervisor check cycle and exit.")
    supervise.set_defaults(func=do_supervise)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
