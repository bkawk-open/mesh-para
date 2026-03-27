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
import hashlib
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

from research import build_layout, load_state, parse_metrics, read_text, run_shell, write_text


DEFAULT_AUTORESEARCH_ROOT = "artifacts/autoresearch"
DEFAULT_MANAGER_ROOT = "artifacts/manager"
DEFAULT_MANAGER_NAME = "default"
DEFAULT_STRATEGY_DIR = "strategies"
DEFAULT_REMOTE_PROJECT_DIR = "/data/projects/mesh-para/cadresearch"
DEFAULT_DATASET_CACHE = "/data/projects/mesh-para/cadresearch/artifacts/abc_cache_512_boundary"
DEFAULT_AUDIT_CACHE = "/data/projects/mesh-para/cadresearch/artifacts/abc_cache_2048"
DEFAULT_REMOTE_AUDIT_ROOT = "/data/projects/mesh-para/cadresearch/artifacts/manager/default/audit_workspaces"
ACTIVE_RUN_PATTERN = re.compile(r"research\.py loop --run-name ([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class StrategyPreset:
    name: str
    description: str
    prompt_extra: str
    min_improvement: float = 0.001
    iterations: int = 16
    path: str = ""


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


@dataclass(frozen=True)
class RetroSignal:
    run_name: str
    strategy: str
    recommendation: str


@dataclass(frozen=True)
class AuditResult:
    run_name: str
    status: str
    score: float | None
    metrics: dict[str, float]
    log_path: str
    train_hash: str


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
        "audit_queue": root / "audit_queue.json",
        "logs": root / "logs",
        "retros": root / "retros",
        "audits": root / "audits",
        "workspaces": root / "workspaces",
    }


def ensure_manager_dirs(paths: dict[str, Path]) -> None:
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["retros"].mkdir(parents=True, exist_ok=True)
    paths["audits"].mkdir(parents=True, exist_ok=True)
    paths["workspaces"].mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(read_text(path).encode("utf-8")).hexdigest()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def load_strategies(project_dir: Path, strategy_dir: str) -> tuple[StrategyPreset, ...]:
    root = (project_dir / strategy_dir).resolve()
    if not root.exists():
        raise SystemExit(f"Strategy directory not found: {root}")
    strategies: list[StrategyPreset] = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(read_text(path))
        strategies.append(
            StrategyPreset(
                name=data["name"],
                description=data["description"],
                prompt_extra=data["prompt_extra"],
                min_improvement=float(data.get("min_improvement", 0.001)),
                iterations=int(data.get("iterations", 16)),
                path=str(path.relative_to(project_dir)),
            )
        )
    if not strategies:
        raise SystemExit(f"No strategy files found in {root}")
    return tuple(strategies)


def load_run_history(project_dir: Path, run_name: str, artifact_root: str) -> list[dict[str, Any]]:
    layout = build_layout(project_dir, run_name, artifact_root)
    if not layout.history_path.exists():
        return []
    return load_jsonl(layout.history_path)


def load_manager_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(read_text(path))


def load_audit_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(read_text(path))
    def valid(items: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("run_name") or not item.get("best_train_path") or not item.get("train_hash"):
                continue
            rows.append(item)
        return rows
    if isinstance(payload, list):
        return valid(payload)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return valid(payload["items"])
    return []


def save_audit_queue(path: Path, queue: list[dict[str, Any]]) -> None:
    save_json(path, {"items": queue})


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


def strategy_map(strategies: tuple[StrategyPreset, ...]) -> dict[str, StrategyPreset]:
    return {strategy.name: strategy for strategy in strategies}


def choose_strategy(
    strategies: tuple[StrategyPreset, ...],
    manager_history: list[dict[str, Any]],
    preferred: str | None = None,
) -> StrategyPreset:
    strategies_by_name = strategy_map(strategies)
    if preferred:
        if preferred not in strategies_by_name:
            raise SystemExit(f"Unknown strategy {preferred!r}. Available: {', '.join(strategies_by_name)}")
        return strategies_by_name[preferred]
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


def load_audit_result(audit_json_path: Path) -> AuditResult | None:
    if not audit_json_path.exists():
        return None
    payload = json.loads(read_text(audit_json_path))
    return AuditResult(
        run_name=str(payload.get("run_name", "")),
        status=str(payload.get("status", "")),
        score=float(payload["score"]) if payload.get("score") is not None else None,
        metrics=dict(payload.get("metrics", {})),
        log_path=str(payload.get("log_path", "")),
        train_hash=str(payload.get("train_hash", "")),
    )


def audit_paths_for(paths: dict[str, Path], run_name: str) -> dict[str, Path]:
    stem = sanitize_name(run_name)
    return {
        "json": paths["audits"] / f"{stem}.json",
        "sync_log": paths["audits"] / f"{stem}.sync.log",
        "train_log": paths["audits"] / f"{stem}.train.log",
        "workspace": paths["audits"] / f"{stem}_workspace",
    }


def cached_audit_result(paths: dict[str, Path], run_name: str, best_train_path: Path) -> AuditResult | None:
    audit_paths = audit_paths_for(paths, run_name)
    cached = load_audit_result(audit_paths["json"])
    if cached is None:
        return None
    if cached.train_hash != file_sha256(best_train_path):
        return None
    return cached


def queue_audit_target(
    queue: list[dict[str, Any]],
    run_name: str,
    best_train_path: Path,
    best_score: float,
) -> bool:
    train_hash = file_sha256(best_train_path)
    for item in queue:
        if item.get("train_hash") == train_hash:
            return False
    queue.append(
        {
            "run_name": run_name,
            "train_hash": train_hash,
            "best_train_path": str(best_train_path),
            "best_score": best_score,
            "queued_at": utc_now(),
        }
    )
    return True


def ensure_audit_result(
    project_dir: Path,
    paths: dict[str, Path],
    run_name: str,
    best_train_path: Path,
    remote_project_dir: str,
    remote_audit_root: str,
    audit_cache: str,
    audit_timeout: int,
) -> AuditResult:
    audit_paths = audit_paths_for(paths, run_name)
    train_hash = file_sha256(best_train_path)
    cached = load_audit_result(audit_paths["json"])
    if cached is not None and cached.train_hash == train_hash:
        return cached

    workspace = audit_paths["workspace"]
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_train_path, workspace / "train.py")

    remote_workspace = f"{remote_audit_root.rstrip('/')}/{sanitize_name(run_name)}"
    remote_prepare = f"{remote_project_dir.rstrip('/')}/prepare.py"
    prepare_cmd = (
        "ssh bkawk.local "
        f"\"rm -rf {shlex.quote(remote_workspace)} && "
        f"mkdir -p {shlex.quote(remote_workspace)} && "
        f"ln -s {shlex.quote(remote_prepare)} {shlex.quote(remote_workspace)}/prepare.py\""
    )
    prepare_result = run_shell(prepare_cmd, workspace, log_path=audit_paths["sync_log"], timeout_seconds=120)
    if prepare_result.returncode != 0:
        result = AuditResult(
            run_name=run_name,
            status="sync_error",
            score=None,
            metrics={},
            log_path=str(audit_paths["sync_log"].relative_to(project_dir)),
            train_hash=train_hash,
        )
        write_text(
            audit_paths["json"],
            json.dumps(
                {
                    "run_name": result.run_name,
                    "status": result.status,
                    "score": result.score,
                    "metrics": result.metrics,
                    "log_path": result.log_path,
                    "train_hash": result.train_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return result

    sync_cmd = f"scp train.py bkawk.local:{remote_workspace}/train.py"
    sync_result = run_shell(sync_cmd, workspace, log_path=audit_paths["sync_log"], timeout_seconds=120)
    if sync_result.returncode != 0:
        result = AuditResult(
            run_name=run_name,
            status="sync_error",
            score=None,
            metrics={},
            log_path=str(audit_paths["sync_log"].relative_to(project_dir)),
            train_hash=train_hash,
        )
        write_text(
            audit_paths["json"],
            json.dumps(
                {
                    "run_name": result.run_name,
                    "status": result.status,
                    "score": result.score,
                    "metrics": result.metrics,
                    "log_path": result.log_path,
                    "train_hash": result.train_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return result

    train_cmd = (
        "ssh bkawk.local "
        f"'cd {remote_workspace} && CADRESEARCH_CACHE_DIR={audit_cache} PYTHONUNBUFFERED=1 python3 train.py'"
    )
    train_result = run_shell(train_cmd, workspace, log_path=audit_paths["train_log"], timeout_seconds=audit_timeout)
    train_output = f"{train_result.stdout or ''}\n{train_result.stderr or ''}"
    metrics = parse_metrics(train_output)
    status = "ok" if train_result.returncode == 0 and "val_score" in metrics else "train_error"
    score = float(metrics["val_score"]) if "val_score" in metrics else None
    result = AuditResult(
        run_name=run_name,
        status=status,
        score=score,
        metrics=metrics,
        log_path=str(audit_paths["train_log"].relative_to(project_dir)),
        train_hash=train_hash,
    )
    write_text(
        audit_paths["json"],
        json.dumps(
            {
                "run_name": result.run_name,
                "status": result.status,
                "score": result.score,
                "metrics": result.metrics,
                "log_path": result.log_path,
                "train_hash": result.train_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return result


def audit_allows_promotion(
    current_audit: AuditResult | None,
    candidate_audit: AuditResult | None,
    max_regression: float,
) -> bool:
    if candidate_audit is None or candidate_audit.status != "ok" or candidate_audit.score is None:
        return False
    if current_audit is None or current_audit.status != "ok" or current_audit.score is None:
        return True
    return candidate_audit.score >= (current_audit.score - max_regression)


def process_next_audit(
    project_dir: Path,
    paths: dict[str, Path],
    artifact_root: str,
    remote_project_dir: str,
    remote_audit_root: str,
    audit_cache: str,
    audit_timeout: int,
) -> dict[str, Any] | None:
    queue = load_audit_queue(paths["audit_queue"])
    if not queue:
        return None

    remaining: list[dict[str, Any]] = []
    current = queue[0]
    for item in queue[1:]:
        remaining.append(item)

    run_name = str(current.get("run_name", ""))
    if not run_name:
        save_audit_queue(paths["audit_queue"], remaining)
        return {"run_name": "", "status": "skipped"}

    best_train_path = resolve_state_path(str(current.get("best_train_path", "")), project_dir)
    if not best_train_path.exists():
        save_audit_queue(paths["audit_queue"], remaining)
        return {"run_name": run_name, "status": "missing_train"}

    result = ensure_audit_result(
        project_dir,
        paths,
        run_name,
        best_train_path,
        remote_project_dir,
        remote_audit_root,
        audit_cache,
        audit_timeout,
    )
    save_audit_queue(paths["audit_queue"], remaining)
    return {
        "run_name": run_name,
        "status": result.status,
        "score": result.score,
        "log_path": result.log_path,
    }


def strategy_scorecards(
    project_dir: Path,
    strategies: tuple[StrategyPreset, ...],
    manager_history: list[dict[str, Any]],
    artifact_root: str,
) -> dict[str, StrategyStats]:
    cards = {strategy.name: StrategyStats(name=strategy.name) for strategy in strategies}
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


def recommendation_for_run(delta: float, keepers: int, near_misses: int, errors: int) -> str:
    if delta > 0.0 and keepers > 0:
        return "continue"
    if near_misses > 0 and errors == 0:
        return "probe_nearby"
    if errors >= 2:
        return "reliability"
    return "cool_down"


def load_retro_signals(retros_dir: Path) -> dict[str, RetroSignal]:
    if not retros_dir.exists():
        return {}
    signals: dict[str, RetroSignal] = {}
    strategy_re = re.compile(r"^- Strategy: `([^`]+)`$")
    recommendation_re = re.compile(r"^- Recommendation: `([^`]+)`$")
    for path in sorted(retros_dir.glob("*.md")):
        strategy = ""
        recommendation = ""
        for line in read_text(path).splitlines():
            if not strategy:
                match = strategy_re.match(line)
                if match:
                    strategy = match.group(1)
                    continue
            if not recommendation:
                match = recommendation_re.match(line)
                if match:
                    recommendation = match.group(1)
                    continue
            if strategy and recommendation:
                break
        if strategy and recommendation:
            signals[strategy] = RetroSignal(
                run_name=path.stem,
                strategy=strategy,
                recommendation=recommendation,
            )
    return signals


def choose_strategy_with_stats(
    strategies: tuple[StrategyPreset, ...],
    manager_history: list[dict[str, Any]],
    cards: dict[str, StrategyStats],
    retro_signals: dict[str, RetroSignal],
    preferred: str | None = None,
) -> StrategyPreset:
    strategies_by_name = strategy_map(strategies)
    if preferred:
        if preferred not in strategies_by_name:
            raise SystemExit(f"Unknown strategy {preferred!r}. Available: {', '.join(strategies_by_name)}")
        return strategies_by_name[preferred]

    recent = [
        row.get("strategy")
        for row in reversed(manager_history)
        if row.get("strategy") and row.get("status") == "launched"
    ]
    last_strategy = recent[0] if recent else None

    def retro_adjustment(strategy_name: str) -> float:
        signal = retro_signals.get(strategy_name)
        if signal is None:
            return 0.0
        if signal.recommendation == "continue":
            return 75.0
        if signal.recommendation == "probe_nearby":
            return 30.0
        if signal.recommendation == "reliability":
            return -60.0
        if signal.recommendation == "cool_down":
            return -120.0
        return 0.0

    ordered = sorted(
        strategies,
        key=lambda strategy: (
            cards[strategy.name].score + retro_adjustment(strategy.name),
            cards[strategy.name].avg_best_delta,
            -cards[strategy.name].launches,
        ),
        reverse=True,
    )
    if not ordered:
        return strategies[0]
    if last_strategy is None:
        return ordered[0]
    for strategy in ordered:
        signal = retro_signals.get(strategy.name)
        if strategy.name != last_strategy:
            return strategy
        if signal is not None and signal.recommendation in {"continue", "probe_nearby"}:
            return strategy
    return ordered[0]


def write_retrospective(
    project_dir: Path,
    retros_dir: Path,
    run_name: str,
    artifact_root: str,
    manager_row: dict[str, Any],
) -> Path | None:
    layout = build_layout(project_dir, run_name, artifact_root)
    state = load_state(layout)
    if state is None or not layout.history_path.exists():
        return None
    history = load_run_history(project_dir, run_name, artifact_root)
    if not history:
        return None

    seed = next((row for row in history if row.get("iteration") == 0), None)
    best_score = float(state["best_score"])
    seed_score = float(seed.get("val_score", best_score)) if seed else best_score
    delta = best_score - seed_score
    keepers = [row for row in history if row.get("status") == "keep"]
    near_misses = [row for row in history if row.get("status") == "near_miss"]
    errors = [row for row in history if str(row.get("status", "")).endswith("error")]
    latest = history[-1]
    recommendation = recommendation_for_run(delta, len(keepers), len(near_misses), len(errors))
    retros_dir.mkdir(parents=True, exist_ok=True)
    path = retros_dir / f"{sanitize_name(run_name)}.md"
    lines = [
        f"# Retrospective: {run_name}",
        "",
        f"- Strategy: `{manager_row.get('strategy', '')}`",
        f"- Source run: `{manager_row.get('resolved_source_run', manager_row.get('source_run', ''))}`",
        f"- Started from score: `{seed_score:.6f}`",
        f"- Best score: `{best_score:.6f}`",
        f"- Net delta: `{delta:.6f}`",
        f"- Iterations completed: `{state.get('iterations_completed', 0)}`",
        f"- Keepers: `{len(keepers)}`",
        f"- Near misses: `{len(near_misses)}`",
        f"- Errors: `{len(errors)}`",
        f"- Latest status: `{latest.get('status', '')}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Notes",
    ]
    if keepers:
        best_keeper = keepers[-1]
        lines.append(
            f"- Best keeper was iteration `{best_keeper.get('iteration')}` with score `{float(best_keeper.get('val_score', 0.0)):.6f}`."
        )
    else:
        lines.append("- No keeper in this run.")
    if near_misses:
        lines.append(
            "- Near-miss iterations: "
            + ", ".join(
                f"`{row.get('iteration')}` ({float(row.get('val_score', 0.0)):.6f})"
                for row in near_misses[:5]
            )
        )
    if errors:
        lines.append(
            "- Error iterations: "
            + ", ".join(f"`{row.get('iteration')}` ({row.get('status')})" for row in errors[:5])
        )
    write_text(path, "\n".join(lines) + "\n")
    return path


def write_missing_retrospectives(
    project_dir: Path,
    paths: dict[str, Path],
    manager_history: list[dict[str, Any]],
    artifact_root: str,
) -> None:
    for row in manager_history:
        if row.get("status") != "launched":
            continue
        run_name = row.get("run_name")
        if not run_name:
            continue
        write_retrospective(project_dir, paths["retros"], run_name, artifact_root, row)


def resolve_source_candidate(
    project_dir: Path,
    paths: dict[str, Path],
    artifact_root: str,
    manager_history: list[dict[str, Any]],
    explicit_source_run: str,
    remote_project_dir: str,
    remote_audit_root: str,
    audit_cache: str,
    audit_timeout: int,
    audit_max_regression: float,
    refresh_audits: bool,
    audit_queue: list[dict[str, Any]] | None = None,
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
    if refresh_audits:
        candidate["audit"] = ensure_audit_result(
            project_dir,
            paths,
            candidate["run_name"],
            candidate["best_train_path"],
            remote_project_dir,
            remote_audit_root,
            audit_cache,
            audit_timeout,
        )
    else:
        candidate["audit"] = cached_audit_result(paths, candidate["run_name"], candidate["best_train_path"])
        if candidate["audit"] is None and audit_queue is not None:
            queue_audit_target(audit_queue, candidate["run_name"], candidate["best_train_path"], candidate["best_score"])
    for managed in completed_manager_runs(project_dir, manager_history, artifact_root):
        if managed["best_score"] <= candidate["best_score"]:
            continue
        if refresh_audits:
            managed["audit"] = ensure_audit_result(
                project_dir,
                paths,
                managed["run_name"],
                managed["best_train_path"],
                remote_project_dir,
                remote_audit_root,
                audit_cache,
                audit_timeout,
            )
        else:
            managed["audit"] = cached_audit_result(paths, managed["run_name"], managed["best_train_path"])
            if managed["audit"] is None and audit_queue is not None:
                queue_audit_target(audit_queue, managed["run_name"], managed["best_train_path"], managed["best_score"])
        if audit_allows_promotion(candidate.get("audit"), managed["audit"], audit_max_regression):
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
    ensure_manager_dirs(paths)
    strategies = load_strategies(project_dir, args.strategy_dir)
    active = sorted(active_run_names())
    refresh_audits = args.allow_live_audits or not active
    audit_queue = load_audit_queue(paths["audit_queue"])
    queue_before = len(audit_queue)
    manager_history = load_jsonl(paths["history"])
    write_missing_retrospectives(project_dir, paths, manager_history, args.artifact_root)
    retro_signals = load_retro_signals(paths["retros"])
    cards = strategy_scorecards(project_dir, strategies, manager_history, args.artifact_root)
    resolved = resolve_source_candidate(
        project_dir,
        paths,
        args.artifact_root,
        manager_history,
        args.source_run,
        args.remote_project_dir,
        args.remote_audit_root,
        args.audit_cache,
        args.audit_timeout,
        args.audit_max_regression,
        refresh_audits,
        audit_queue if not refresh_audits else None,
    )
    if not refresh_audits:
        save_audit_queue(paths["audit_queue"], audit_queue)
    history = load_run_history(project_dir, resolved["run_name"], args.artifact_root)
    analysis = analyze_run(history, args.tail_window)
    strategy = choose_strategy_with_stats(strategies, manager_history, cards, retro_signals, args.strategy)

    print(f"source_run:      {args.source_run}")
    print(f"audit_mode:      {'refresh' if refresh_audits else 'cached_only'}")
    if active:
        print(f"active_runs:     {','.join(active)}")
    if not refresh_audits:
        print(f"audit_queue:     {len(audit_queue)} (added {len(audit_queue) - queue_before})")
    print(f"resolved_run:    {resolved['run_name']}")
    print(f"resolved_score:  {resolved['best_score']:.6f}")
    audit = resolved.get("audit")
    if audit is not None:
        print(f"resolved_audit:  {audit.status} {format(audit.score, '.6f') if audit.score is not None else 'NA'}")
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
    for name in [s.name for s in strategies]:
        card = cards[name]
        signal = retro_signals.get(name)
        recommendation = signal.recommendation if signal is not None else "none"
        print(
            f"strategy_{name}: launches={card.launches} improvements={card.improvements} "
            f"near_misses={card.near_misses} errors={card.errors} avg_delta={card.avg_best_delta:.6f} "
            f"score={card.score:.2f} recommendation={recommendation}"
        )


def do_launch_next(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    paths = manager_paths(project_dir, args.manager_root, args.manager_name)
    ensure_manager_dirs(paths)
    strategies = load_strategies(project_dir, args.strategy_dir)

    active = active_run_names()
    if args.require_idle and active:
        raise SystemExit(f"Active research runs detected: {', '.join(sorted(active))}")

    manager_history = load_jsonl(paths["history"])
    write_missing_retrospectives(project_dir, paths, manager_history, args.artifact_root)
    retro_signals = load_retro_signals(paths["retros"])
    cards = strategy_scorecards(project_dir, strategies, manager_history, args.artifact_root)
    resolved = resolve_source_candidate(
        project_dir,
        paths,
        args.artifact_root,
        manager_history,
        args.source_run,
        args.remote_project_dir,
        args.remote_audit_root,
        args.audit_cache,
        args.audit_timeout,
        args.audit_max_regression,
        True,
    )
    save_audit_queue(paths["audit_queue"], [])
    source_history = load_run_history(project_dir, resolved["run_name"], args.artifact_root)
    analysis = analyze_run(source_history, args.tail_window)
    strategy = choose_strategy_with_stats(strategies, manager_history, cards, retro_signals, args.strategy)

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
        "source_audit_score": resolved["audit"].score if resolved.get("audit") is not None else None,
        "source_audit_status": resolved["audit"].status if resolved.get("audit") is not None else "",
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
            audit_result = process_next_audit(
                project_dir,
                paths,
                args.artifact_root,
                args.remote_project_dir,
                args.remote_audit_root,
                args.audit_cache,
                args.audit_timeout,
            )
            if audit_result is not None:
                message = (
                    f"[{timestamp}] idle=true audit_run={audit_result.get('run_name','')} "
                    f"status={audit_result.get('status','')} score={audit_result.get('score','NA')}"
                )
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
    parser.add_argument("--strategy-dir", default=DEFAULT_STRATEGY_DIR, help="Directory containing strategy-pack JSON files.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-run", required=True, help="Existing run to analyze and promote from.")
    common.add_argument("--tail-window", type=int, default=6, help="How many recent iterations define a plateau.")
    common.add_argument("--strategy", default=None, help="Optional explicit strategy preset.")
    common.add_argument("--remote-project-dir", default=DEFAULT_REMOTE_PROJECT_DIR, help="Remote project path on bkawk.local.")
    common.add_argument("--remote-audit-root", default=DEFAULT_REMOTE_AUDIT_ROOT, help="Remote workspace root used for isolated audit reruns.")
    common.add_argument("--audit-cache", default=DEFAULT_AUDIT_CACHE, help="Remote cache dir used for larger audit reruns.")
    common.add_argument("--audit-timeout", type=int, default=1200, help="Timeout in seconds for audit reruns.")
    common.add_argument("--audit-max-regression", type=float, default=0.010, help="Maximum allowed audit-score drop when promoting a better main benchmark result.")

    recommend = subparsers.add_parser("recommend", parents=[common], help="Suggest the next strategy for a source run.")
    recommend.add_argument("--allow-live-audits", action="store_true", help="Allow recommend to refresh missing audits even when research runs are active.")
    recommend.set_defaults(func=do_recommend)

    launch = subparsers.add_parser("launch-next", parents=[common], help="Promote the best baseline and launch the next run.")
    launch.add_argument("--run-name", default=None, help="Optional explicit run name.")
    launch.add_argument("--agent", choices=["codex", "claude", "none"], default="codex", help="Editing agent to invoke.")
    launch.add_argument("--agent-model", default=None, help="Optional model override for the agent CLI.")
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
