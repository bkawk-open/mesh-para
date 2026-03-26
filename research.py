"""
Autonomous experiment harness for cadresearch.

This script mirrors the operational loop of autoresearch:
- keep a single editable file (`train.py`)
- run a fixed-time benchmark
- keep only changes that improve `val_score`
- record every attempt
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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"^([a-z_]+):\s+([-+]?[0-9]*\.?[0-9]+)$", re.MULTILINE)
DEFAULT_RUN_NAME = "default"
DEFAULT_ARTIFACT_ROOT = "artifacts/autoresearch"


@dataclass
class RunLayout:
    root: Path
    run_dir: Path
    logs_dir: Path
    candidates_dir: Path
    best_dir: Path
    history_path: Path
    leaderboard_path: Path
    state_path: Path
    best_train_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in METRIC_RE.findall(text):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def format_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def build_layout(project_dir: Path, run_name: str, artifact_root: str) -> RunLayout:
    root = project_dir / artifact_root
    run_dir = root / run_name
    logs_dir = run_dir / "logs"
    candidates_dir = run_dir / "candidates"
    best_dir = run_dir / "best"
    return RunLayout(
        root=root,
        run_dir=run_dir,
        logs_dir=logs_dir,
        candidates_dir=candidates_dir,
        best_dir=best_dir,
        history_path=run_dir / "history.jsonl",
        leaderboard_path=run_dir / "leaderboard.tsv",
        state_path=run_dir / "state.json",
        best_train_path=best_dir / "train.py",
    )


def ensure_layout(layout: RunLayout) -> None:
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    layout.candidates_dir.mkdir(parents=True, exist_ok=True)
    layout.best_dir.mkdir(parents=True, exist_ok=True)
    if not layout.leaderboard_path.exists():
        write_text(
            layout.leaderboard_path,
            "iteration\tstatus\tval_score\tmacro_iou\tparam_rmse_norm\tparam_score\ttrain_log\tagent_log\tnotes\n",
        )


def load_state(layout: RunLayout) -> dict[str, Any] | None:
    if not layout.state_path.exists():
        return None
    return json.loads(read_text(layout.state_path))


def save_state(layout: RunLayout, state: dict[str, Any]) -> None:
    write_text(layout.state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def append_history(layout: RunLayout, record: dict[str, Any]) -> None:
    with layout.history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def append_leaderboard(layout: RunLayout, record: dict[str, Any]) -> None:
    row = [
        str(record.get("iteration", "")),
        str(record.get("status", "")),
        format_float(record.get("val_score")),
        format_float(record.get("macro_iou")),
        format_float(record.get("param_rmse_norm")),
        format_float(record.get("param_score")),
        str(record.get("train_log", "")),
        str(record.get("agent_log", "")),
        str(record.get("notes", "")).replace("\t", " "),
    ]
    with layout.leaderboard_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def py_compile_file(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)


def run_shell(command: str, cwd: Path, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    if log_path is not None:
        combined = f"$ {command}\n"
        if result.stdout:
            combined += result.stdout
        if result.stderr:
            combined += ("\n" if result.stdout else "") + result.stderr
        write_text(log_path, combined)
    return result


def recent_summary(layout: RunLayout, limit: int = 5) -> str:
    if not layout.history_path.exists():
        return "No previous experiments recorded."
    lines = [line for line in read_text(layout.history_path).splitlines() if line.strip()]
    if not lines:
        return "No previous experiments recorded."
    rows = [json.loads(line) for line in lines[-limit:]]
    parts = []
    for row in rows:
        parts.append(
            f"iter {row.get('iteration')}: status={row.get('status')} "
            f"val={format_float(row.get('val_score'))} notes={row.get('notes', '')}"
        )
    return "\n".join(parts)


def build_agent_prompt(
    project_dir: Path,
    best_score: float,
    best_metrics: dict[str, float],
    iteration: int,
    extra: str,
    history_summary: str,
) -> str:
    return f"""
You are running one cadresearch experiment in {project_dir}.

Read and follow:
- program.md
- README.md

Rules:
- Modify train.py only.
- Do not edit prepare.py, build_dataset.py, program.md, README.md, or dataset artifacts.
- Make exactly one focused experiment.
- Keep the code runnable with the existing fixed 5-minute training budget.
- Stop after editing train.py.

Current best score:
- val_score: {best_score:.6f}
- macro_iou: {best_metrics.get('macro_iou', float('nan')):.6f}
- param_rmse_norm: {best_metrics.get('param_rmse_norm', float('nan')):.6f}
- param_score: {best_metrics.get('param_score', float('nan')):.6f}

Recent experiment history:
{history_summary}

This is iteration {iteration}. Prefer a model-side change over extra loss tricks.
{extra.strip()}
""".strip()


def run_agent_edit(
    agent: str,
    prompt: str,
    project_dir: Path,
    log_path: Path,
    model: str | None,
) -> subprocess.CompletedProcess[str]:
    if agent == "none":
        write_text(log_path, "Agent disabled for this run.\n")
        return subprocess.CompletedProcess(args=["none"], returncode=0, stdout="", stderr="")
    if agent == "codex":
        cmd = ["codex", "exec", "-C", str(project_dir), "-s", "workspace-write", "--skip-git-repo-check", "--ephemeral"]
        if model:
            cmd.extend(["-m", model])
        cmd.append(prompt)
    elif agent == "claude":
        cmd = ["claude", "-p", "--dangerously-skip-permissions", "--add-dir", str(project_dir)]
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
    else:
        raise ValueError(f"Unsupported agent: {agent}")

    result = subprocess.run(cmd, cwd=str(project_dir), text=True, capture_output=True)
    combined = f"$ {' '.join(shlex.quote(part) for part in cmd)}\n"
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += ("\n" if result.stdout else "") + result.stderr
    write_text(log_path, combined)
    return result


def seed_state(
    layout: RunLayout,
    project_dir: Path,
    train_path: Path,
    baseline_log: Path | None,
) -> dict[str, Any]:
    train_text = read_text(train_path)
    metrics: dict[str, float] = {}
    baseline_log_rel = ""
    if baseline_log is not None:
        metrics = parse_metrics(read_text(baseline_log))
        baseline_log_rel = os.path.relpath(baseline_log, project_dir)
    if "val_score" not in metrics:
        raise SystemExit("Need a baseline log with val_score to seed the autonomous loop.")

    write_text(layout.best_train_path, train_text)
    state = {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "best_iteration": 0,
        "best_score": metrics["val_score"],
        "best_metrics": metrics,
        "best_train_sha256": sha256_text(train_text),
        "best_train_path": os.path.relpath(layout.best_train_path, project_dir),
        "best_log": baseline_log_rel,
        "iterations_completed": 0,
    }
    save_state(layout, state)

    record = {
        "timestamp": utc_now(),
        "iteration": 0,
        "status": "seed",
        "val_score": metrics.get("val_score"),
        "macro_iou": metrics.get("macro_iou"),
        "param_rmse_norm": metrics.get("param_rmse_norm"),
        "param_score": metrics.get("param_score"),
        "train_log": baseline_log_rel,
        "agent_log": "",
        "notes": "seeded from existing log",
    }
    append_history(layout, record)
    append_leaderboard(layout, record)
    return state


def maybe_seed(layout: RunLayout, project_dir: Path, train_path: Path, baseline_log: Path | None) -> dict[str, Any]:
    state = load_state(layout)
    if state is not None:
        return state
    return seed_state(layout, project_dir, train_path, baseline_log)


def run_training(
    project_dir: Path,
    train_command: str,
    train_log_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        train_command,
        cwd=str(project_dir),
        shell=True,
        text=True,
        capture_output=True,
    )
    combined = f"$ {train_command}\n"
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += ("\n" if result.stdout else "") + result.stderr
    write_text(train_log_path, combined)
    return result


def run_loop(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    train_path = project_dir / "train.py"
    layout = build_layout(project_dir, args.run_name, args.artifact_root)
    ensure_layout(layout)

    baseline_log = Path(args.baseline_log).resolve() if args.baseline_log else None
    state = maybe_seed(layout, project_dir, train_path, baseline_log)
    best_train_text = read_text(layout.best_train_path)
    write_text(train_path, best_train_text)

    for _ in range(args.iterations):
        state = load_state(layout) or state
        best_score = float(state["best_score"])
        best_metrics = dict(state.get("best_metrics", {}))
        iteration = int(state.get("iterations_completed", 0)) + 1

        agent_log_path = layout.logs_dir / f"iter_{iteration:04d}_agent.log"
        train_log_path = layout.logs_dir / f"iter_{iteration:04d}_train.log"
        candidate_path = layout.candidates_dir / f"iter_{iteration:04d}_train.py"

        best_train_text = read_text(layout.best_train_path)
        write_text(train_path, best_train_text)
        baseline_hash = sha256_text(best_train_text)

        prompt = build_agent_prompt(
            project_dir=project_dir,
            best_score=best_score,
            best_metrics=best_metrics,
            iteration=iteration,
            extra=args.agent_prompt_extra or "",
            history_summary=recent_summary(layout),
        )
        agent_result = run_agent_edit(args.agent, prompt, project_dir, agent_log_path, args.agent_model)

        record: dict[str, Any] = {
            "timestamp": utc_now(),
            "iteration": iteration,
            "agent_log": os.path.relpath(agent_log_path, project_dir),
            "train_log": os.path.relpath(train_log_path, project_dir),
            "status": "pending",
            "notes": "",
        }

        current_text = read_text(train_path)
        write_text(candidate_path, current_text)
        candidate_hash = sha256_text(current_text)

        if agent_result.returncode != 0:
            write_text(train_path, best_train_text)
            record["status"] = "agent_error"
            record["notes"] = f"agent exited {agent_result.returncode}"
            append_history(layout, record)
            append_leaderboard(layout, record)
            state["iterations_completed"] = iteration
            state["updated_at"] = utc_now()
            save_state(layout, state)
            continue

        if candidate_hash == baseline_hash:
            record["status"] = "no_change"
            record["notes"] = "agent left train.py unchanged"
            append_history(layout, record)
            append_leaderboard(layout, record)
            state["iterations_completed"] = iteration
            state["updated_at"] = utc_now()
            save_state(layout, state)
            continue

        try:
            py_compile_file(train_path)
        except subprocess.CalledProcessError as exc:
            write_text(train_path, best_train_text)
            record["status"] = "compile_error"
            record["notes"] = f"py_compile failed with exit {exc.returncode}"
            append_history(layout, record)
            append_leaderboard(layout, record)
            state["iterations_completed"] = iteration
            state["updated_at"] = utc_now()
            save_state(layout, state)
            continue

        if args.pre_train_command:
            pre_result = run_shell(args.pre_train_command, project_dir, layout.logs_dir / f"iter_{iteration:04d}_pretrain.log")
            if pre_result.returncode != 0:
                write_text(train_path, best_train_text)
                record["status"] = "sync_error"
                record["notes"] = f"pre-train command exited {pre_result.returncode}"
                append_history(layout, record)
                append_leaderboard(layout, record)
                state["iterations_completed"] = iteration
                state["updated_at"] = utc_now()
                save_state(layout, state)
                continue

        train_result = run_training(project_dir, args.train_command, train_log_path)
        train_metrics = parse_metrics(read_text(train_log_path))
        record.update(train_metrics)

        if train_result.returncode != 0 or "val_score" not in train_metrics:
            write_text(train_path, best_train_text)
            record["status"] = "train_error"
            record["notes"] = (
                f"train exited {train_result.returncode}" if train_result.returncode != 0 else "val_score missing from train log"
            )
            append_history(layout, record)
            append_leaderboard(layout, record)
            state["iterations_completed"] = iteration
            state["updated_at"] = utc_now()
            save_state(layout, state)
            continue

        score = float(train_metrics["val_score"])
        improved = score > (best_score + args.min_improvement)
        if improved:
            record["status"] = "keep"
            record["notes"] = f"improved by {score - best_score:.6f}"
            write_text(layout.best_train_path, current_text)
            state["best_iteration"] = iteration
            state["best_score"] = score
            state["best_metrics"] = train_metrics
            state["best_train_sha256"] = candidate_hash
            state["best_log"] = os.path.relpath(train_log_path, project_dir)
        else:
            record["status"] = "revert"
            record["notes"] = f"delta {score - best_score:.6f}"
            write_text(train_path, best_train_text)

        state["iterations_completed"] = iteration
        state["updated_at"] = utc_now()
        save_state(layout, state)
        append_history(layout, record)
        append_leaderboard(layout, record)

    final_state = load_state(layout) or state
    print(f"run_dir:    {layout.run_dir}")
    print(f"best_score: {final_state['best_score']:.6f}")
    print(f"best_iter:  {final_state['best_iteration']}")
    print(f"completed:  {final_state['iterations_completed']}")


def show_status(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    layout = build_layout(project_dir, args.run_name, args.artifact_root)
    state = load_state(layout)
    if state is None:
        raise SystemExit(f"No state found for run {args.run_name!r} in {layout.run_dir}")
    print(f"run_dir:              {layout.run_dir}")
    print(f"best_score:           {state['best_score']:.6f}")
    print(f"best_iteration:       {state['best_iteration']}")
    print(f"iterations_completed: {state['iterations_completed']}")
    metrics = state.get("best_metrics", {})
    for key in ("macro_iou", "param_rmse_norm", "param_score"):
        if key in metrics:
            print(f"{key}: {metrics[key]:.6f}")
    print(f"best_train:           {state['best_train_path']}")
    print(f"best_log:             {state.get('best_log', '')}")


def seed_only(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    train_path = project_dir / "train.py"
    layout = build_layout(project_dir, args.run_name, args.artifact_root)
    ensure_layout(layout)
    if layout.state_path.exists() and not args.force:
        raise SystemExit(f"State already exists at {layout.state_path}. Use --force to overwrite.")
    if args.force and layout.run_dir.exists():
        shutil.rmtree(layout.run_dir)
        ensure_layout(layout)
    baseline_log = Path(args.baseline_log).resolve() if args.baseline_log else None
    state = seed_state(layout, project_dir, train_path, baseline_log)
    print(f"Seeded {layout.run_dir} at best_score={state['best_score']:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous experiment harness for cadresearch.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-dir", default=".", help="Path to the cadresearch project.")
    common.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Name for this autonomous run.")
    common.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT, help="Artifact root relative to project dir.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", parents=[common], help="Seed run state from an existing best log.")
    seed_parser.add_argument("--baseline-log", required=True, help="Path to a log containing val_score.")
    seed_parser.add_argument("--force", action="store_true", help="Overwrite any existing run state.")
    seed_parser.set_defaults(func=seed_only)

    status_parser = subparsers.add_parser("status", parents=[common], help="Show current best status for a run.")
    status_parser.set_defaults(func=show_status)

    loop_parser = subparsers.add_parser("loop", parents=[common], help="Run autonomous edit/train/keep-or-revert iterations.")
    loop_parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run.")
    loop_parser.add_argument("--agent", choices=["codex", "claude", "none"], default="codex", help="Editing agent to invoke.")
    loop_parser.add_argument("--agent-model", default=None, help="Optional model override for the agent CLI.")
    loop_parser.add_argument("--agent-prompt-extra", default="", help="Extra instruction appended to the agent prompt.")
    loop_parser.add_argument("--baseline-log", default=None, help="Optional log to seed the run if state is missing.")
    loop_parser.add_argument("--pre-train-command", default="", help="Optional shell command to run before training, e.g. syncing train.py to a server.")
    loop_parser.add_argument("--train-command", required=True, help="Shell command that runs train.py and prints metrics to stdout.")
    loop_parser.add_argument("--min-improvement", type=float, default=0.0, help="Minimum required improvement over the current best.")
    loop_parser.set_defaults(func=run_loop)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
