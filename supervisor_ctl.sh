#!/bin/zsh
set -euo pipefail

PROJECT_DIR=/Volumes/bkawk/projects/mesh-para/cadresearch
SESSION_NAME=mesh_para_supervisor
SOURCE_RUN=boundary512_refocused
POLL_SECONDS=600
STALE_SECONDS=1800
SUPERVISOR_NAME=autonomy
LOG_PATH="$PROJECT_DIR/artifacts/manager/default/logs/autonomy.log"
PID_PATH="$PROJECT_DIR/artifacts/manager/default/logs/autonomy.pid"

manager_cmd=(
  python3 "$PROJECT_DIR/manager.py"
  --project-dir "$PROJECT_DIR"
  supervise
  --source-run "$SOURCE_RUN"
  --poll-seconds "$POLL_SECONDS"
  --stale-seconds "$STALE_SECONDS"
  --supervisor-name "$SUPERVISOR_NAME"
)

manager_cmd_string="${(j: :)${(q)manager_cmd[@]}}"

screen_exists() {
  (screen -ls 2>/dev/null || true) | grep -q "\.${SESSION_NAME}"
}

supervisor_pid() {
  ps -ax -o pid=,command= | rg --no-line-number "manager.py --project-dir $PROJECT_DIR supervise --source-run $SOURCE_RUN --poll-seconds $POLL_SECONDS --stale-seconds $STALE_SECONDS --supervisor-name $SUPERVISOR_NAME" | rg -v -- "--once" | awk '{print $1}' | head -n 1 || true
}

running_pid() {
  if [[ -f "$PID_PATH" ]]; then
    local pid
    pid="$(cat "$PID_PATH" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  local pid
  pid="$(supervisor_pid)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid"
    return 0
  fi
  return 1
}

write_pid() {
  local pid
  pid="$(supervisor_pid)"
  if [[ -n "$pid" ]]; then
    mkdir -p "$(dirname "$PID_PATH")"
    printf '%s\n' "$pid" > "$PID_PATH"
  fi
}

start_supervisor() {
  if screen_exists || running_pid >/dev/null; then
    echo "Supervisor already running."
    status_supervisor
    return 0
  fi

  mkdir -p "$(dirname "$LOG_PATH")"
  screen -dmS "$SESSION_NAME" /bin/zsh -lc \
    "cd '$PROJECT_DIR' && exec ${manager_cmd_string} >> '$LOG_PATH' 2>&1"

  sleep 1
  if ! running_pid >/dev/null; then
    echo "Supervisor failed to start. Recent log output:"
    tail -n 20 "$LOG_PATH" || true
    return 1
  fi

  write_pid
  echo "Supervisor started."
  status_supervisor
}

stop_supervisor() {
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  pkill -f "python3 $PROJECT_DIR/manager.py --project-dir $PROJECT_DIR supervise --source-run $SOURCE_RUN" >/dev/null 2>&1 || true
  rm -f "$PID_PATH"
  echo "Supervisor stopped."
}

status_supervisor() {
  local pid="none"
  if running_pid >/dev/null; then
    pid="$(running_pid)"
  fi
  echo "session: $(screen_exists && echo present || echo missing)"
  echo "pid: $pid"
  echo "log: $LOG_PATH"
  tail -n 5 "$LOG_PATH" 2>/dev/null || true
}

run_once() {
  cd "$PROJECT_DIR"
  "${manager_cmd[@]}" --once
}

case "${1:-status}" in
  start)
    start_supervisor
    ;;
  stop)
    stop_supervisor
    ;;
  restart)
    stop_supervisor
    start_supervisor
    ;;
  status)
    status_supervisor
    ;;
  once)
    run_once
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|once}" >&2
    exit 1
    ;;
esac
