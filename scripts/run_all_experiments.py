"""Run all experiment configuration files found in a directory.

This script discovers all `.json` config files under `params/experiments` (by default)
and launches training runs using the same invocation pattern provided:

    uv run python esnlir/training/train.py --config-file <config.json>

For each config, a log file is created in the logs directory named after the
config file (e.g. `model-foo.json` -> `logs/model-foo.log`). Stdout and stderr
are merged in that log. You can disable log file creation with `--no-logs` to
avoid large disk usage (output goes directly to the console and is not saved).

Features:
  * --configs-dir to change where configs are discovered.
  * --logs-dir to change where logs are written.
  * --skip-existing to skip runs whose log file already exists (resume).
  * --dry-run to only print the commands that would be executed.
  * --parallel N to run up to N experiments concurrently (default 1 = sequential).
  * Graceful Ctrl+C handling: running child processes are terminated.

Examples (PowerShell):
  uv run python scripts/run_all_experiments.py --dry-run
  uv run python scripts/run_all_experiments.py --parallel 2 --skip-existing

Note: Parallel execution will start processes simultaneously; ensure your
machine has enough GPU/CPU resources or keep it sequential.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import List
import subprocess
import threading
import queue


def find_config_files(configs_dir: Path) -> List[Path]:
    return sorted([p for p in configs_dir.glob("*.json") if p.is_file()])


def build_command(config_path: Path) -> List[str]:
    return [
        "uv",
        "run",
        "python",
        "esnlir/training/train.py",
        "--config-file",
        str(config_path.as_posix()),
    ]


def run_command(cmd: List[str], log_file: Path, dry_run: bool, no_logs: bool) -> int:
    if dry_run:
        target = "(no logs)" if no_logs else f"> {log_file}"
        print(f"DRY-RUN: {' '.join(cmd)} {target}")
        return 0
    if no_logs:
        # Directly stream to console; no file opened.
        print(f"[START] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("Interrupted. Killed process.")
            return 130
        print(f"[END] exit={code}")
        return code
    # Logging enabled
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"Interrupted. Killed process for {log_file.name}")
            return 130


def worker(task_queue: "queue.Queue[tuple[List[str], Path]]", dry_run: bool, results: List[tuple[Path, int]], no_logs: bool):
    while True:
        try:
            cmd, log_file = task_queue.get_nowait()
        except queue.Empty:
            break
        exit_code = run_command(cmd, log_file, dry_run, no_logs=no_logs)
        results.append((log_file, exit_code))
        task_queue.task_done()


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="Run all experiment configs.")
    parser.add_argument("--configs-dir", type=Path, default=Path("params/experiments"), help="Directory containing experiment JSON config files.")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Directory where log files will be written.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose log file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--parallel", type=int, default=1, help="Maximum number of concurrent runs.")
    parser.add_argument("--no-logs", action="store_true", help="Do not persist logs; stream output to console only.")
    return parser.parse_args(argv)


def build_tasks(config_files: List[Path], logs_dir: Path, skip_existing: bool, no_logs: bool) -> "queue.Queue[tuple[List[str], Path]]":
    tasks: "queue.Queue[tuple[List[str], Path]]" = queue.Queue()
    for cfg in config_files:
        if no_logs:
            # Use dummy path (stem only) for summary reporting
            log_path = Path(cfg.stem)
        else:
            log_path = logs_dir / (cfg.stem + ".log")
            if skip_existing and log_path.exists():
                print(f"SKIP (exists): {log_path}")
                continue
        tasks.put((build_command(cfg), log_path))
    return tasks


def execute_tasks(tasks: "queue.Queue[tuple[List[str], Path]]", parallelism: int, dry_run: bool, no_logs: bool) -> List[tuple[Path, int]]:
    parallelism = max(1, parallelism)
    results: List[tuple[Path, int]] = []
    threads = [threading.Thread(target=worker, args=(tasks, dry_run, results, no_logs), daemon=True) for _ in range(parallelism)]
    for t in threads:
        t.start()
    try:
        tasks.join()
    except KeyboardInterrupt:
        print("\nReceived interrupt. Attempting to stop remaining tasks...")
    finally:
        for t in threads:
            t.join()
    return results


def summarize(results: List[tuple[Path, int]]) -> int:
    failures = [p for p, code in results if code != 0]
    print("\nSummary:\n---------")
    for p, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"{p.name}: {status}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f" - {f.name}")
        return 1
    return 0


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if not args.configs_dir.exists():
        print(f"Configs directory does not exist: {args.configs_dir}", file=sys.stderr)
        return 2
    config_files = find_config_files(args.configs_dir)
    if not config_files:
        print(f"No config files found in {args.configs_dir}")
        return 0
    tasks = build_tasks(config_files, args.logs_dir, args.skip_existing, args.no_logs)
    if tasks.empty():
        print("No tasks to run (all skipped).")
        return 0
    mode = "NO-LOGS" if args.no_logs else f"logs dir={args.logs_dir}"
    print(f"Starting {tasks.qsize()} experiment(s) with parallelism={args.parallel} ({mode})...")
    results = execute_tasks(tasks, args.parallel, args.dry_run, args.no_logs)
    return summarize(results)


if __name__ == "__main__":
    if os.name != "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main(sys.argv[1:]))
