"""Launch configured experiments on Google Colab through google-colab-cli."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runners.collect_artifacts import make_source_archive, unpack_archive  # noqa: E402
from experiments.runners.config import job_name, load_toml, run_name, table  # noqa: E402
from experiments.runners.remote_bootstrap import render_bootstrap  # noqa: E402

JOB_PATHS = {
    "crt_text_responses": "experiments/jobs/crt_text_responses.py",
    "research_experiments": "experiments/jobs/research_experiments.py",
}
HARDWARE_RE = re.compile(r"\bHardware:\s*([^|]+)", re.IGNORECASE)


def repo_root() -> Path:
    result = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(result.strip()).resolve()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def run_cmd_retry(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    attempts: int = 6,
    delay: float = 5.0,
) -> subprocess.CompletedProcess:
    """Run a flaky file-transfer command, retrying on non-zero exit.

    ``colab upload``/``download`` intermittently report a just-created remote
    file as "not found" right after ``colab exec`` finishes; a short retry loop
    lets the remote filesystem settle and makes the transfer reliable.
    """

    result: subprocess.CompletedProcess | None = None
    for attempt in range(1, attempts + 1):
        print(f"+ {' '.join(cmd)}  (attempt {attempt}/{attempts})", flush=True)
        result = subprocess.run(cmd, cwd=cwd, check=False, text=True)
        if result.returncode == 0:
            return result
        if attempt < attempts:
            print(
                f"  command failed (exit {result.returncode}); retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
    raise subprocess.CalledProcessError(result.returncode if result else 1, cmd)


def check_colab_cli() -> None:
    if shutil.which("colab") is None:
        raise SystemExit(
            "Could not find `colab`. Install it in WSL with "
            "`uv tool install google-colab-cli` or `pipx install google-colab-cli`."
        )


def git_value(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "filter.nbstripout.clean=cat",
                "-c",
                "filter.nbstripout.smudge=cat",
                "-c",
                "filter.nbstripout.required=false",
                *args,
            ],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def resolve_configs(path: Path) -> list[Path]:
    config = load_toml(path)
    suite = config.get("suite")
    if not isinstance(suite, dict):
        return [path]
    raw_configs = suite.get("configs")
    if not isinstance(raw_configs, list) or not all(isinstance(item, str) for item in raw_configs):
        raise ValueError("[suite].configs must be a list of config paths")
    base = path.parent
    return [
        (base / item).resolve() if not Path(item).is_absolute() else Path(item)
        for item in raw_configs
    ]


def session_hardware(status_output: str) -> str | None:
    match = HARDWARE_RE.search(status_output)
    return match.group(1).strip().upper() if match else None


def requested_hardware(gpu: str | None, tpu: str | None) -> str:
    return (gpu or tpu or "CPU").upper()


def ensure_session(
    session: str,
    gpu: str | None,
    tpu: str | None,
    *,
    allocation_attempts: int,
    allocation_retry_delay: float,
) -> None:
    cmd = ["colab", "status", "-s", session]
    print("+ " + " ".join(cmd), flush=True)
    status = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if status.stdout:
        print(status.stdout, end="" if status.stdout.endswith("\n") else "\n", flush=True)
    if status.returncode == 0 and "not found" not in status.stdout.lower():
        actual = session_hardware(status.stdout)
        requested = requested_hardware(gpu, tpu)
        if actual == requested:
            return
        print(
            f"[launch] session {session!r} has {actual or 'unknown hardware'}, "
            f"but this run requires {requested}; replacing the session.",
            flush=True,
        )
        run_cmd(["colab", "stop", "-s", session])

    cmd = ["colab", "new", "-s", session]
    if gpu:
        cmd.extend(["--gpu", gpu])
    if tpu:
        cmd.extend(["--tpu", tpu])
    result: subprocess.CompletedProcess | None = None
    for attempt in range(1, allocation_attempts + 1):
        print(
            f"+ {' '.join(cmd)}  (allocation attempt {attempt}/{allocation_attempts})",
            flush=True,
        )
        result = subprocess.run(cmd, check=False, text=True)
        if result.returncode == 0:
            return
        if attempt < allocation_attempts:
            print(
                f"[launch] accelerator allocation failed (exit {result.returncode}); "
                f"retrying in {allocation_retry_delay:.0f}s",
                flush=True,
            )
            time.sleep(allocation_retry_delay)
    raise subprocess.CalledProcessError(result.returncode if result else 1, cmd)


def write_local_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_one(
    config_path: Path,
    *,
    root: Path,
    session: str,
    keep_session: bool,
    default_gpu: str | None,
    default_tpu: str | None,
    exec_timeout: float,
    allocation_attempts: int,
    allocation_retry_delay: float,
    dry_run: bool,
) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    job = job_name(config)
    if job not in JOB_PATHS:
        valid = ", ".join(sorted(JOB_PATHS))
        raise ValueError(f"Unknown job {job!r}; choose one of: {valid}")

    colab_cfg = table(config, "colab")
    remote_cfg = table(config, "remote")
    source = str(remote_cfg.get("source", "archive"))
    remote_repo_dir = str(remote_cfg.get("repo_dir", "/content/mindscopex_analysis"))
    remote_output_root = str(remote_cfg.get("output_root", "/content/mindscopex_artifacts"))
    remote_config_path = f"/content/{name}_config.toml"
    remote_source_archive = f"/content/{name}_source.zip"
    gpu = default_gpu if default_gpu is not None else colab_cfg.get("gpu")
    tpu = default_tpu if default_tpu is not None else colab_cfg.get("tpu")

    local_run_dir = root / "results" / "runs" / f"{timestamp()}_{name}"
    local_run_dir.mkdir(parents=True, exist_ok=True)
    local_config = local_run_dir / "config.toml"
    shutil.copyfile(config_path, local_config)

    payload = {
        "run_name": name,
        "job": job,
        "job_path": JOB_PATHS[job],
        "source": source,
        "remote_repo_dir": remote_repo_dir,
        "remote_output_root": remote_output_root,
        "remote_config_path": remote_config_path,
        "remote_source_archive": remote_source_archive,
        "repo_url": remote_cfg.get(
            "repo_url",
            git_value(["config", "--get", "remote.origin.url"], root),
        ),
        "repo_ref": remote_cfg.get("repo_ref", git_value(["rev-parse", "HEAD"], root) or "main"),
    }

    write_local_manifest(
        local_run_dir / "launcher_manifest.json",
        {
            "config": str(config_path),
            "session": session,
            "gpu": gpu,
            "tpu": tpu,
            "keep_session": keep_session,
            "payload": payload,
            "git_commit": git_value(["rev-parse", "HEAD"], root),
            "git_status": git_value(["status", "--short"], root),
        },
    )

    bootstrap = local_run_dir / "remote_bootstrap.py"
    bootstrap.write_text(render_bootstrap(payload), encoding="utf-8")

    if source == "archive":
        make_source_archive(root, local_run_dir / "source.zip")
    elif source != "git":
        raise ValueError(f"Unknown [remote].source={source!r}")

    if dry_run:
        print(f"Dry run prepared {local_run_dir}")
        return local_run_dir

    ensure_session(
        session,
        str(gpu) if gpu else None,
        str(tpu) if tpu else None,
        allocation_attempts=allocation_attempts,
        allocation_retry_delay=allocation_retry_delay,
    )
    run_cmd_retry(["colab", "upload", "-s", session, str(local_config), remote_config_path])
    if source == "archive":
        run_cmd_retry(
            [
                "colab",
                "upload",
                "-s",
                session,
                str(local_run_dir / "source.zip"),
                remote_source_archive,
            ]
        )
    exec_result = run_cmd(
        [
            "colab",
            "exec",
            "-s",
            session,
            "--timeout",
            str(exec_timeout),
            "-f",
            str(bootstrap),
        ],
        check=False,
    )

    # Retrieve artifacts and logs even when the remote job failed: the bootstrap
    # always archives run_dir (incl. job.log), so partial results and the captured
    # stdout/stderr survive for diagnosis before we surface the failure.
    remote_zip = f"{remote_output_root}/{name}.zip"
    remote_fallback_zip = f"/content/{name}_artifacts.zip"
    local_zip = local_run_dir / "artifacts.zip"
    artifact_downloaded = False
    for candidate in (remote_zip, remote_fallback_zip):
        try:
            run_cmd_retry(
                ["colab", "download", "-s", session, candidate, str(local_zip)],
                attempts=12,
                delay=10.0,
            )
            unpack_archive(local_zip, local_run_dir / "artifacts")
            artifact_downloaded = True
            break
        except subprocess.CalledProcessError as exc:
            print(
                f"[launch] artifact download failed for {name} from {candidate}: {exc}",
                flush=True,
            )
    run_cmd(
        ["colab", "log", "-s", session, "-o", str(local_run_dir / "colab_log.md")],
        check=False,
    )
    run_cmd(
        ["colab", "log", "-s", session, "-o", str(local_run_dir / "colab_log.jsonl")],
        check=False,
    )
    if exec_result.returncode != 0:
        raise RuntimeError(
            f"remote job {name!r} failed (colab exec returncode={exec_result.returncode}); "
            f"inspect {local_run_dir}"
        )
    if not artifact_downloaded:
        raise RuntimeError(
            f"remote job {name!r} finished but no artifact archive could be downloaded; "
            f"inspect {local_run_dir}"
        )
    return local_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_or_suite", type=Path)
    parser.add_argument("--session", "-s", default="mindscopex")
    parser.add_argument("--gpu", "-g", default=None)
    parser.add_argument("--tpu", default=None)
    parser.add_argument("--keep", action="store_true", help="Leave the Colab VM running.")
    parser.add_argument(
        "--exec-timeout",
        default=3600.0,
        type=float,
        help="Seconds to wait for `colab exec` to finish.",
    )
    parser.add_argument(
        "--allocation-attempts",
        default=6,
        type=int,
        help="Attempts to allocate the requested Colab accelerator.",
    )
    parser.add_argument(
        "--allocation-retry-delay",
        default=30.0,
        type=float,
        help="Seconds between Colab accelerator allocation attempts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but do not call Colab.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "In a suite, keep going to the next config if one fails "
            "(log it, reset the session, and continue instead of aborting)."
        ),
    )
    args = parser.parse_args()
    if args.allocation_attempts < 1:
        parser.error("--allocation-attempts must be at least 1")
    if args.allocation_retry_delay < 0:
        parser.error("--allocation-retry-delay must be non-negative")

    root = repo_root()
    configs = resolve_configs(args.config_or_suite.resolve())
    if not args.dry_run:
        check_colab_cli()

    launched: list[Path] = []
    failed: list[str] = []
    try:
        for config_path in configs:
            try:
                launched.append(
                    launch_one(
                        config_path,
                        root=root,
                        session=args.session,
                        keep_session=args.keep,
                        default_gpu=args.gpu,
                        default_tpu=args.tpu,
                        exec_timeout=args.exec_timeout,
                        allocation_attempts=args.allocation_attempts,
                        allocation_retry_delay=args.allocation_retry_delay,
                        dry_run=args.dry_run,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                print(f"[launch] config {config_path.name} failed: {exc}", flush=True)
                print("[launch] --continue-on-error: resetting session, moving on", flush=True)
                failed.append(config_path.name)
                if not args.dry_run:
                    # Drop the (possibly dead) session so the next config gets a fresh VM.
                    run_cmd(["colab", "stop", "-s", args.session], check=False)
    finally:
        if not args.keep and not args.dry_run:
            run_cmd(["colab", "stop", "-s", args.session], check=False)

    print("Prepared runs:")
    for path in launched:
        print(f"- {path}")
    if failed:
        print("Failed configs (see per-run dirs / streamed log):")
        for name in failed:
            print(f"- {name}")


if __name__ == "__main__":
    main()
