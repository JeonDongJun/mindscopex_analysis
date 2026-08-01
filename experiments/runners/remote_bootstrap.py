"""Render the self-contained Python bootstrap sent to a Colab VM."""

from __future__ import annotations

import json
from typing import Any

QWEN35_TRANSFORMERS_REVISION = "b70d02fc724d04c916832ca4ead03ff05e8fb1ee"


def render_bootstrap(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return f'''\
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PAYLOAD = json.loads(r"""{serialized}""")
QWEN35_TRANSFORMERS_REVISION = "{QWEN35_TRANSFORMERS_REVISION}"


def run(cmd, *, cwd=None):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd], cwd=str(cwd) if cwd else None)


def run_capturing(cmd, log_path):
    """Run a command, teeing stdout+stderr live to our stdout and to log_path."""
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        proc.wait()
    return proc.returncode


def prepare_source():
    repo_dir = Path(PAYLOAD["remote_repo_dir"])
    source = PAYLOAD.get("source", "archive")
    if source == "archive":
        archive = Path(PAYLOAD["remote_source_archive"])
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(repo_dir))
        return repo_dir

    if source == "git":
        repo_url = PAYLOAD["repo_url"]
        repo_ref = PAYLOAD.get("repo_ref", "main")
        if (repo_dir / ".git").exists():
            run(["git", "-C", repo_dir, "fetch", "--all", "--tags"])
        else:
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            run(["git", "clone", repo_url, repo_dir])
        run(["git", "-C", repo_dir, "checkout", repo_ref])
        return repo_dir

    raise ValueError(f"Unknown source mode: {{source!r}}")


def ensure_dependencies(repo_dir):
    probe = subprocess.run(
        [sys.executable, "-c", "from transformers import AutoModelForMultimodalLM"],
        capture_output=True,
    )
    if probe.returncode != 0:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers @ "
            f"git+https://github.com/huggingface/transformers.git@{{QWEN35_TRANSFORMERS_REVISION}}",
            "torchvision",
            "pillow",
        ])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=repo_dir)


def main():
    repo_dir = prepare_source()
    os.environ["MINDSCOPEX_ROOT"] = str(repo_dir.resolve())
    os.chdir(repo_dir)
    ensure_dependencies(repo_dir)
    job_path = repo_dir / PAYLOAD["job_path"]
    output_root = Path(PAYLOAD["remote_output_root"])
    run_dir = output_root / PAYLOAD["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    job_log = run_dir / "job.log"
    returncode = run_capturing(
        [
            sys.executable,
            job_path,
            "--config",
            PAYLOAD["remote_config_path"],
            "--output-root",
            output_root,
        ],
        job_log,
    )

    # Always archive whatever exists (incl. job.log) so partial artifacts and the
    # captured stdout/stderr survive even when the job crashed or OOM-ed.
    archive_base = output_root / PAYLOAD["run_name"]
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", str(run_dir)))
    fallback_path = Path("/content") / f"{{PAYLOAD['run_name']}}_artifacts.zip"
    shutil.copy2(archive_path, fallback_path)
    print(
        f"REMOTE_ARTIFACT_ZIP={{archive_path}} "
        f"size={{archive_path.stat().st_size}}",
        flush=True,
    )
    print(
        f"REMOTE_ARTIFACT_FALLBACK={{fallback_path}} "
        f"size={{fallback_path.stat().st_size}}",
        flush=True,
    )
    if returncode != 0:
        print(f"JOB_FAILED returncode={{returncode}}", flush=True)
        sys.exit(returncode)


main()
'''
