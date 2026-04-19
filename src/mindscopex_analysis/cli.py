from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.console import Console

from mindscopex_analysis.pipeline import run_persona_comparison

console = Console()


def _configs_dir() -> str:
    """소스 트리(src/..) 또는 상위에 있는 configs/ 를 탐색."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        d = p / "configs"
        if (d / "config.yaml").is_file():
            return str(d)
    raise FileNotFoundError("configs/config.yaml 을 찾을 수 없습니다. 프로젝트 루트에 configs/ 가 있어야 합니다.")


@hydra.main(version_base=None, config_path=_configs_dir(), config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    console.print(f"[bold green]출력 디렉터리[/bold green] {run_dir}")
    summary = run_persona_comparison(cfg, output_dir=run_dir)
    console.print(summary)


if __name__ == "__main__":
    main()
