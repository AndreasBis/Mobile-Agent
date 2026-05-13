import os
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class CFG:

    model_path: Path = Path.home() / "llm/models/gemma-4-E2B-it-Q4_K_M.gguf"

    system_prompt: str = (
        "You are Gemma-4-E2B, a helpful assistant running on a flagship smartphone."
    )

    context_tokens: int = 65536
    output_tokens: int = 4096

    threads: int = 8
    seed: int = 42

    llama_cli_binary: str = "llama-cli"
    flash_attention: str = "on"
    cache_memory: int = 6144
    color: str = "on"

    physical_batch_size: int = 256
    logical_batch_size: int = 1024

    kv_cache_dtype_k: str = "f16"
    kv_cache_dtype_v: str = "f16"

    temperature: float = 1.0
    min_p: float = 0.0
    top_p: float = 0.95
    top_k: int = 64


class Runner:

    def __init__(self, cfg: CFG) -> None:

        self.cfg = cfg

    def build_command(self) -> list[str]:

        command = [
            self.cfg.llama_cli_binary,
            "-cnv",
            "-m",
            str(self.cfg.model_path),
            "-sys",
            self.cfg.system_prompt,
            "-t",
            str(self.resolve_threads()),
            "-s",
            str(self.cfg.seed),
            "-fa",
            self.cfg.flash_attention,
            "-cram",
            str(self.cfg.cache_memory),
            "-co",
            self.cfg.color,
            "-c",
            str(self.cfg.context_tokens),
            "-n",
            str(self.cfg.output_tokens),
            "-ctk",
            self.cfg.kv_cache_dtype_k,
            "-ctv",
            self.cfg.kv_cache_dtype_v,
            "--temp",
            str(self.cfg.temperature),
            "--min-p",
            str(self.cfg.min_p),
            "--top-p",
            str(self.cfg.top_p),
            "--top-k",
            str(self.cfg.top_k),
            "-ub",
            str(self.cfg.physical_batch_size),
            "-b",
            str(self.cfg.logical_batch_size),
        ]

        return command

    def run(self) -> int:

        command = self.build_command()
        os.execvp(command[0], command)

        return 1

    def resolve_threads(self) -> int:

        try:
            completed_process = subprocess.run(
                ["nproc"],
                check=True,
                capture_output=True,
                text=True,
            )

        except (FileNotFoundError, subprocess.CalledProcessError):
            return self.cfg.threads

        try:
            return int(completed_process.stdout.strip())

        except ValueError:
            return self.cfg.threads


if __name__ == "__main__":
    raise SystemExit(Runner(CFG()).run())
