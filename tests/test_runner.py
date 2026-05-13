from pathlib import Path

from mobile_agent.main import CFG
from mobile_agent.runner import Runner


def test_command_uses_configured_model_path() -> None:

    cfg = CFG(
        model_path=Path("/tmp/model.gguf"),
    )

    runner = Runner(cfg)
    runner.resolve_threads = lambda: 6

    command = runner.build_command()

    assert "-m" in command
    assert "/tmp/model.gguf" in command


def test_command_passes_batch_and_cache_settings() -> None:

    cfg = CFG(
        physical_batch_size=512,
        logical_batch_size=2048,
        kv_cache_dtype_k="f16",
        kv_cache_dtype_v="f16",
    )

    runner = Runner(cfg)
    runner.resolve_threads = lambda: 6

    command = runner.build_command()

    assert command[command.index("-ub") + 1] == "512"
    assert command[command.index("-b") + 1] == "2048"
    assert command[command.index("-ctk") + 1] == "f16"
    assert command[command.index("-ctv") + 1] == "f16"


def test_command_uses_resolved_thread_count() -> None:

    cfg = CFG()
    runner = Runner(cfg)
    runner.resolve_threads = lambda: 6

    command = runner.build_command()

    assert command[command.index("-t") + 1] == "6"
