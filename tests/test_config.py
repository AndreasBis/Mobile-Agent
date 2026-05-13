from pathlib import Path

from mobile_agent.config import CFG


def test_default_model_path_points_to_llm_models() -> None:

    cfg = CFG()

    assert cfg.model_path == Path.home() / "models/gemma-4-E2B-it-Q4_K_M.gguf"


def test_config_accepts_custom_model_path() -> None:

    cfg = CFG(
        model_path=Path("/tmp/model.gguf"),
    )

    assert cfg.model_path == Path("/tmp/model.gguf")
