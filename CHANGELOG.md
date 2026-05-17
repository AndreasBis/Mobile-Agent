# Changelog

## 0.1.2 - Runtime Defaults

- Reduced the default context window from 65536 tokens to 32768 tokens.
- Disabled multi-modality to lower Android memory pressure.
- Added and commented out MTP arguments until Termux ships a llama.cpp build based on b9180 or newer.

## 0.1.1 - Initial Commit

- Added the initial Termux-focused `llama-cli` launcher for running a local Gemma GGUF model on a Samsung S25+.
- Added a conventional Python package layout under `src/mobile_agent`.
- Added package entrypoints for direct imports and `python -m mobile_agent`.
- Added focused tests for configuration defaults and `llama-cli` command construction.
- Documented setup, model download, run instructions, repository layout, and observed performance notes.
