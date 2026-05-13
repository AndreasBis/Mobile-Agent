## Mobile Agent

Run a local Gemma GGUF model on a Samsung S25+ through Termux and `llama-cli`.

This project is a small launcher around the Termux `llama-cpp` package. It keeps Python responsible for configuration and command construction, then replaces the Python process with `llama-cli` so inference runs directly in the llama.cpp runtime.

The goal is to keep the phone setup simple: one local GGUF model, one Python package, and a predictable command path that can be adjusted without turning the project into a full Android application.

## Repository Layout

```text
.
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── src/
│   └── mobile_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── runner.py
│       ├── __main__.py
│       └── main.py
├── tests/
│   ├── test_config.py
│   └── test_runner.py
└── sandbox/
```

The launcher implementation lives in `src/mobile_agent/main.py`. The surrounding package files provide stable import paths, `python -m mobile_agent` support, and focused test targets for the configuration and command builder.

## Target Environment

The setup is designed for:

- Samsung S25+
- Termux on Android
- Termux `llama-cpp` package
- Python 3.12+
- A local GGUF model file

The default model path is:

```bash
~/models/gemma-4-E2B-it-Q4_K_M.gguf
```

## Fresh Termux Setup

Update Termux, grant storage access, and install the required packages:

```bash
pkg update && pkg upgrade -y
termux-setup-storage
pkg install -y llama-cpp python git wget curl openssh
mkdir -p ~/models ~/logs
```

Optional SSH setup:

```bash
passwd
sshd
termux-wake-lock
```

Connect from another machine:

```bash
ssh -p 8022 <termux-username>@<phone-ip>
```

Release the wake lock when SSH use is done:

```bash
termux-wake-unlock
```

## Download The Model

Install the Hugging Face download dependencies:

```bash
python -m pip install --no-deps huggingface_hub
python -m pip install filelock fsspec packaging pyyaml requests tqdm typing-extensions httpx
export HF_HUB_DISABLE_XET=1
echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc
source ~/.bashrc
```

Download the default model:

```bash
cd ~/models
hf download unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf --local-dir .
```

## Run

Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

Run the launcher:

```bash
python -m mobile_agent
```

If you want to run the launcher script directly from the cloned repository:

```bash
python src/mobile_agent/main.py
```

## Tests

Install test dependencies:

```bash
python -m pip install -e . pytest
```

Run the tests from the repository root:

```bash
python -m pytest
```

## Performance Notes

Observed performance with the default Q4_K_M model on the S25+ is approximately:

- Prompt processing: around `48 t/s`
- Token generation: around `15 t/s`

Prompt processing is usually sensitive to `-b` and `-ub`. Token generation for a single user is usually not, because decode runs one new token at a time.

Changing KV cache type between `f16`, `q8_0`, and `q4_0` may have little effect on generation speed for this setup. That usually means decode is limited by CPU kernels, memory bandwidth, scheduling, or thermals rather than KV cache format.

This project intentionally stays with `llama-cli` and GGUF for the main path. LiteRT-LM can be faster on supported Android runtimes, especially when hardware acceleration is available, but it is not a direct replacement for this Termux-based CLI workflow.
