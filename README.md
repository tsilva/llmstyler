<div align="center">
  <img src="./logo.png" alt="llmstyler logo" width="412" />

  **🧵 Change how your LLM talks 🧵**
</div>

llmstyler is a Python CLI for building speaking-style supervised fine-tuning
datasets and publishing the resulting model artifacts. It starts with YAML
configs, uses datamixxer to sample chat rows from Hugging Face datasets, rewrites selected
assistant responses through OpenRouter, and generates Runbook training files for
Unsloth QLoRA jobs.

The main workflow is: build a reusable base mix, restyle the marked rows, then
launch a Runbook training step that publishes versioned Hugging Face datasets and
model exports.

## Install

```bash
git clone https://github.com/tsilva/llmstyler.git
cd llmstyler
python3 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the CLI from the repo root:

```bash
llmstyler --help
```

## Simplest workflow

Create a complete pipeline config from a few values:

```bash
llmstyler new my-style --owner your-hf-user --style-prompt prompt.txt
```

Check local prerequisites, estimate rewrite cost, then run the local dataset
pipeline:

```bash
llmstyler doctor configs/pipelines/my-style.yaml
llmstyler estimate configs/pipelines/my-style.yaml
llmstyler run configs/pipelines/my-style.yaml
```

`llmstyler run` is local-first: it builds the base mix, restyles rows, and
generates Runbook training files without publishing. Remote training expects the
restyled dataset to be available on Hugging Face, so launch publishing and remote
training explicitly:

```bash
llmstyler run configs/pipelines/my-style.yaml --publish
```

Inspect generated artifacts and cached pipeline state:

```bash
llmstyler inspect configs/pipelines/my-style.yaml
```

## Advanced commands

```bash
llmstyler build-mix configs/base_mixes/smoltalk_style_mix.yaml

llmstyler restyle configs/styles/trump_public_speaking.yaml --estimate-only
llmstyler restyle configs/styles/trump_public_speaking.yaml

llmstyler run configs/pipelines/trump.yaml
llmstyler run configs/pipelines/trump.yaml --publish training.num_epochs=1

llmstyler train-style configs/pipelines/trump.yaml
llmstyler train-style configs/pipelines/trump.yaml training.num_epochs=1 exports.onnx.enabled=false
llmstyler train-style configs/pipelines/trump.yaml --dry-run training.num_epochs=1
llmstyler train-style configs/pipelines/trump.yaml --force

llmstyler runstep configs/train/qwen25_3b_trump.yaml
llmstyler runstep configs/train/qwen25_3b_trump.yaml --no-run
```

## Notes

- Python 3.11 or newer is required.
- `datamixxer` is required for `llmstyler build-mix`. Install it from
  `https://github.com/tsilva/datamixxer`.
- Example configs live under `configs/pipelines/`, `configs/base_mixes/`,
  `configs/styles/`, and `configs/train/`.
- `llmstyler new` writes full pipeline YAML from a compact set of flags. The
  generated config keeps dataset publishing disabled by default; pass
  `llmstyler run ... --publish` when you want Hub uploads and remote training.
- `OPENROUTER_API_KEY` is required for `llmstyler restyle` unless using
  `--estimate-only`. `llmstyler estimate` does not call OpenRouter.
- `llmstyler train-style` composes a full pipeline config with OmegaConf, so any
  nested value can be overridden from the CLI with dotlist syntax such as
  `owner=my-hf-org`, `model.base_model=...`, `training.num_epochs=1`, or
  `exports.onnx.enabled=false`.
- `llmstyler train-style` records per-step hashes in
  `runs/<pipeline-id>/pipeline/status.json`. A step is skipped when its resolved
  config hash is unchanged and its expected artifacts still exist. Use `--force`
  to rerun everything.
- `HF_TOKEN`, or an authenticated Hugging Face CLI session, is required because
  dataset and model artifact publishing is enabled by default in pipeline configs. Pass
  `--no-push-to-hub` to dataset commands for local-only runs. Publishing commands
  preflight Hugging Face auth and repository creation before running expensive
  generation work.
- Hub uploads are public by default. Set `hub.private: true` in a dataset config
  only when the target repository should be private.
- `llmstyler runstep` writes the generated Runbook files and launches `runbook`
  with the GPU, timeout, output notebook path, and secrets from the training
  config. Use `--no-run` to generate files without launching remote training.
- The `runbook` CLI must be installed before launching a runstep. Install it
  from `https://github.com/tsilva/runbook`.
- Remote training expects Modal/Runbook secrets named `huggingface-secret` and
  optionally `wandb-secret`.
- Generated local artifacts are ignored by git: `datasets/`, `runs/`, and
  `outputs/`.
- Published artifact names should be immutable and versioned, such as
  `owner/style-mix-restyled-name-v1`, `owner/model-style-qlora-v1`,
  `owner/model-style-gguf-v1`, and `owner/model-style-onnx-v1`.
- `llmstyler build-mix` writes the datamixxer artifact under
  `.datamixxer/mixes/<hash>/` and mirrors it to `datasets/basemix_restyle/` for
  restyle configs.
- Restyle outputs include a checkpoint, preview file, manifest, and dataset card.
  Training outputs include model cards for adapter, merged, GGUF, and optional
  ONNX exports.

## Architecture

![llmstyler architecture diagram](./architecture.png)

## License

No license file is present in this repository.
