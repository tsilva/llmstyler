> Historical experiment record. The former root Infisical binding is preserved as
> `docs/legacy-infisical-metadata.json`; its key retrieval flow is retired. Current
> agent work uses the explicit keyless local entrypoint documented in README.md.
> The commands below record the earlier run and do not authorize live credential
> access, publishing, session creation or training.

# Warm-direct smoke experiment — October 7, 2026

## Result

The complete loop ran successfully: pinned datamixxer data, warm-direct rewrites,
fidelity screening, Studio QLoRA training on beast-3, and comparison against
base and prompt-only responses. The adapter is saved locally; nothing was
published. This small run verifies the integration but does not establish a
reliable improvement in style or answer quality.

Credentials now come from the dedicated `llmstyler` Infisical project,
Development (`dev`), root folder (`/`). Its new OpenRouter key was validated
and read back, with a $10 total limit. Reported rewriting cost was $0.18359;
total key usage after rewriting and fidelity screening was $0.28475.

## Data and fidelity

The pinned SmolTalk2 mix contains 58 training conversations and six untouched
test conversations. Twenty-eight rows were selected for restyling, requiring
56 assistant-message rewrites by Grok 4.3. All rewrites completed. Ordering,
roles, user/system/tool content, and unselected rows were verified unchanged.

Review exposed content loss, including a JSON response rewritten as prose.
The experiment gate freezes fenced code/structured blocks and asks Grok 4.3 to
screen factual detail, caveats, uncertainty, and instruction preservation.
Ten conversations were restored to their originals: rows 2, 9, 19, 20, 31, 33,
47, 49, 52, and 55. The actual blend is 18 styled and 40 original conversations.
The same teacher judged fidelity, so acceptance is not an independent guarantee;
sampled inspection also found aggressive compression. Production rewriting
needs stronger preservation checks and reviewed examples.

The reviewed conversations have a maximum length of 3,378 tokens. None exceeds
the 4,096-token context. Held-out conversations stayed outside training,
fidelity screening, and model selection.

## Training

Base model: `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`, pinned revision
`216d96b52c97a3c8294776df8477592f6828981c`. Studio loaded that exact local snapshot.
QLoRA used rank 8, alpha 16, dropout 0, batch size 2, accumulation 4, learning
rate 0.0003, warmup ratio 0.05, seed 3407, and one epoch. Packing was disabled;
loss was restricted to assistant completions. Evaluation during training was
disabled; the held-out prompts were evaluated afterward.

Studio job `job_20261007_184010_a908f603` completed eight optimizer steps and
saved an adapter. Final batch loss was 1.4669, not an evaluation score.
Remote adapter directory:

```text
/home/tsilva/.local/share/llmstyler-unsloth/studio/outputs/216d96b52c97a3c8294776df8477592f6828981c__project-warm-direct-smoke-qlora_1791394834
```

## Comparison

Each condition used six held-out first-response prompts and eight fixed
diagnostics covering facts, uncertainty, math, code, JSON formatting,
explanation, brevity, and multiple turns. Generation was greedy, with a
192-new-token limit and seed 3407. Prompt-only used the compiled warm-direct
pack and three examples. Base and adapter used Qwen's ordinary system prompt.

| Condition | Completed | Mean output tokens | Token-cap hits |
| --- | ---: | ---: | ---: |
| Base | 14/14 | 137.2 | 6 |
| Prompt-only | 14/14 | 83.7 | 1 |
| Adapter | 14/14 | 103.6 | 4 |

The cap censors lengths; these numbers do not establish a percentage improvement
in conciseness or quality. Fourteen prompts cannot establish general capability.

All three calculated the discounted price as 68 euros and supplied a working
larger-of-two-numbers function. All failed strict JSON-only output by adding
code fences. The adapter still failed the lightning/thunder explanation. Its
uncertainty answer admitted insufficient information but continued into a
long, capped list. Its decline email was brief and actually declined, unlike
the prompt-only response, but invented an upcoming meeting. Held-out adapter
responses retained filler, unsupported biographical claims, and long
introductions. The warm-direct voice is not reliably learned.

## Artifacts and reproduction

Generated artifacts are under `runs/warm-direct-smoke/`, ignored by Git:
`evaluation-spec.json`, `model-pin.json`, `comparison-summary.json`, three
`evaluation-*.json` files, `studio-request.json`, `studio-start.json`,
`studio-status.json`, `reviewed-token-lengths.json`, `experiment-status.json`,
`reviewed-studio/{train.jsonl,review.json}`, and `adapter/`. Raw generated data
and checkpoints remain under `datasets/`. The reviewed training SHA256 is
`0f8bab487f100d90918530b77b47cf44dd104be129f8f2d68712bd7155542634`.

```bash
python3 scripts/infisical.py run run configs/pipelines/warm-direct-smoke.yaml --no-run \
  restyle.style.provider=openrouter restyle.style.model=x-ai/grok-4.3 \
  restyle.style.base_url=http://127.0.0.1:8082/api/v1 \
  restyle.style.reasoning_effort=null restyle.style.request_timeout=90 \
  restyle.style.workers=8 \
  restyle.style.pricing_usd_per_1m.input=1.25 \
  restyle.style.pricing_usd_per_1m.output=2.5
llmstyler export-studio configs/pipelines/warm-direct-smoke.yaml --output-dir <new-bundle>
.venv/bin/python experiments/review_smoke.py
```

The checked-in recipe now defaults to AgentBridge, `codex/gpt-6.1-sol`, and high
reasoning. The overrides above restore the original OpenRouter teacher settings
for reproducing this run. New Codex rewrites are a separate experiment; the
historical Grok judge script and results here describe the original experiment.

Import the reviewed file through Studio with the recorded request's settings.
`experiments/studio_smoke.py` runs as Studio's owning OS user in its Python
environment, using a short-lived local owner session without logging tokens.
Start-request IDs reconcile retries; use a new ID when intentionally changing
a failed request's settings.

On beast-3, `experiments/evaluate_smoke.py` takes the bundle directory; pass
`--adapter` with the saved adapter directory for the third condition. Remote
artifacts are under
`/home/tsilva/.local/share/llmstyler-unsloth/experiments/warm-direct-smoke/`.
Baseline and adapter evaluation services finished successfully and released
their processes. The fourteen-test repository suite and Ruff checks passed.

The next experiment should prioritize faithful, reviewed rewrites and a stronger
base model. Keep a prompt-only baseline and expand capability/format checks
before choosing a blend or increasing training duration.
