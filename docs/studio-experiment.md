> Current agent boundary: `experiments/studio_smoke.py` and the retired direct-OpenRouter
> judge return fail-closed diagnostics. They do not read owner storage, mint sessions
> or retrieve Infisical values. Use Studio's human sign-in flow. The trusted
> application server bridge remains an operator deployment, not an agent credential broker.

# First style-tuning experiment

The first milestone is a working loop from a portable voice to a small training
dataset, an adapter, and a comparison against the unchanged base model. A short
GPU smoke test confirms that the plumbing works. It does not establish that
fine-tuning improves style or preserves general capabilities.

See the [October 7 smoke experiment report](experiments/warm-direct-smoke-2026-10-07.md)
for the completed dataset review, Studio training, and three-condition comparison.

## Recipes

- Voice: `styles/warm-direct.yaml`.
- Data and training: `configs/pipelines/warm-direct-smoke.yaml`.
- Data engine: datamixxer, using a pinned SmolTalk2 dataset revision.
- Smoke base model: `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`.
- Proposed blend: restyle half the training rows. The first run retained 18 styled
  rows after fidelity screening, with 40 original rows.
- Test data: the untouched datamixxer test split, exported as `heldout.jsonl`.

The current recipe rewrites through AgentBridge at
`http://127.0.0.1:8082/api/v1` with `codex/gpt-6.1-sol` and high reasoning.
It needs a running AgentBridge with Codex authentication, with no OpenRouter key
on the client. The completed October 7 experiment used direct OpenRouter
`x-ai/grok-4.3`; its report records those original settings. Changing the teacher
invalidates old rewrite checkpoints and creates a new dataset generation pass.
The historical `experiments/review_smoke.py` judge is retired and returns a
fail-closed diagnostic. A replacement evaluation workflow needs separate review;
the teacher-as-judge description applies only to the completed original run.

Use full conversation history for rewriting. Review the generated preview before
training, especially code, numbers, citations, and caveats. The first run used
`experiments/review_smoke.py` to revert rejected conversations before upload.
This experiment-specific gate uses the teacher itself as a judge and requires
human review; it is not a validated quality metric. The rewrite prompt
instructs the teacher to preserve them; it does not prove preservation.

## Beast-3 environment

The machine has an NVIDIA RTX 4090 with 24 GB VRAM. Studio is installed separately
from either repository under
`/home/tsilva/.local/share/llmstyler-unsloth/studio`. Installation logs and package
constraints live in the parent directory. Follow the official installation
instructions at https://unsloth.ai/docs/new/studio/install when updating.

Verified on October 7, 2026: Unsloth and Unsloth Zoo `2026.10.2`, Python 3.12,
PyTorch `2.8.0`, and CUDA available on the RTX 4090. The official installer
initially selected an older Unsloth release under the package-age cutoff; the
latest requested Unsloth packages were then upgraded explicitly, retaining the
seven-day cutoff for other Python dependencies. A complete package snapshot is
saved in `installed-requirements.txt`, and selected versions in `environment.json`.

Open [Unsloth Studio](https://beast3-unsloth.tsilva.eu/) directly, or use its tile
on the [Beast 3 launcher](https://beast-3.tsilva.eu/). The hostname uses the
existing `beast3` Cloudflare Tunnel and the shared **Beast 3 services — owner
only** Access application. Its existing owner policy and login settings also
apply to Studio. Tunnel-side Access JWT validation is enabled for that
application. A local sign-in bridge verifies the Access JWT signature, issuer,
application audience, expiry, and exact owner email, then issues a normal Studio
session. Opening the Cloudflare hostname signs you in automatically. Studio's
password and direct local API authentication remain intact for fallback access.

The route is Cloudflare → the existing tunnel → Caddy on `127.0.0.1:18083` →
Studio on `127.0.0.1:8888`. Browser navigation and `/cloudflare-auth/*` use the
sign-in bridge on `127.0.0.1:8889`; API requests, assets, and WebSockets continue
directly to Studio. Caddy forwards HTTPS protocol information and handles
WebSockets automatically. All three origin ports listen only on loopback.
The Caddy route is saved in `/etc/caddy/Caddyfile`; the launcher entry is in
`/var/lib/caddy/beast3-entry/index.html`. Both files were backed up before editing.

Studio starts automatically through the user systemd unit
`/home/tsilva/.config/systemd/user/unsloth-studio.service`. User lingering was
already enabled on beast-3, allowing the service to run without an SSH session.
Use these commands on beast-3 to check or restart it:

```bash
systemctl --user status unsloth-studio.service
journalctl --user -u unsloth-studio.service -n 100
systemctl --user restart unsloth-studio.service
```

The bridge is managed separately by `unsloth-cloudflare-login.service`, using
Studio's existing Python environment. Its deployed source is
`/home/tsilva/.local/share/llmstyler-unsloth/bridge.py`; the maintained source and
security tests are in `deploy/unsloth-cloudflare/`. The unit pins the public host,
Access issuer, application audience, and owner email. It holds no password or
long-lived API key. Login exchanges require a same-origin POST with a custom
header, and responses are never cached or logged. Normal Studio session tokens
are stored by its frontend and refreshed by its existing auth flow.

This is a local integration, not an upstream no-auth setting. It does not modify
Unsloth's installed files. After upgrading Studio, run the bridge tests in
Studio's Python environment and verify automatic sign-in: the bridge depends on
Studio's auth helpers, HTML entry format, and frontend storage keys. It refuses
to serve the frontend if the entry format changes. To revert, restore the Caddy
backup named `Caddyfile.before-studio-autologin-*`, validate and reload Caddy,
then disable the bridge service; Studio returns to its existing password login.

Restart only when no training or inference job is running. Save the selected
model, Studio/Unsloth versions, dataset manifest, training settings, and final
metrics with each experiment so it can be repeated. Keep experiment datasets
local to beast-3 unless deliberately publishing them.

The service uses this launch command (do not launch a second instance while the
service is running):

```bash
UNSLOTH_STUDIO_HOME=/home/tsilva/.local/share/llmstyler-unsloth/studio \
  /home/tsilva/.local/share/llmstyler-unsloth/studio/bin/unsloth studio \
  --host 127.0.0.1 --port 8888
```

For a direct LAN fallback, create an SSH tunnel from the Mac when one is not
already running:

```bash
ssh -N -o ProxyCommand=none -o Hostname=beast-3.local \
  -L 127.0.0.1:18888:127.0.0.1:8888 beast-3.local
```

Open http://127.0.0.1:18888 for that fallback. The SSH forwarding session is
temporary; Studio itself is managed by systemd. A fresh Studio installation
requires completing its initial password setup yourself.

The existing Trump-style dataset is also exported as a compatibility smoke
bundle under `runs/legacy-trump-studio-smoke/` and copied to
`/home/tsilva/.local/share/llmstyler-unsloth/experiments/legacy-trump-studio-smoke/`.
It contains 64 styled training conversations and 100 untouched held-out
conversations. It tests the old dataset's Studio handoff; it is separate from
the warm-direct experiment and does not establish warm-direct voice quality.

## Evaluation

Compare the same held-out user prompts under three conditions: the base model,
the base model with the compiled style pack, and the trained adapter. Keep model
generation settings the same. Assess voice adherence and answer quality
separately. Include short factual answers, uncertainty, explanations, code, and
multi-turn conversations. A lower training loss alone is not evidence of a
better voice.

After the smoke test, use a larger base model and enough reviewed examples to
compare several restyle proportions. Choose the proportion from observed style
and capability results rather than treating the smoke recipe's 50% as a default.
