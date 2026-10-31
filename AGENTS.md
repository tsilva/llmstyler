# Agent operation

Use `uv run --frozen python scripts/agent.py run ...` for agent-operated
LLMStyler commands. `scripts/agent.py check` validates the local policy without
contacting a service. The ordinary CLI and SDK are separate operator interfaces;
do not use them to bypass the agent boundary.

Agent mode uses the existing keyless AgentBridge destination
`http://127.0.0.1:8082/api/v1`. It strips inherited service credentials and proxy
settings, skips dotenv loading, and rejects alternate destinations, redirects
and explicit authentication headers. It is application policy, not OS isolation.
Keep model selection, prompts, reasoning and cache fingerprints independent of
the agent-mode marker.

Hugging Face access/publishing, external datamixxer execution and remote Runbook
training have no reviewed agent credential transport here and must fail closed
in this mode. Do not fetch raw keys, read token caches, reuse another project's
broker, invent service folders, or use the deprecated Infisical path to bypass
that restriction. `docs/legacy-infisical-metadata.json` is historical inventory,
not an operational native proxy binding.

The Studio smoke and retired direct-OpenRouter review scripts return explicit
diagnostics. Agents must not read Studio owner storage or mint owner sessions.
Trusted Studio application-server authentication is separate; use fake inputs
for code tests and leave real sign-in, enrollment, deployment and training to
separately authorized work.

## Validation

Preserve the frozen dependency graph and seven-day release hold. Relevant CI uses
Python 3.11 and 3.14, Ruff, pytest, the full locked dependency audit, offline CLI
and import checks, and `uv build`. Keep validation outputs outside the checkout
and omit local dotenv, credentials, private datasets and run outputs from test
copies. Do not call the real gateway, Hub, W&B, Modal or Studio to validate this
boundary. Update README.md when changing the agent entrypoint or its behavior.
