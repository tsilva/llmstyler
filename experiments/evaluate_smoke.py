"""Run on beast-3 with Studio's Python; keep outputs local to the experiment."""

# Unsloth must patch Transformers before Transformers is imported.
# ruff: noqa: I001

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from unsloth import FastLanguageModel

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    root = args.bundle
    spec = json.loads((root / "evaluation-spec.json").read_text())
    pin_path = root / "model-pin.json"
    model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
    else:
        pin = {"model_id": model_id, "revision": HfApi().model_info(model_id).sha}
        pin_path.write_text(json.dumps(pin, indent=2) + "\n")
    snapshot = snapshot_download(model_id, revision=pin["revision"])
    pin["snapshot_path"] = snapshot
    pin_path.write_text(json.dumps(pin, indent=2) + "\n")
    print("Pinned model:", pin["revision"], flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter) if args.adapter else snapshot,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    set_seed(spec["seed"])
    if not args.adapter:
        lengths = []
        for line in (root / "original-train.jsonl").read_text().splitlines():
            row = json.loads(line)
            lengths.append(len(tokenizer.apply_chat_template(row["messages"], tokenize=True)))
        stats = {"rows": len(lengths), "min_tokens": min(lengths), "max_tokens": max(lengths),
                 "over_512": sum(x > 512 for x in lengths), "over_2048": sum(x > 2048 for x in lengths),
                 "over_4096": sum(x > 4096 for x in lengths)}
        (root / "token-lengths.json").write_text(json.dumps(stats, indent=2) + "\n")
        print("Training length diagnostics:", stats, flush=True)
    conditions = ["adapter"] if args.adapter else ["base", "prompt_style"]
    versions = {p: importlib.metadata.version(p) for p in ["unsloth", "torch", "transformers", "trl", "peft"]}
    for condition in conditions:
        results = []
        for case in spec["cases"]:
            context = [m for m in case["messages"] if m["role"] != "system"]
            prefix = spec["style_messages"] if condition == "prompt_style" else [
                {"role": "system", "content": spec["base_system_prompt"]}
            ]
            inputs = tokenizer.apply_chat_template(
                prefix + context, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            tokens = inputs["input_ids"].shape[-1]
            if tokens + spec["generation"]["max_new_tokens"] > 4096:
                results.append({**case, "input_tokens": tokens, "skipped": "Exceeds 4096 context; not truncated"})
                continue
            start = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, **spec["generation"], use_cache=True)
            generated = output[0, tokens:]
            results.append({**case, "input_tokens": tokens, "output_tokens": len(generated),
                            "hit_token_limit": len(generated) == spec["generation"]["max_new_tokens"],
                            "response": tokenizer.decode(generated, skip_special_tokens=True),
                            "elapsed_seconds": round(time.monotonic() - start, 3)})
            print(condition, case["id"], "complete", flush=True)
        artifact = {"condition": condition, "model": pin, "adapter": str(args.adapter) if args.adapter else None,
                    "versions": versions, "generation": spec["generation"], "seed": spec["seed"], "results": results}
        (root / f"evaluation-{condition}.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
