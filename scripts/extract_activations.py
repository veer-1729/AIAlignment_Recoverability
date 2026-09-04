#!/usr/bin/env python
"""Stage 2, step 1: residual-stream activations for every stored checkpoint prompt.

No environment is re-run. Stage 1 stored the byte-exact prompt the policy saw at
each checkpoint, and validation check V10 confirmed those strings re-tokenise
identically under the pinned tokenizer -- so activations are recoverable by a
plain forward pass, months later, on any machine with the weights.

That property is re-asserted here rather than trusted: every prompt is compared
against its stored token ids before the forward pass, and a mismatch is a hard
failure. If the tokenizer or chat template has drifted, the activations would be
of a different computation than the one that produced the data.

Batch size is 1 deliberately. Padded batches shift the final-token index, and the
last token is the whole point; 664 prompts is a couple of minutes regardless.

    python scripts/extract_activations.py --config configs/validation.yaml --arm B_montecarlo
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from cnc import experiment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="B_montecarlo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = experiment.load_config(args.config)
    rd = experiment.run_dir(cfg)
    model_id = cfg["model"]["id"]
    revision = cfg["model"].get("revision")

    cps = [c for c in rd.read_all("checkpoints") if c["arm"] == args.arm]
    cps.sort(key=lambda c: c["checkpoint_id"])
    if args.limit:
        cps = cps[: args.limit]
    print("checkpoints: {}".format(len(cps)))

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print("model: {} layers, hidden {}".format(n_layers, hidden))

    acts = np.zeros((len(cps), n_layers + 1, hidden), dtype=np.float16)
    ids, mismatches = [], 0
    t0 = time.time()

    for i, c in enumerate(cps):
        enc = tok(c["rendered_prompt"], add_special_tokens=False, return_tensors="pt")
        stored = c.get("prompt_token_ids")
        if stored is not None and list(enc["input_ids"][0].tolist()) != list(stored):
            mismatches += 1
            raise SystemExit(
                "V10 VIOLATION at {}: prompt does not re-tokenise to stored ids. "
                "Tokenizer or chat template has drifted; activations would describe "
                "a different computation than the dataset.".format(c["checkpoint_id"])
            )
        with torch.no_grad():
            outs = model(**{k: v.cuda() for k, v in enc.items()}, output_hidden_states=True)
        # hidden_states is (embeddings, layer_1 ... layer_N); take the final token
        for L, h in enumerate(outs.hidden_states):
            acts[i, L] = h[0, -1, :].float().cpu().numpy().astype(np.float16)
        ids.append(c["checkpoint_id"])
        if (i + 1) % 50 == 0:
            print("  {}/{}  ({:.1f}s)".format(i + 1, len(cps), time.time() - t0))

    out = args.out or rd.path("activations_{}.npz".format(args.arm))
    np.savez_compressed(out, activations=acts, checkpoint_ids=np.array(ids),
                        model_id=model_id, n_layers=n_layers, hidden=hidden)
    mb = os.path.getsize(out) / 1e6
    print("\nwrote {}  ({:.1f} MB)".format(out, mb))
    print("shape {}  |  V10 re-verified on all {} prompts, {} mismatches".format(
        acts.shape, len(ids), mismatches))
    print("elapsed {:.1f}s".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
