#!/usr/bin/env -S uv run --quiet --with torch --with transformers --python 3.12 python
"""E3 item 3 — cross-check llama.cpp/q8_0 vectors against the reference
HF transformers implementation in fp32.

[B] in e3_pooling.zig already proves the pooling *type* is honoured. This adds
the other half: that llama.cpp's forward pass plus q8_0 quantization stay
numerically faithful to the reference. A pass here means retrieval quality is
not being eroded by something subtler than a wrong pooling mode.

Reads /tmp/zkb-e3-vectors.json (written by `zig build e3`).
Threshold: cos > 0.99 — quantization noise for q8_0 should be well inside that.
"""

import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-Embedding-0.6B"
THRESHOLD = 0.99

with open("/tmp/zkb-e3-vectors.json") as f:
    data = json.load(f)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

print(f"E3 item 3 — {data['model']} (llama.cpp) vs {MODEL} (transformers fp32)\n")

failures = 0
for item in data["vectors"]:
    batch = tok(item["text"], return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    # Qwen3-Embedding pools the last token. No padding here (batch of 1), so
    # index -1 is the real final token.
    h = out.last_hidden_state[0, -1]
    h = torch.nn.functional.normalize(h, dim=-1)

    ref = torch.tensor(item["vec"], dtype=torch.float32)
    ref = torch.nn.functional.normalize(ref, dim=-1)

    cos = torch.dot(h, ref).item()
    ok = cos > THRESHOLD
    if not ok:
        failures += 1
    print(f"  cos={cos:.6f}  {'PASS' if ok else 'FAIL'}  {item['text'][:46]}")

print(f"\n{'E3-3 PASS' if failures == 0 else 'E3-3 FAIL'}: {failures} failure(s)")
sys.exit(1 if failures else 0)
