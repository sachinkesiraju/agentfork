"""Worker used by ``test_kv_transfer.py`` to decode a transferred cache in a fresh process."""

from __future__ import annotations

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(cache_path: str, context_len: int, question: str) -> None:
    from agentfork.kvzip_router import CompressedKVCache

    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    with open(cache_path, "rb") as f:
        compressed = CompressedKVCache.from_bytes(f.read())

    question_ids = tokenizer(question, return_tensors="pt").input_ids
    pos = context_len
    with torch.no_grad():
        out = model(
            input_ids=question_ids,
            past_key_values=compressed.cache,
            position_ids=torch.arange(pos, pos + question_ids.shape[-1]).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    probs = torch.softmax(out.logits[:, -1, :], dim=-1)
    top5 = torch.topk(probs, k=5)
    result = {
        "top_ids": top5.indices[0].tolist(),
        "top_tokens": tokenizer.batch_decode(top5.indices[0].unsqueeze(-1)),
        "top_probs": top5.values[0].tolist(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
