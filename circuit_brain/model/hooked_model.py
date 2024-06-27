from typing import Union, List, Optional

import torch
import numpy as np

import transformer_lens as tl
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from rich.progress import track


torch.set_grad_enabled(False)


class BrainAlignedLMModel:
    def __init__(self, hf_model_id):
        self.ht = HookedTransformer.from_pretrained(hf_model_id)

    def to_tokens(self, sentences: List[str]):
        return self.ht.to_tokens(sentences)

    def to_string(self, tokens: List[int]):
        return self.ht.to_string(tokens)

    def run_with_cache(self, tokens, batch_size=8):
        tok_batch = tokens.chunk(len(tokens) // batch_size)

        logits = []
        caches = []

        for toks in track(tok_batch, description="Infer w/ cache..."):
            l, c = self.ht.run_with_cache(toks)
            l, c = l.to("cpu"), c.to("cpu")  # free gpu memory
            logits.append(l)
            caches.append(c)

        agg_logits = torch.cat(logits, dim=0)
        agg_cache_dict = {}
        for k in caches[0].keys():
            agg_cache_dict[k] = torch.cat([c[k] for c in caches], dim=0)

        agg_cache = ActivationCache(agg_cache_dict, self.ht)

        return agg_logits, agg_cache 

        
    def resid_post(
        self,
        cache: ActivationCache,
        avg: bool = True,
        chunk: bool = False,
        chunk_size: int = 4,
        apply_ln: bool = True,
    ):
        if avg and chunk:
            raise ValueError(
                "Pooling schemes average and chunking cannot be used at the same time!"
            )
        accum_resid = cache.accumulated_resid(
            apply_ln=apply_ln
        )  # layer, batch, pos, d_model

        if avg:
            return torch.mean(accum_resid, axis=2)
        if chunk:
            pos_chunked = accum_resid.chunk(chunk_size, axis=2)
            return torch.cat([torch.mean(c, axis=2) for c in pos_chunked], dim=-1)
        return accum_resid
