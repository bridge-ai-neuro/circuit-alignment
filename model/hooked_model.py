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
    ActivationCache
)


torch.set_grad_enabled(False)


class BrainAlignedLMModel:
    def __init__(self, hf_model_id):
        self.model = HookedTransformer.from_pretrained(hf_model_id)
    
    def to_tokens(self, sentences: List[str]):
        return self.model.to_tokens(sentences)

    def to_string(self, tokens: List[int]):
        return self.model.to_string(tokens)

    def resid_post(self, tokens:List[int], avg:bool=True, chunk:bool=False, chunk_size:int=4, apply_ln:bool=True):
        """Runs the model on the given tokens and return all of the activations
        after the residual connection. If `avg=True`, then for each layer, we
        average across the token dimension. Otherwise, if `chunk=True`, we block
        the tokens according to the `chunk_size`, average the embeddings within
        each chunk and then concatenate these embeddings together. If
        `apply_ln=True`, then normalize."""
        if avg and chunk:
            raise ValueError("Pooling schemes average and chunking cannot be used at the same time!")
        logits, cache = self.model.run_with_cache(tokens)
        accum_resid = cache.accumulated_resid(apply_ln=apply_ln)  # layer, batch, pos, d_model

        if avg:
            return torch.mean(accum_resid, axis=2)
        if chunk:
            pos_chunked = accum_resid.chunk(chunk_size, axis=2)
            return torch.cat([
                torch.mean(c, axis=2) for c in pos_chunked
            ], dim=-1)
        return accum_resid
            

