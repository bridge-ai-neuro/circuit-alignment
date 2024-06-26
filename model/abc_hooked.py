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


class BrainAlignedLMModel:
    def __init__(self, hf_model_id):
        self.model = HookedTransformer.from_pretrained(hf_model_id)

   def to_tokens(self, sentences: Union[List[str], List[List[str]]], stagger=False: Optional[bool]):
        """Tokenize given sentences using model-specified scheme with the
        appropriate attention mask scaled to the length of the sentences.
        Padding on the right is performed on all sentences.

        sentences (Union[List[str], List[List[str]]):  a list of sentences or a
        list of sentences that has already been separated word-wise
        
        stagger (Optional[bool]): if true, for each sentence every additional
        word is tokenized as a separate sentence, manifest this as separate
        attention masks.
        """
        if not stagger:
             
