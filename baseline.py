import sys
import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignedLMModel
from circuit_brain.dproc import fMRIDataset

import torch
import numpy as np
from rich.progress import track
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True, help="hf_id of model")
parser.add_argument(
    "--batch-size", type=int, required=True, help="batch size during inference"
)
parser.add_argument(
    "--only-store-repr", action="store_true", default=False,
    help="don't compute alignment only store representations"
)
parser.add_argument(
    "--repr_fname", type=str, required=False,
    help="file name for all representations"
)
parser.add_argument(
    "--window-size", type=int, required=True,
    help="context window for each fmri recording"
)
parser.add_argument(
    "--remove-format-chars", default=False, action="store_true",
    help="remove formatting characters in text"
)
parser.add_argument(
    "--remove-punc-spacing", default=False, action="store_true",
    help="remove punctuation spacing in text"
)
parser.add_argument(
    "--uniform-lam", default=False, action="store_true",
    help="do not use lam per voxel"
)
options = parser.parse_args()


if __name__ == "__main__":
    m = BrainAlignedLMModel(options.model_name)
    hp = fMRIDataset.get_dataset(
        "hp",
        "data/HP_data",
        window_size=options.window_size,
        remove_format_chars=options.remove_format_chars,
        remove_punc_spacing=options.remove_punc_spacing
    )
    ff = f"{int(options.remove_format_chars)}{int(options.remove_punc_spacing)}"
    if options.repr_fname is None:
        model_repr = utils.per_subject_model_repr(hp.fmri_contexts, m, options.batch_size) 
    else:
        model_repr = pickle.load(open(options.repr_fname, "rb"))

    if options.only_store_repr:
        pickle.dump(
            subject_model_repr,
            open(f"data/base_align_data/{options.model_name}-repr.pkl", "wb+")
        )
        sys.exit(0)
    
    # compute brain-alignment scores
    print(options.uniform_lam)
    ridge_cv = utils.RidgeCV(n_splits=5, lam_per_target=not options.uniform_lam)
    pickle.dump(
        utils.across_subject_alignment(hp, model_repr, 5, 10, ridge_cv),
        open(f"data/base_align_data/{options.model_name}-br2-{ff}.pkl", "wb+"),
    )
