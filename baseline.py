import pickle
import argparse

import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignedLMModel
from circuit_brain.dproc import fMRIDataset

import numpy as np
from rich.progress import track
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True, help="hf_id of model")
options = parser.parse_args()


m = BrainAlignedLMModel(options.model_name)
das = fMRIDataset.get_dataset("das", "data/DS_data")
hp = fMRIDataset.get_dataset("hp", "data/HP_data", window_size=20)

# check brain alignment on hp dataset
hp_folds = hp.kfold(5, 5)


def per_subject_alignment(subject_kfold, repr_cache=dict()):
    use_cache = False if len(repr_cache.keys()) == 0 else True
    fold_r2 = []
    for k, (test_sents, test_fmri), (train_sents, train_fmri) in subject_kfold:
        if use_cache:
            train_repr, test_repr = repr_cache[k]
        else:
            train_toks = m.to_tokens(train_sents)
            test_toks = m.to_tokens(test_sents)

            _, train_cache = m.run_with_cache(train_toks)
            _, test_cache = m.run_with_cache(test_toks)

            train_repr = m.resid_post(train_cache)
            test_repr = m.resid_post(test_cache)
            repr_cache[k] = (train_repr, test_repr)
        layer_r2 = []
        for l in track(
            range(len(train_repr)), description=f"f{k+1} Align across layers..."
        ):
            # pca = PCA(n_components=100)
            train_repr_l, test_repr_l = train_repr[l].numpy(), test_repr[l].numpy()
            # weights, _ = utils.cross_val_ridge(pca.fit_transform(train_repr_l), train_fmri)
            weights, _ = utils.cross_val_ridge(train_repr_l, train_fmri)
            # print("%Explained variance:", np.sum(pca.explained_variance_ratio_))
            layer_r2.append(
                utils.R2r((test_repr_l).dot(weights), test_fmri)
                # utils.R2r(pca.transform(test_repr_l).dot(weights), test_fmri)
            )
        fold_r2.append(layer_r2)
    return fold_r2, repr_cache


subject_r2 = []
cache = dict()
for sidx in range(len(hp)):
    if sidx == 0:
        sidx_r2, cache = per_subject_alignment(hp_folds(sidx))
    else:
        sidx_r2, cache = per_subject_alignment(hp_folds(sidx), cache)
    subject_r2.append(sidx_r2)
pickle.dump(subject_r2, open(f"{options.model_name}-base-alignment.pkl", "wb+"))
