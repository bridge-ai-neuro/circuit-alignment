import re
from typing import List

from .base import fMRIDataset

import numpy as np
from pathlib import Path


class HarryPotterfMRI(fMRIDataset):
    subject_idxs = ["F", "H", "I", "J", "K", "L", "M", "N"]

    def __init__(self, ddir, window_size, remove_format_chars=True, remove_punc_spacing=True):
        self.window_size = window_size
        self.ddir = Path(ddir)
        self.fmri_dir = self.ddir / "fMRI"
        self.voxel_n = self.ddir / "voxel_neighborhoods"

        # load metadata
        self.words = np.load(self.fmri_dir / "words_fmri.npy")
        self.word_timing = np.load(self.fmri_dir / "time_words_fmri.npy")
        self.fmri_timing = np.load(self.fmri_dir / "time_fmri.npy")
        runs = np.load(self.fmri_dir / "runs_fmri.npy")

        # remove the edges of each run
        self.fmri_timing = np.concatenate(
            [self.fmri_timing[runs == i][20:-15] for i in range(1, 5)]
        )

        # load subject recordings
        self.subjects = [
            np.load(self.fmri_dir / f"data_subject_{i}.npy") for i in self.subject_idxs
        ]

        # normalize subject recordings
        for i in range(len(self.subjects)):
            self.subjects[i] = (
                self.subjects[i] - np.mean(self.subjects[i], axis=0)
            ) / np.std(self.subjects[i], axis=0)

        # for each fMRI measurement find its word context according to window size
        self.fmri_contexts = []
        for mri_time in self.fmri_timing:
            f = filter(lambda x: x[0] <= mri_time, zip(self.word_timing, self.words))
            m = map(lambda x: x[1], f)
            self.fmri_contexts.append(" ".join(list(m)[-self.window_size :]))

        if remove_format_chars:
            self.fmri_contexts = list(map(lambda x: re.sub(r"@|\+", "", x), self.fmri_contexts))

        if remove_punc_spacing:
            # unify all em-dash
            uni_em_d = lambda x: re.sub(r"--", "—", x)
            self.fmri_contexts = map(uni_em_d, self.fmri_contexts)

            # remove the spacing around a em-dash
            remove_em_spacing = lambda x: re.sub(r"\s*—\s*", "—", x)
            self.fmri_contexts = list(map(remove_em_spacing, self.fmri_contexts))

            # remove the spacing around ellipses 
            es_pat = r"\.\s+\."
            remove_ellipse_spacing = lambda x: re.sub(es_pat, "..", x)

            while any([re.search(es_pat, v) for v in self.fmri_contexts]):
                self.fmri_contexts = list(map(remove_ellipse_spacing, self.fmri_contexts))
            

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx], self.words

    def kfold(self, folds: int, trim: int):
        print(
            f"There are {len(self.fmri_timing)} fMRI measurements available.",
            f"Splitting {folds} folds means roughly {len(self.fmri_timing) // folds}",
            f"measurements / fold, and {len(self.words) // folds} words / fold.",
        )
        fold_size = len(self.fmri_timing) // folds
        assert 2 * trim <= fold_size

        def kfold_subject(idx):
            subject = self.subjects[idx]
            for f in range(folds):
                if f == 0:
                    start = 0
                else:
                    start = trim + fold_size * f
                if f == folds - 1:
                    end = len(self.fmri_timing)
                else:
                    end = fold_size * (f + 1) - trim

                train_st = max(start - trim, 0)
                train_ed = min(end + trim, len(self.fmri_timing))

                yield f, (self.fmri_contexts[start:end], subject[start:end]), (
                    np.concatenate(
                        [self.fmri_contexts[:train_st], self.fmri_contexts[train_ed:]]
                    ),
                    np.concatenate([subject[:train_st], subject[train_ed:]]),
                )

        return kfold_subject
