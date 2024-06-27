from base import fMRIDataset

import h5py
import numpy as np
from pathlib import Path


class NarrativeReadListenfMRI(fMRIDataset):
    subject_idxs = list(range(1, 10))
    modes = ["reading", "listening"]
    splits = ["trn", "val"]

    def __init__(self, ddir, window_size):
        self.window_size = window_size
        self.ddir = Path(ddir)
        self.fmri_dir = self.ddir / "reponses"
        self.text_dir = self.ddir / "stimuli"
