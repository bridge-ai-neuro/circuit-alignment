from .base import fMRIDataset

import pandas as pd
from pathlib import Path


class DAS(fMRIDataset):
    def __init__(self, ddir):
        self.ddir = Path(ddir)
        self.df = pd.read_csv(self.ddir / "brain-lang-data_participant_20230728.csv")

    @property
    def base_sents(self):
        return self.df[self.df["cond"] == "B"].sentence.unique()

    @property
    def drive_sents(self):
        return self.df[self.df["cond"] == "D"].sentence.unique()

    @property
    def supp_sents(self):
        return self.df[self.df["cond"] == "S"].sentence.unique()
