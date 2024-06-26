from typing import List
from torch.utils.data import Dataset


class fMRIDataset(Dataset):
    def __init__(self):
        self.recordings = None
        self.text = None
        self.window_size = None
        self.ddir = None

    def __len__(self):
        raise NotImplemented()

    def __getitem__(self, idx):
        raise NotImplemented()

    def kfold(self, folds: int, trim: int):
        raise NotImplemented()
