import torch
import numpy as np

import circuit_brain.utils.ridge_torch_utils as rtu
import circuit_brain.utils.ridge_np_utils as rnu


class RidgeCV:
    def __init__(self, lam_per_target=True, device="cuda"):
        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'."
        if device == "cpu":
            self.r2_score = rnu.R2
            self.r2r_score = rnu.R2r
            self._cvr = rnu.cross_val_ridge
            self.device = device
        else:
            self.r2_score = rtu.r2_score
            self.r2r_score = rtu.r2r_score
            self._cvr = rtu.cross_val_ridge
            self.device = torch.device(device)
        self.W = None

    def fit(self, x, y, n_splits=5, lams=[1e-3, 1e-4, 1e-5]):
        if self.device != "cpu":
            x, y = x.to(self.device), y.to(self.device)
        self.W = self._cvr(x, y, n_splits, lams)

    def predict(self, x):
        if self.W is None:
            raise ValueError("RidgeCV.fit needs to be run before calling inference.")
        if self.device == "cpu":
            return x.dot(self.W)
        x = x.to(self.W.device).float()
        return torch.matmul(x, self.W)
