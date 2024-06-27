from .base import fMRIDataset
from .hpot import HarryPotterfMRI
from .das import DAS


def get_dataset(dataset_id, ddir, **kwargs):
    if dataset_id == "hp":
        assert (
            kwargs.get("window_size", None) is not None
        ), "Must provide window size for Harry Potter dataset!"
        return HarryPotterfMRI(ddir, kwargs["window_size"])
    elif dataset_id == "das":
        return DAS(ddir)
    else:
        raise ValueError(f"Invalid dataset ID: {dataset_id}")


fMRIDataset.get_dataset = get_dataset
