import circuit_brain.utils as utils
from circuit_brain.model import BrainAlignedLMModel
from circuit_brain.dproc import fMRIDataset


m = BrainAlignedLMModel("gpt2-small")
das = fMRIDataset.get_dataset("das", "data/DS_data")
hp = fMRIDataset.get_dataset("hp", "data/HP_data", window_size=20)

# run all sentences from
