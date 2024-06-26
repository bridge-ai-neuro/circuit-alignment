import argparse

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead 
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


