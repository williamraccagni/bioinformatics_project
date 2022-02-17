from epigenomic_dataset import load_epigenomes
from sklearn.model_selection import StratifiedShuffleSplit
from bioinformatica.source.type_hints import *
import pandas as pd
import numpy as np


def load_dataset(data_parameters: Tuple[str, int, str]) -> Tuple[pd.DataFrame, np.array]:
    cell_line, window_size, epigenomes_type = data_parameters
    epigenomes, labels = load_epigenomes(
        cell_line=cell_line,
        dataset='fantom',
        regions=epigenomes_type,
        window_size=window_size
    )
    labels = labels.values.ravel()
    return epigenomes, labels


def holdouts(holdout_parameters: Tuple[int, float, int]) -> StratifiedShuffleSplit:
    n_split, test_size, random_state = holdout_parameters
    return StratifiedShuffleSplit(n_splits=n_split, test_size=test_size, random_state=random_state)