from bioinformatica.source.preprocessing.correlation import filter_uncorrelated, filter_correlated_features
from bioinformatica.source.preprocessing.elaboration import balance, robust_zscoring, drop_constant_features
from bioinformatica.source.preprocessing.feature_selection import boruta
from bioinformatica.source.preprocessing.imputation import nan_check, nan_filter, imputation
from bioinformatica.source.datasets.loader import get_data
import pandas as pd
import numpy as np
from bioinformatica.source.type_hints import *


def epigenomic_preprocessing(dataset: pd.DataFrame, labels: np.array, p_value_threshold: float, min_correlation: float,
                             correlation_threshold: float) -> Tuple[pd.DataFrame, np.array]:
    if nan_check(dataset):
        dataset, labels = nan_filter(dataset, labels)
        dataset = imputation(dataset)

    dataset = drop_constant_features(dataset)
    dataset = robust_zscoring(dataset)

    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, min_correlation)
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)

    dataset = boruta(dataset, labels, 300, 0.05, 2)

    return dataset, labels


def pipeline(retrieve_parameters: Tuple[Tuple[Tuple[str, int, str], str], int]) -> Tuple[pd.DataFrame, np.array] or \
                                                                                   Tuple[np.array, np.array]:

    load_parameters, random_state = retrieve_parameters

    p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95

    dataset, labels = get_data(load_parameters)

    if load_parameters[-1] == 'epigenomic':
        dataset, labels = epigenomic_preprocessing(dataset, labels, p_value_threshold, min_correlation,
                                                   correlation_threshold)
    else:
        dataset, labels = get_data(load_parameters)
    return dataset, labels
