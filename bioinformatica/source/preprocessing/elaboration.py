import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from bioinformatica.source.type_hints import *


balances = {
    'under_sample': RandomUnderSampler,
    'over_sample': RandomOverSampler,
    'SMOTE': SMOTE
}


def drop_constant_features(dataset: pd.DataFrame) -> pd.DataFrame:
    non_const_features = [feature for feature in dataset.columns if dataset[feature].nunique() > 1]
    return dataset[non_const_features]


def balance(dataset: pd.DataFrame or np.array, labels: np.array, random_state: int, type_of_balance: str, data_type: str)\
        -> Tuple[pd.DataFrame, np.array] or Tuple[np.array, np.array]:
    if data_type == 'sequences':
        resample_dataset = []
        for sample in dataset:
            resample_dataset.append(sample.ravel())
        resample_dataset = np.array(resample_dataset)
        sampler = balances.get(type_of_balance)(random_state=random_state)
        dataset, labels = sampler.fit_resample(resample_dataset, labels)
        dataset = []
        for sample in resample_dataset:
            dataset.append(np.reshape(sample, (-1, 4)))
        dataset = np.array(dataset)
    else:
        sampler = balances.get(type_of_balance)(random_state=random_state)
        dataset, labels = sampler.fit_resample(dataset, labels)
    return dataset, labels


def robust_zscoring(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(dataset.values),
        columns=dataset.columns,
        index=dataset.index
    )