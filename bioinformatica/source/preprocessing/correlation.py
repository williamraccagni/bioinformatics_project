import pandas as pd
import numpy as np
from minepy import MINE
from scipy.stats import pearsonr, spearmanr, entropy


tests = [
    pearsonr,
    spearmanr
]


def uncorrelated_test(dataset: pd.DataFrame, labels: np.array, p_value_threshold: float) -> list:
    scores = []
    for test in tests:
        score = {}
        for feature, x in dataset.items():
            score[feature] = test(x.values.ravel(), labels)
        scores.append(score)
    uncorrelated = set()
    for score in scores:
        for key in score:
            if score.get(key)[1] > p_value_threshold:
                uncorrelated.add(key)
    return list(uncorrelated)


def mic(dataset: pd.DataFrame, labels: np.array) -> dict:
    score = {feature: None for feature in dataset}
    for feature, x in dataset.items():
            mine = MINE()
            mine.compute_score(x.values.ravel(), labels)
            score[feature] = mine.mic()
    return score


def filter_uncorrelated(dataset: pd.DataFrame, labels: np.array, p_value_threshold: float, correlation_threshold: float)\
        -> pd.DataFrame:
    uncorrelated_features = uncorrelated_test(dataset, labels, p_value_threshold)
    mic_score = mic(dataset[uncorrelated_features], labels)
    for key in mic_score:
        if mic_score.get(key) >= correlation_threshold:
            uncorrelated_features.remove(key)
    return dataset.drop(columns=uncorrelated_features)


def filter_correlated_features(dataset: pd.DataFrame, p_value_threshold: float, correlation_threshold: float) \
        -> pd.DataFrame:
    features = feature_correlation(dataset)
    to_drop = []
    for indices, correlation, p_value in features:
        first, second = [int(v) for v in indices.split(' ')]
        if p_value < p_value_threshold and correlation > correlation_threshold:
            if entropy(dataset[dataset.columns[first]]) > entropy(dataset[dataset.columns[second]]):
                to_drop.append(second)
            else:
                to_drop.append(first)
    dataset.drop(dataset.columns[list(set(to_drop))], axis=1, inplace=True)
    return dataset


def feature_correlation(dataset: pd.DataFrame) -> list:
    score = []
    for i in range(len(dataset.columns)):
        for j in range(i + 1, len(dataset.columns)):
            if ' '.join([str(x) for x in sorted((i, j))]) not in [score[0] for score in score]:
                correlation, p_value = pearsonr(dataset[dataset.columns[i]].values.ravel(),
                                                dataset[dataset.columns[j]].values.ravel())
                correlation = np.abs(correlation)
                score.append((' '.join([str(x) for x in sorted((i, j))]), correlation, p_value))
    return score
