from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


def PCA_function(dataset: pd.DataFrame or np.array, n_components: int = 50, random_state: int = 0)\
        -> pd.DataFrame or np.array:
    return PCA(n_components=n_components, random_state=random_state).fit_transform(dataset)


def TSNE_function(dataset: pd.DataFrame or np.array, n_components: int = 2, random_state: int = 0,
                  perplexity: int = 30, PCA_before: bool = True, PCA_n_components: int = 50) \
        -> pd.DataFrame or np.array:
    if PCA_before:
        dataset = PCA_function(dataset, PCA_n_components, random_state=random_state)
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)\
        .fit_transform(dataset)
