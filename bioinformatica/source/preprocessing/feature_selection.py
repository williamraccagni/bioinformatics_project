import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import cpu_count


def boruta(dataset: pd.DataFrame, labels: np.array, max_iter: int, p_value_threshold: float, random_state: int) \
        -> pd.DataFrame:
    forest = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5)
    boruta_selector = BorutaPy(
        forest,
        n_estimators='auto',
        verbose=2,
        alpha=p_value_threshold,
        max_iter=max_iter,
        random_state=random_state
    )
    boruta_selector.fit(dataset.values, labels)
    return dataset[dataset.columns[np.where(boruta_selector.support_ == True)]]

