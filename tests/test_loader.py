from math import ceil
from bioinformatica.source.datasets.loader import *


def test_get_data():
    parameters = (('K562', 200, 'enhancers'), 'epigenomic')
    dataset, labels = get_data(parameters)
    assert dataset is not None, 'error while loading dataset'
    assert labels is not None, 'error while loading labels'
    assert isinstance(dataset, pd.DataFrame), 'error while loading dataset'
    assert isinstance(labels, np.ndarray), 'error while loading labels'


def test_get_holdouts():
    parameters = (('K562', 200, 'enhancers'), 'epigenomic')
    dataset, labels = get_data(parameters)
    dataset = dataset.head(100)
    labels = labels[:100]
    holdout_parameters = 1, 0.2, 1
    n_split, test_size, random_state = holdout_parameters
    for training, test in get_holdouts(dataset, labels, holdout_parameters, parameters[-1]):
        training_set, training_labels = training
        test_set, test_labels = test
        assert len(test_set) == ceil(len(dataset) * test_size), 'error while creating training and test set'
        assert len(dataset) == len(training_set) + len(test_set), 'error while creating training and test set'
