from bioinformatica.source.preprocessing.correlation import *
from bioinformatica.source.datasets.loader import *
from bioinformatica.source.preprocessing.imputation import imputation


def test_filter_uncorrelated():
    parameters = ('GM12878', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = imputation(dataset)
    dataset = dataset.head(5)
    labels = labels[:5]
    p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95

    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, correlation_threshold)
    assert isinstance(dataset, pd.DataFrame), 'error during non correlation filtering'
    return dataset


def test_filter_correlated_features():
    parameters = ('GM12878', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = imputation(dataset)
    assert dataset is not None, 'error while imputing dataset'
    dataset = dataset.head(100)
    p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)
    assert isinstance(dataset, pd.DataFrame), 'error during correlation filtering'


