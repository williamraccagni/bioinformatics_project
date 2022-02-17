from bioinformatica.source.preprocessing.feature_selection import *
from bioinformatica.source.datasets.loader import *
from bioinformatica.source.preprocessing.imputation import imputation


def test_boruta():
    parameters = ('GM12878', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)

    dataset = imputation(dataset)
    dataset = dataset.head(200)
    labels = labels[:200]

    dataset = boruta(dataset, labels, 10, 0.01, 42)
    assert dataset is not None, 'error while executing boruta'
